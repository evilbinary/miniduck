#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""body/tools/solve_stance.py — 用 MuJoCo 求解静态可站立标称姿态。

为什么需要：RL 里 `target = default_pos + scale·action`，零动作即回到标称姿态。
若标称姿态本身站不住（实测膝弯曲 0.55 rad 时质心前移，0.4 s 内前倾倒地），
每个 episode 开局就摔，训练信号极差。标称姿态应当是**静态平衡姿态**。

做法：对腿部的 (hip_pitch, knee_pitch) 做网格搜索，踝关节取 -(hip+knee)
（近似保持脚掌贴地），每个候选姿态在 MuJoCo 里静置 N 秒，按
「是否倒下 + 高度偏差 + 剩余动能」打分，输出最优姿态与 YAML 片段。

用法：
    python body/tools/solve_stance.py                      # 两版都解
    python body/tools/solve_stance.py --robot miniduck-S --seconds 1.5
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOT_YAML = ROOT / "body" / "robot.yaml"

sys.path.insert(0, str(ROOT / "train" / "envs"))          # 复用执行器模型层
from actuators import (  # noqa: E402
    build_bam_controller,
    hold_pose_steps,
    sync_bam_target_state,
)
from duck_env import align_to_ground  # noqa: E402


def leg_joints(robot: dict) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """识别腿部的 hip/knee/ankle 关节名与下标。"""
    names = [j["name"] for j in robot["joints"]]
    hips = [n for n in names if n.endswith("hip_pitch")]
    knees = [n for n in names if n.endswith("knee_pitch")]
    ankles = [n for n in names if n.endswith("ankle_pitch")]
    if not (len(hips) == len(knees) == len(ankles)) or not hips:
        raise SystemExit(f"腿部关节不规整：hips={hips} knees={knees} ankles={ankles}")
    return hips, knees, ankles, {n: i for i, n in enumerate(names)}


def joint_addrs(m, mujoco, joint_names: list[str]):
    """robot.yaml 关节顺序 → (qpos 地址, actuator id)；找不到填 None。"""
    qadr, act = [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr.append(int(m.jnt_qposadr[jid]) if jid >= 0 else None)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        act.append(int(aid) if aid >= 0 else None)
    return qadr, act


def spawn_z_estimate(robot: dict) -> float:
    """按腿部运动链估算出生高度（躯干离地）。

    不要依赖 robot.yaml 里的 base_height_m：它是"上次求解的结果"，
    几何一改就过期；用过期值出生会让机器人先自由落体再着地，
    求解器会把"摔"误判成"姿态不稳"（实测 L 版就这样误报过）。
    估算 = 沿腿链的 |z 偏移| 之和 + 脚掌厚度余量。
    """
    leg = [j for j in robot["joints"] if j.get("role") == "leg"]
    # 只沿**一条**链路求和：左右腿是镜像的，全加起来会翻倍
    left = [j for j in leg if j["name"].startswith("L_")]
    chain = left or leg
    total = sum(abs(float((j.get("xyz") or [0.0, 0.0, 0.0])[2])) for j in chain)
    return max(0.03, total + 0.006)


def simulate(
    m, d, pose, qadr, act, seconds, mujoco, torso_id, robot,
    actuator_model="pd", bam_controller=None, spawn_z: float | None = None,
) -> tuple[float, float, float]:
    """把关节设到 pose 静置 seconds 秒，返回 (最终倾斜°, 躯干高度 mm, 动能 J)。"""
    mujoco.mj_resetData(m, d)
    for i, adr in enumerate(qadr):
        if adr is not None:
            d.qpos[adr] = pose[i]
    d.ctrl[:] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    # ★ 出生方式必须与运行时（env / check_physics）完全一致：脚底贴地对齐。
    # 曾经用"按运动链估算出生高度"覆盖 qpos[2]，导致脚先穿地、被接触约束弹出，
    # 落到另一个构型后误判为"稳定"（实测把脚在质心前方 5 cm 的姿态标成了稳定解）。
    align_to_ground(m, d, mujoco)

    # 力矩执行器：必须按力矩语义驱动（PD 或 BAM），不能直接写 ctrl=角度
    prof = robot["_servo_profile"]
    if bam_controller is not None:
        sync_bam_target_state(bam_controller, pose)
    hold_pose_steps(
        m, d, pose, [j["name"] for j in robot["joints"]], prof, mujoco,
        int(seconds / m.opt.timestep),
        actuator_model=actuator_model,
        bam_controller=bam_controller,
    )

    quat_w = float(d.qpos[3])
    tilt = float(np.degrees(2 * np.arccos(np.clip(abs(quat_w), -1.0, 1.0))))
    height = float(d.xpos[torso_id][2]) * 1000.0 if torso_id >= 0 else float(d.qpos[2]) * 1000.0
    ke = 0.5 * float(np.sum(m.body_mass * np.sum(d.cvel**2, axis=1)))
    return tilt, height, ke


def solve(robot_name: str, cfg: dict, seconds: float, step: float, actuator_model: str = "pd") -> dict | None:
    import mujoco

    robot = dict(cfg["robots"][robot_name])
    robot["_servo_profile"] = cfg["servo_profiles"][robot["servo"]]
    m = mujoco.MjModel.from_xml_path(
        str(ROOT / cfg["generate"]["mjcf"]["target"].format(robot=robot_name))
    )
    d = mujoco.MjData(m)
    joints = robot["joints"]
    names = [j["name"] for j in joints]
    hips, knees, ankles, idx = leg_joints(robot)
    qadr, act = joint_addrs(m, mujoco, names)
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    h_ref = float(robot["geometry"]["base_height_m"])

    bam_ctrl = None
    if actuator_model == "bam":
        bam_ctrl = build_bam_controller(
            robot["_servo_profile"], m, d,
            [j["name"] for j in joints if not j.get("passive")],
        )

    z_est = spawn_z_estimate(robot)
    print(f"  按运动链估算出生高度 z={z_est * 1000:.1f} mm"
          f"（robot.yaml 现值 {h_ref * 1000:.1f} mm，若差距大说明需回填）")

    # 搜索范围由关节实际限位推导，**不要硬编码**：
    # 实测 L 版（矮胖短腿）的稳定区在 knee≈1.2~1.4 rad，硬编码 0~1.0 会
    # 直接漏掉整个可行域，得出"站不住"的错误结论。
    hip_lim = [joints[idx[n]] for n in hips]
    knee_lim = [joints[idx[n]] for n in knees]
    h_lo = max(j["lo"] for j in hip_lim)
    h_hi = min(j["hi"] for j in hip_lim)
    k_lo = max(j["lo"] for j in knee_lim)
    k_hi = min(j["hi"] for j in knee_lim)
    grid_h = np.arange(h_lo, h_hi + 1e-9, step)
    grid_k = np.arange(k_lo, k_hi + 1e-9, step)
    print(f"  搜索范围（按限位）：hip {h_lo:+.2f}~{h_hi:+.2f}，knee {k_lo:+.2f}~{k_hi:+.2f}，"
          f"步长 {step}")
    # 踝关节补偿族：保持脚掌贴地需要"髋+膝"的补偿，但符号约定因机型而异
    # （实测：只试 ankle=-(hip+knee) 会整族落在"脚不平"的姿态上，L 版因此假报无解）。
    # 这里显式枚举几种常见约定，代价仅 4 倍。
    def ankle_candidates(hip: float, knee: float) -> list[float]:
        return [-(hip + knee), -(hip - knee), -hip + knee, 0.0]

    results = []
    for hip, knee in itertools.product(grid_h, grid_k):
        for ankle in ankle_candidates(hip, knee):
            if not (-3.15 <= ankle <= 3.15):
                continue
            pose = [0.0] * len(joints)
            for h, k, a in zip(hips, knees, ankles):
                pose[idx[h]], pose[idx[k]], pose[idx[a]] = float(hip), float(knee), float(ankle)
            if any(
                pose[i] < joints[i]["lo"] - 1e-9 or pose[i] > joints[i]["hi"] + 1e-9
                for i in range(len(joints))
            ):
                continue
        tilt, height, ke = simulate(
            m, d, pose, qadr, act, seconds, mujoco, torso_id, robot,
            actuator_model=actuator_model, bam_controller=bam_ctrl, spawn_z=z_est,
        )
        score = tilt + 1000.0 * abs(height / 1000.0 - h_ref) + 10.0 * ke
        results.append(
            {
                "score": score,
                "tilt": tilt,
                "height": height,
                "ke": ke,
                "hip": float(hip),
                "knee": float(knee),
                "ankle": float(ankle),
                "pose": pose,
            }
        )

    print(f"=== {robot_name} ===")
    print(f"搜索 {len(results)} 个候选姿态（{len(hips)} 条腿同步，踝 = -(髋+膝)，静置 {seconds}s）")
    stable = [r for r in results if r["tilt"] < 5.0]
    if not stable:
        print("  ✗ 未找到静置 <5° 倾倒的姿态 → 先跑 check_physics.py 修正质量账/初始高度")
        for r in sorted(results, key=lambda x: x["score"])[:3]:
            print(
                f"    best-effort: hip={r['hip']:+.2f} knee={r['knee']:+.2f} ankle={r['ankle']:+.2f}"
                f" → 倾斜 {r['tilt']:.1f}°，高度 {r['height']:.1f} mm"
            )
        return None

    best = min(stable, key=lambda r: r["score"])
    print(
        f"  ✓ 静态站立姿态：hip={best['hip']:+.2f} knee={best['knee']:+.2f} "
        f"ankle={best['ankle']:+.2f} rad（稳定候选 {len(stable)} 个）"
    )
    print(
        f"    静置后：倾斜 {best['tilt']:.2f}°，躯干高度 {best['height']:.1f} mm"
        f"（h_ref {h_ref * 1000:.0f} mm），动能 {best['ke']:.2e} J"
    )
    pose_yaml = "[" + ", ".join(f"{v:g}" for v in best["pose"]) + "]"
    print("\n  写入 body/robot.yaml：")
    print(f"    stance_pose: {pose_yaml}")
    print("    腿关节 default 字段（决定 RL 动作参考位，零动作应≈站立）：")
    for n in hips + knees + ankles:
        print(f"      {n}: default={best['pose'][idx[n]]:g}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=0.05, help="网格步长（rad）")
    ap.add_argument("--actuator", default="pd", choices=["pd", "bam"],
                    help="用哪种执行器模型求解（应与训练所用一致）")
    args = ap.parse_args()

    cfg = yaml.safe_load(ROBOT_YAML.read_text(encoding="utf-8"))
    names = [args.robot] if args.robot else list(cfg["robots"])
    for n in names:
        solve(n, cfg, args.seconds, args.step, args.actuator)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
