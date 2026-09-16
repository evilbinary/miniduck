#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""body/tools/check_physics.py — 用 MuJoCo 做物理层面的健全性检查。

verify_gen.py 只验证生成物**结构**自洽（XML 良构、关节/连杆数量、引用有效）。
本脚本验证**物理**可用性，能抓到结构校验抓不到的问题：

    1. 模型能否被 MuJoCo 加载（geom size 分量、default 继承等易错点）
    2. 质量账：连杆质量之和 vs 舵机质量之和 vs 整机质量目标
       —— 舵机质量常被漏算，导致仿真里机器人"过轻"、扭矩显得过强
    3. 站立基座高度自洽：默认姿态下脚底离地多少
       —— 与 robot.yaml 的 geometry.base_height_m 对比，暴露运动学与几何不一致
    4. 1 秒自由落体：是否稳稳落在地面（而不是穿透/弹飞），以及姿态是否垮掉

用法：
    python body/tools/check_physics.py            # 两版都查
    python body/tools/check_physics.py --robot miniduck-S
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOT_YAML = ROOT / "body" / "robot.yaml"

sys.path.insert(0, str(ROOT / "train" / "envs"))          # 复用执行器模型层与几何辅助
from actuators import (  # noqa: E402
    bam_available,
    build_bam_controller,
    hold_pose_steps,
    sync_bam_target_state,
)
from duck_env import (  # noqa: E402
    align_to_ground,
    geom_lowest_point,
    is_foot_geom,
)


def _bam_usable() -> bool:
    return bam_available()


def _is_passive(robot: dict, joint_name: str) -> bool:
    for j in robot["joints"]:
        if j["name"] == joint_name:
            return bool(j.get("passive"))
    return False

OKS: list[str] = []
FAILS: list[str] = []
NOTES: list[str] = []


def ok(m: str) -> None:
    OKS.append(m)


def fail(m: str) -> None:
    FAILS.append(m)


def note(m: str) -> None:
    NOTES.append(m)


def foot_lowest_z(m, d) -> tuple[float, str]:
    """脚底最低点 z 与其所属 body 名（几何分量处理复用 duck_env，避免两处各写一遍）。"""
    import mujoco

    mujoco.mj_forward(m, d)
    lowest, who = float("inf"), ""
    for g in range(m.ngeom):
        if not is_foot_geom(m, mujoco, g):
            continue
        z = geom_lowest_point(m, d, mujoco, g)
        if z < lowest:
            lowest = z
            who = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
    return lowest, who


def set_pose(m, d, robot: dict, mujoco) -> None:
    """把模型设到 stance_pose（训练/真机实际使用的初始姿态）。

    注意：不能用 MuJoCo 默认全零姿态做落地测试——那与训练起始状态不一致，
    会得出误导性结论（例如"出生悬空""1 s 后倾倒"，而 stance_pose 下其实是稳的）。
    """
    # ★ 必须整体重置：否则会带着上一次仿真（例如另一种执行器模型下已经摔倒的
    #   位姿与速度）继续跑，出生姿态就不是 stance_pose 了，接触约束会把人弹飞
    #   （实测动能 4.1 J ≈ 5 m/s 弹射，被误读成"姿态站不住"）。
    mujoco.mj_resetData(m, d)
    pose = robot.get("stance_pose") or [j["default"] for j in robot["joints"]]
    for i, j in enumerate(robot["joints"]):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
        if jid >= 0:
            d.qpos[int(m.jnt_qposadr[jid])] = pose[i]
    d.ctrl[:] = 0.0
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    # 出生高度按姿态算出（脚底贴地）——不要用 robot.yaml 里手写的 base_height_m
    align_to_ground(m, d, mujoco)


def check_robot(name: str, cfg: dict, models: list[str]) -> None:
    try:
        import mujoco
    except ImportError:
        fail("未安装 mujoco（pip install mujoco / python -m pip install mujoco）")
        return

    robot = cfg["robots"][name]
    prof = cfg["servo_profiles"][robot["servo"]]
    path = ROOT / cfg["generate"]["mjcf"]["target"].format(robot=name)

    # 1) 加载
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    except Exception as e:  # noqa: BLE001
        fail(f"[{name}] MuJoCo 加载失败：{e}")
        return
    d = mujoco.MjData(m)
    ok(f"[{name}] MuJoCo 加载成功（nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody}）")

    # 2) 质量账
    sim_mass = mujoco.mj_getTotalmass(m)
    link_mass = sum(l.get("mass_g", 0.0) for l in robot["links"]) / 1000.0
    servo_mass = len(robot["joints"]) * float(prof.get("weight_g", 0.0)) / 1000.0
    ok(f"[{name}] 仿真总质量 {sim_mass * 1000:.0f} g（= robot.yaml 连杆质量之和）")
    if servo_mass > 0.5 * link_mass:
        note(
            f"[{name}] 舵机质量 {servo_mass * 1000:.0f} g 未计入连杆质量："
            f"含舵机后约 {(sim_mass + servo_mass) * 1000:.0f} g，"
            "仿真会偏轻、扭矩显得过强 → robot.yaml 的 mass_g 需回填含舵机的实测值"
        )
    geo_mass = robot.get("geometry", {}).get("mass_g")
    if geo_mass:
        lo, hi = geo_mass
        total = (sim_mass + servo_mass) * 1000.0
        if not (lo * 0.8 <= total <= hi * 1.2):
            note(
                f"[{name}] 含舵机总质量 {total:.0f} g 偏离目标 {lo}–{hi} g"
                "（占位质量待 CAD/实测回填）"
            )

    # 3) 站立高度自洽（在 stance_pose 下评估，与训练起始状态一致）
    set_pose(m, d, robot, mujoco)
    lowest, who = foot_lowest_z(m, d)
    base_h = float(robot.get("geometry", {}).get("base_height_m", 0.0))
    gap = lowest  # 地面在 z=0
    ok(f"[{name}] 默认姿态脚底最低点 z={gap * 1000:.1f} mm（{who}）")
    if gap > 0.002:
        note(
            f"[{name}] 出生时脚底悬空 {gap * 1000:.1f} mm（基座 {base_h * 1000:.0f} mm）："
            "geometry.base_height_m 与腿部运动学不一致，会先自由落体再着地 → 需回填实测值"
        )
    elif gap < -0.005:
        note(f"[{name}] 出生时脚底已陷入地面 {abs(gap) * 1000:.1f} mm，初始姿态需抬高基座")

    # 4) 静置测试：**对每种执行器模型分别验证**
    #    「pd / bam 可切换」是架构特性，必须被测试保证而不是靠假设：
    #      默认模型（robot.yaml actuators.default）站不住 → FAIL（训练会开局就摔）
    #      非默认模型站不住 → NOTE（切过去之前需重跑 solve_stance.py）
    act_cfg = cfg.get("actuators") or {}
    default_model = act_cfg.get("default", "pd")
    solved_with = act_cfg.get("pose_solved_with")
    n_steps = int(1.0 / m.opt.timestep)
    pose = robot.get("stance_pose") or [j["default"] for j in robot["joints"]]
    joint_names = [j["name"] for j in robot["joints"]]

    for model in models:
        if model == "bam" and not _bam_usable():
            note(f"[{name}] 跳过 bam 检查：BAM 不可用（需 Python ≥3.12 + better-actuator-models）")
            continue
        set_pose(m, d, robot, mujoco)
        bam_ctrl = None
        if model == "bam":
            bam_ctrl = build_bam_controller(
                prof, m, d, [n for n in joint_names if not _is_passive(robot, n)]
            )
        if bam_ctrl is not None:
            sync_bam_target_state(bam_ctrl, pose)
        hold_pose_steps(
            m, d, pose, joint_names, prof, mujoco, n_steps,
            actuator_model=model, bam_controller=bam_ctrl,
        )
        quat_w = float(d.qpos[3])
        tilt_deg = float(np.degrees(2 * np.arccos(np.clip(abs(quat_w), -1.0, 1.0))))
        ke = 0.5 * float(np.sum(m.body_mass * np.sum(d.cvel**2, axis=1)))
        msg = (
            f"[{name}/{model}] 静置 1s：高度 {d.qpos[2] * 1000:.1f} mm，倾斜 {tilt_deg:.1f}°，"
            f"触点 {d.ncon}，动能 {ke:.2e} J"
        )
        if not np.isfinite(d.qpos).all():
            fail(f"[{name}/{model}] 仿真数值发散（NaN/Inf）")
            continue
        tag = "（默认模型）" if model == default_model else ""
        if tilt_deg <= 45.0:
            ok(msg + f" → 可作 stand 基线{tag}")
        elif model == default_model:
            fail(
                msg + f"：**默认**执行器模型下 stance_pose 不可站立 → 跑 "
                f"solve_stance.py --actuator {default_model} 重解"
                f"（当前姿态标注为在 {solved_with} 下求解）"
            )
        else:
            note(
                msg + f"：非默认模型（{model}）下站不住；切到 {model} 前需重跑 "
                f"solve_stance.py --actuator {model} 并更新 stance_pose/default"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None, help="只查指定机型")
    ap.add_argument("--actuator", default="all", choices=["all", "pd", "bam"],
                    help="要验证的执行器模型（默认两种都验）")
    args = ap.parse_args()

    cfg = yaml.safe_load(ROBOT_YAML.read_text(encoding="utf-8"))
    names = [args.robot] if args.robot else list(cfg["robots"])
    models = ["pd", "bam"] if args.actuator == "all" else [args.actuator]

    for n in names:
        check_robot(n, cfg, models)

    for m in OKS:
        print(f"PASS  {m}")
    for m in NOTES:
        print(f"NOTE  {m}")
    for m in FAILS:
        print(f"FAIL  {m}")
    print(f"\n{len(OKS)} passed, {len(FAILS)} failed, {len(NOTES)} notes")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
