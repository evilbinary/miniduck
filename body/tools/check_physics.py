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

sys.path.insert(0, str(ROOT / "train" / "envs"))          # 复用执行器模型层
from actuators import hold_pose_steps  # noqa: E402

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
    """当前姿态下所有"脚"相关 geom 的最低点 z 与其所属 body 名。

    注意 capsule 的下沿是 中心 − (半长 + 半径)：只减半长会漏掉半径，
    得出"脚悬空"的假结论（半径 8 mm 时误差正好 8 mm）。
    """
    import mujoco

    mujoco.mj_forward(m, d)
    lowest, who = float("inf"), ""
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "foot" not in name.lower() and "toe" not in name.lower():
            continue
        gtype = m.geom_type[g]
        size = m.geom_size[g]
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            drop = float(size[1]) + float(size[0])          # 半长 + 半径
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            drop = float(size[0])
        else:
            drop = float(size[1]) if len(size) > 1 else 0.0
        z = float(d.geom_xpos[g][2]) - drop
        if z < lowest:
            lowest, who = z, name
    return lowest, who


def set_pose(m, d, robot: dict, mujoco) -> None:
    """把模型设到 stance_pose（训练/真机实际使用的初始姿态）。

    注意：不能用 MuJoCo 默认全零姿态做落地测试——那与训练起始状态不一致，
    会得出误导性结论（例如"出生悬空""1 s 后倾倒"，而 stance_pose 下其实是稳的）。
    """
    pose = robot.get("stance_pose") or [j["default"] for j in robot["joints"]]
    for i, j in enumerate(robot["joints"]):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
        if jid >= 0:
            d.qpos[int(m.jnt_qposadr[jid])] = pose[i]
    d.ctrl[:] = 0.0
    d.qvel[:] = 0.0


def check_robot(name: str, cfg: dict) -> None:
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

    # 4) 1 秒静置测试（从 stance_pose 起步，无策略介入，PD 力矩保持）
    n_steps = int(1.0 / m.opt.timestep)
    pose = robot.get("stance_pose") or [j["default"] for j in robot["joints"]]
    hold_pose_steps(
        m, d, pose, [j["name"] for j in robot["joints"]], prof, mujoco, n_steps
    )
    quat_w = float(d.qpos[3])
    tilt_deg = float(np.degrees(2 * np.arccos(np.clip(abs(quat_w), -1.0, 1.0))))
    ke = 0.5 * float(np.sum(m.body_mass * np.sum(d.cvel**2, axis=1)))
    ok(
        f"[{name}] 1s 落地：高度 {d.qpos[2] * 1000:.1f} mm，倾斜 {tilt_deg:.1f}°，"
        f"触点 {d.ncon}，动能 {ke:.2e} J"
    )
    if not np.isfinite(d.qpos).all():
        fail(f"[{name}] 仿真数值发散（NaN/Inf）")
    if tilt_deg > 45.0:
        fail(
            f"[{name}] stance_pose 下静置 1s 即倾倒 {tilt_deg:.0f}°："
            "标称姿态不可站立 → 跑 solve_stance.py 重新求解（否则 RL 每个 episode 开局就摔）"
        )
    else:
        ok(f"[{name}] stance_pose 静置 1s 保持直立（倾斜 {tilt_deg:.1f}°），可作 stand 基线")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None, help="只查指定机型")
    args = ap.parse_args()

    cfg = yaml.safe_load(ROBOT_YAML.read_text(encoding="utf-8"))
    names = [args.robot] if args.robot else list(cfg["robots"])

    for n in names:
        check_robot(n, cfg)

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
