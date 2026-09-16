# -*- coding: utf-8 -*-
"""train/envs/actuators.py — 执行器模型层。

为什么需要这一层（见 DESIGN 3.4）：
    MuJoCo 内置的 `<position kp kv>` 把舵机当成"瞬时到位的位置源"，
    而 XL330 / STS3032 实际是**有电流限制的位置伺服 + 有刷直流电机 + 减速箱**。
    在快速摆动、大负载、堵转附近，两者差异巨大；用理想位置源训练出来的策略
    上真机会明显不符（力矩过大、响应过快、没有掉压与摩擦造成的死区）。

两种模型：
    PdTorque   力矩级 PD 基线：tau = kp·(q*−q) − kv·q̇，按堵转扭矩裁剪。
               比理想位置源接近真机（有增益上限、饱和、跟随误差），但不含
               摩擦非线性、掉压、齿隙。
    BAM        Better Actuator Models（Rhoban/bam, ICRA 2025）：电压控制 +
               直流电机力矩 + M1–M6 非线性摩擦 + 掉压，由库内 MujocoController
               接管 ctrl。BAM 要求 MJCF 用 `<motor>` 执行器（gen.py 已如此生成）。

注意：BAM 只在 Python ≥3.12 可用（见 README），故本模块延迟导入、按需启用。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PdTorque:
    """力矩级 PD（替代 MuJoCo 内置位置执行器）。"""

    kp: float
    kv: float
    tau_limit: float

    def torque(self, q_target: float, q: float, qd: float) -> float:
        tau = self.kp * (q_target - q) - self.kv * qd
        return max(-self.tau_limit, min(self.tau_limit, tau))

    def torques(self, targets: list[float], qs: list[float], qds: list[float]) -> list[float]:
        return [self.torque(t, q, qd) for t, q, qd in zip(targets, qs, qds)]

    @classmethod
    def from_servo_profile(cls, prof: dict) -> "PdTorque":
        g = prof.get("gains") or {}
        return cls(
            kp=float(g.get("kp", 12.0)),
            kv=float(g.get("kv", 0.4)),
            tau_limit=float(prof.get("stall_torque_nm", 0.5)),
        )


def bam_available() -> bool:
    try:
        import bam.mujoco  # noqa: F401
        import bam.model  # noqa: F401
    except Exception:  # noqa: BLE001  （含 Python<3.12 / 未安装 / mujoco DLL 不可用）
        return False
    return True


def build_bam_controller(
    prof: dict,
    mj_model,
    mj_data,
    joint_names: list[str],
):
    """构造 BAM 的 MujocoController。

    关键点（据 BAM 文档）：
        * MJCF 的执行器必须是 `<motor name="{joint}" .../>`，name 与关节同名；
        * **同一块电池的所有关节必须放进同一个 controller**，否则掉压会被低估
          （每个 controller 独立算电流，电流不会相加）。本项目所有关节共用
          一条总线电源，因此只建一个 controller。
    """
    from bam.model import load_model
    from bam.mujoco import MujocoController

    bam_cfg = prof.get("bam")
    if not bam_cfg:
        raise ValueError("servo_profiles 缺少 bam 配置段")

    model = load_model(motor_name=bam_cfg["motor_name"], model=bam_cfg["model"])
    return MujocoController(
        model=model,
        actuator=list(joint_names),
        mujoco_model=mj_model,
        mujoco_data=mj_data,
        vin_drop_resistance=bam_cfg.get("vin_drop_resistance"),
        vin_min=bam_cfg.get("vin_min"),
    )


def hold_pose_steps(
    m,
    d,
    pose: list[float],
    joint_names: list[str],
    prof: dict,
    mujoco,
    n_steps: int,
) -> None:
    """用 PD 力矩把关节保持在 pose 上并推进 n_steps 个物理步。

    供求解/检查类工具复用（solve_stance.py、check_physics.py）。
    注意：MJCF 的执行器是 `<motor>`（力矩），直接写 ctrl=角度 会变成施加
    一个极小力矩，机器人会瞬间瘫掉——必须按力矩语义驱动。
    """
    pd = PdTorque.from_servo_profile(prof)
    ids = []
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid >= 0 and aid >= 0:
            ids.append((i, int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]), int(aid)))
    for _ in range(n_steps):
        for i, qadr, vadr, aid in ids:
            d.ctrl[aid] = pd.torque(pose[i], float(d.qpos[qadr]), float(d.qvel[vadr]))
        mujoco.mj_step(m, d)


def describe(prof: dict) -> str:
    """人类可读的执行器配置摘要（日志用）。"""
    bam_cfg = prof.get("bam") or {}
    g = prof.get("gains") or {}
    return (
        f"PD(kp={g.get('kp')}, kv={g.get('kv')}, τmax={prof.get('stall_torque_nm')}) | "
        f"BAM({bam_cfg.get('motor_name')}, {bam_cfg.get('model')}, "
        f"vin={bam_cfg.get('vin')}V, Rdrop={bam_cfg.get('vin_drop_resistance')}Ω)"
    )
