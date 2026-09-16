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
    ctrl = MujocoController(
        model=model,
        actuator=list(joint_names),
        mujoco_model=mj_model,
        mujoco_data=mj_data,
        vin_drop_resistance=bam_cfg.get("vin_drop_resistance"),
        vin_min=bam_cfg.get("vin_min"),
    )

    # ---- 上游兼容垫片（BAM 1.0.2 的 Feetech 执行器缺初始化）----
    # bam/feetech/actuator.py 的 compute_control 依赖运行期状态 q_target_smooth
    # （限速后的内部目标），但该属性只在 load_log() 里创建：
    #     self.q_target_smooth = np.zeros_like(self.kp)
    # 走内置参数路径（load_model）时不会调用 load_log → AttributeError。
    # 这里显式补上，并在 reset 时同步到初始姿态（见 sync_bam_target_state），
    # 避免开局出现一段"目标从 0 缓慢爬升"的虚假瞬态。
    act = model.actuator
    if not hasattr(act, "q_target_smooth"):
        import numpy as np

        act.q_target_smooth = np.zeros(len(joint_names))
        ctrl._miniduck_shimmed = True

    return ctrl


def sync_bam_target_state(ctrl, pose: list[float]) -> None:
    """把 BAM 的运行期状态对齐到"仿真刚重置"的状态。

    ★ 必须做两件事，否则结果会错：
      1. q_target_smooth ← pose：它是内环的限速平滑目标（上游只在 load_log 里初始化）；
      2. last_ts ← 0：BAM 用 `dt = data.time − last_ts` 计算控制周期，
         而 mj_resetData 会把 data.time 归零。若 last_ts 还停留在上一段仿真
         （例如求解器逐个候选复用时），dt 会变成负数 → 限速项反向 → 控制错乱。
    """
    import numpy as np

    act = getattr(ctrl, "model", None)
    act = getattr(act, "actuator", None)
    if act is not None and hasattr(act, "q_target_smooth"):
        act.q_target_smooth = np.zeros(len(pose)) + np.asarray(pose, dtype=float)
    if hasattr(ctrl, "last_ts"):
        ctrl.last_ts = 0.0


def hold_pose_steps(
    m,
    d,
    pose: list[float],
    joint_names: list[str],
    prof: dict,
    mujoco,
    n_steps: int,
    actuator_model: str = "pd",
    bam_controller=None,
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
    use_bam = actuator_model == "bam" and bam_controller is not None
    if use_bam:
        bam = bam_controller
        # 被动关节（如脚趾）没有执行器，BAM 控制器里没有对应条目 → 跳过
        known = set(getattr(bam, "dof_to_q_target", {}) or {})
        for i, name in enumerate(joint_names):
            if not known or name in known:
                bam.set_q_target(name, pose[i])
    for _ in range(n_steps):
        if use_bam:
            bam.update()                      # BAM 自行写 ctrl
        else:
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
