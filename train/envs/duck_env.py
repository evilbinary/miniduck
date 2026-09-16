# -*- coding: utf-8 -*-
"""train/envs/duck_env.py — miniduck 训练环境骨架（stage_1_stand）。

分层：
    Spec          观测/奖励/动作/随机化的唯一来源（mind/spec/*.yac）
    RobotModel    关节表与限位（body/robot.yaml）
    DuckState     一帧状态（与真机 build_obs 的入参一一对应）
    NullPhysics   占位物理后端：不做动力学，只把关节推到目标位
                  → 让 obs / reward / 映射逻辑可在无 MuJoCo 环境下测试
    StandEnv      stage_1_stand：站立不动的收敛验证

MuJoCo/MJX 后端将在 M2 后续接入，接口保持一致（reset / step）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from spec_loader import Spec, SpecError  # type: ignore  (同目录直接运行)

ROOT = Path(__file__).resolve().parents[2]
ROBOT_YAML = ROOT / "body" / "robot.yaml"


# --------------------------------------------------------------------------
# 机器人模型
# --------------------------------------------------------------------------
class RobotModel:
    def __init__(self, cfg: dict, robot_name: str) -> None:
        self.name = robot_name
        r = cfg["robots"][robot_name]
        self.raw = r
        self.servo = cfg["servo_profiles"][r["servo"]]
        self.joints: list[dict] = r["joints"]
        self.by_name = {j["name"]: j for j in self.joints}
        self.leg_joints: list[str] = r["policy"]["leg_joints"]
        self.n_leg = len(self.leg_joints)
        self.obs_dim: int = r["policy"]["obs_dim"]
        self.act_dim: int = r["policy"]["act_dim"]
        self.default_pos = [j["default"] for j in self.joints]
        self.safe_pose = list(r["safe_pose"])
        # 静态站立姿态：仿真初始状态用；缺省回退到 safe_pose
        self.stance_pose = list(r.get("stance_pose") or r["safe_pose"])
        # 执行器模型默认值来自 robot.yaml 的 actuators 段（配置层单一来源）
        actuators_cfg = cfg.get("actuators") or {}
        self.actuator_default = str(actuators_cfg.get("default", "pd"))
        self.actuator_fallback = bool(actuators_cfg.get("fallback_to_pd_on_error", True))
        self.pose_solved_with = actuators_cfg.get("pose_solved_with")
        leg = [self.by_name[n] for n in self.leg_joints]
        self.leg_lo = min(j["lo"] for j in leg)
        self.leg_hi = max(j["hi"] for j in leg)
        # 动作向量 → 全关节位置向量 的下标映射
        self.leg_index = [self.joints.index(self.by_name[n]) for n in self.leg_joints]

    @classmethod
    def load(cls, robot_name: str = "miniduck-S", path: Path = ROBOT_YAML) -> "RobotModel":
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")), robot_name)

    def action_to_target(self, action: list[float], action_scale: float) -> list[float]:
        """DESIGN 4.2：target = default_pos + scale * action，并做限位截断。"""
        if len(action) != self.n_leg:
            raise ValueError(f"动作维度 {len(action)} != {self.n_leg}")
        out = list(self.default_pos)
        for i, a in enumerate(action):
            idx = self.leg_index[i]
            t = self.default_pos[idx] + action_scale * a
            out[idx] = min(max(t, self.joints[idx]["lo"]), self.joints[idx]["hi"])
        return out


# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------
@dataclass
class DuckState:
    """一帧状态。字段与 DESIGN 6.1 的 build_obs 入参一一对应。"""

    joint_pos: list[float]
    joint_vel: list[float]
    base_ang_vel: list[float]          # 3, rad/s
    projected_gravity: list[float]     # 3
    base_lin_vel: list[float]          # 2, m/s（不可靠时置 0）
    base_height: float                 # m
    imu_euler_deg: list[float]         # 3, 度（与 safety.fall_deg 同单位）
    command: list[float]               # 3, vx, vy, yaw_rate
    contact: list[float]               # 2
    phase: list[float]                 # 2, cos, sin
    quat_xy: list[float] = field(default_factory=lambda: [0.0, 0.0])  # 姿态误差分量
    joint_torque: list[float] | None = None
    self_collision: bool = False


# --------------------------------------------------------------------------
# 物理后端：MuJoCo
# --------------------------------------------------------------------------
class MujocoPhysics:
    """真实物理后端：加载 body/mjcf/{robot}.xml 并逐步仿真。

    与真机的对应关系（这是 sim2real 的关键部分）：
        base_ang_vel        ← 机体坐标系角速度（陀螺）
        projected_gravity   ← 重力在机体系下的投影
        base_lin_vel        ← 机体系线速度（真机上不可靠 → 训练侧按 spec 置 0）
        base_height         ← torso 离地高度
        imu_euler_deg       ← roll/pitch/yaw（度，与 safety.fall_deg 同单位）
        contact             ← 脚/脚趾 geom 与地面的接触
        joint_torque        ← 执行器输出力（对应舵机负载）
    """

    def __init__(
        self,
        model: RobotModel,
        control_dt: float,
        gait_period_s: float,
        actuator_model: str = "pd",
    ) -> None:
        import mujoco  # 延迟导入：无 MuJoCo 时仍可用 NullPhysics 跑管线

        from actuators import PdTorque, build_bam_controller  # 同目录

        self.mj = mujoco
        self.rm = model
        xml = ROOT / "body" / "mjcf" / f"{model.name}.xml"
        if not xml.exists():
            raise FileNotFoundError(f"MJCF 不存在：{xml}（先跑 python body/tools/gen.py）")
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.dt = control_dt
        self.gait_period_s = gait_period_s
        self.substeps = max(1, int(round(control_dt / self.m.opt.timestep)))
        self.t = 0

        # 执行器模型：pd（力矩级 PD 基线）| bam（电压控制 + 直流电机 + M1-M6 摩擦 + 掉压）
        self.actuator_model = actuator_model
        self.pd = PdTorque.from_servo_profile(model.servo)
        self.bam = None
        if actuator_model == "bam":
            joint_names = [j["name"] for j in model.joints if not j.get("passive")]
            self.bam = build_bam_controller(model.servo, self.m, self.d, joint_names)
            self.bam_joints = joint_names

        # 关节名 → qpos / qvel / actuator 下标
        self.qpos_of: dict[str, int] = {}
        self.qvel_of: dict[str, int] = {}
        for j in model.joints:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
            if jid >= 0:
                self.qpos_of[j["name"]] = int(self.m.jnt_qposadr[jid])
                self.qvel_of[j["name"]] = int(self.m.jnt_dofadr[jid])
        # 执行器名 = 关节名（BAM 要求；gen.py 生成 <motor name="{joint}">）
        self.act_of: dict[str, int] = {}
        for j in model.joints:
            aid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, j["name"])
            if aid >= 0:
                self.act_of[j["name"]] = int(aid)
        self.torso_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.floor_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.idx_of = {j["name"]: i for i, j in enumerate(model.joints)}

    # ---------------- 生命周期 ----------------
    def reset(self) -> None:
        """初始状态用 stance_pose（静态站立姿态），不用 default_pos。

        default_pos 是 RL 的动作参考位，实测不是静态平衡姿态（膝弯曲使质心前移，
        0.5 s 内前倾倒地）——从它起步会让 episode 一开场就走向摔倒，
        掩盖策略本身的表现。
        """
        mj, d = self.mj, self.d
        mj.mj_resetData(self.m, d)
        stance = self.rm.stance_pose
        for i, j in enumerate(self.rm.joints):
            name = j["name"]
            if name in self.qpos_of:
                d.qpos[self.qpos_of[name]] = stance[i]
        d.ctrl[:] = 0.0            # 力矩执行器：初始力矩置 0
        d.qvel[:] = 0.0
        mj.mj_forward(self.m, d)
        if self.bam is not None:
            from actuators import sync_bam_target_state

            sync_bam_target_state(self.bam, stance)   # 限速平滑目标对齐初始姿态
        self.t = 0

    def step(self, target_pos: list[float], dt: float | None = None) -> None:
        """target_pos 为**全关节**目标角（未参与策略的关节保持默认位）。

        dt 参数为接口统一而保留（MuJoCo 内部用 substeps × m.opt.timestep 推进）。

        执行器模型：
            pd  → 逐子步计算力矩级 PD（有增益上限/饱和/跟随误差）
            bam → 交给 BAM 的 MujocoController（电压控制 + 直流电机 + 摩擦 + 掉压），
                  它自行写 d.ctrl，并按上一子步负载更新 dof_frictionloss/damping
        """
        d = self.d
        if self.bam is not None:
            for name in self.bam_joints:
                self.bam.set_q_target(name, target_pos[self.idx_of[name]])
            for _ in range(self.substeps):
                self.bam.update()
                self.mj.mj_step(self.m, d)
        else:
            for _ in range(self.substeps):
                for i, j in enumerate(self.rm.joints):
                    name = j["name"]
                    if name in self.act_of:
                        d.ctrl[self.act_of[name]] = self.pd.torque(
                            target_pos[i],
                            float(d.qpos[self.qpos_of[name]]),
                            float(d.qvel[self.qvel_of[name]]),
                        )
                self.mj.mj_step(self.m, d)
        self.t += 1

    # ---------------- 状态提取 ----------------
    def _contacts(self) -> list[float]:
        """左右脚触地：脚/脚趾 geom 与地面有接触即 1.0。"""
        mj, d = self.mj, self.d
        out = [0.0, 0.0]
        for ci in range(d.ncon):
            c = d.contact[ci]
            pair = (int(c.geom1), int(c.geom2))
            if self.floor_gid not in pair:
                continue
            other = pair[0] if pair[1] == self.floor_gid else pair[1]
            bname = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, int(self.m.geom_bodyid[other])) or ""
            low = bname.lower()
            if "foot" in low or "toe" in low or "shank" in low:
                if low.startswith("l_"):
                    out[0] = 1.0
                elif low.startswith("r_"):
                    out[1] = 1.0
                else:
                    out[0] = out[1] = 1.0
        return out

    def state(self) -> DuckState:
        mj, d = self.mj, self.d
        n = len(self.rm.joints)
        jpos = [
            float(d.qpos[self.qpos_of[j["name"]]]) if j["name"] in self.qpos_of else 0.0
            for j in self.rm.joints
        ]
        jvel = [
            float(d.qvel[self.qvel_of[j["name"]]]) if j["name"] in self.qvel_of else 0.0
            for j in self.rm.joints
        ]
        tau = [0.0] * n
        for j in self.rm.joints:
            if j["name"] in self.act_of:
                tau[self.idx_of[j["name"]]] = float(d.actuator_force[self.act_of[j["name"]]])

        # 机体系量：四元数→旋转矩阵，再取机体系分量
        quat = [float(x) for x in d.qpos[3:7]]
        mat = np.zeros(9)
        mj.mju_quat2Mat(mat, quat)
        R = mat.reshape(3, 3)                 # 机体系 → 世界系
        Rt = R.T
        w_world = np.array(d.qvel[3:6], dtype=float)
        v_world = np.array(d.qvel[0:3], dtype=float)
        ang_body = Rt @ w_world
        vel_body = Rt @ v_world
        grav_body = Rt @ np.array([0.0, 0.0, -1.0])

        # 欧拉角（度）
        qw, qx, qy, qz = quat
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

        # 步态相位（与 spec 的 gait_period_s 同源）
        ph = 2.0 * math.pi * (self.t * self.dt) / self.gait_period_s
        height = float(d.xpos[self.torso_bid][2]) if self.torso_bid >= 0 else float(d.qpos[2])

        return DuckState(
            joint_pos=jpos,
            joint_vel=jvel,
            base_ang_vel=[float(x) for x in ang_body],
            projected_gravity=[float(x) for x in grav_body],
            base_lin_vel=[float(vel_body[0]), float(vel_body[1])],
            base_height=height,
            imu_euler_deg=[math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
            command=[0.0, 0.0, 0.0],           # stand 任务：零命令
            contact=self._contacts(),
            phase=[math.cos(ph), math.sin(ph)],
            quat_xy=[float(qx), float(qy)],
            joint_torque=tau,
            self_collision=self._self_collision(),
        )

    def _self_collision(self) -> bool:
        """左右腿互相接触（自撞惩罚用）。"""
        mj, d = self.mj, self.d
        for ci in range(d.ncon):
            c = d.contact[ci]
            names = []
            for g in (int(c.geom1), int(c.geom2)):
                names.append(
                    (mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, int(self.m.geom_bodyid[g])) or "").lower()
                )
            if len(names) == 2:
                l = any(n.startswith("l_") for n in names)
                r = any(n.startswith("r_") for n in names)
                if l and r:
                    return True
        return False


# --------------------------------------------------------------------------
# 物理后端（占位）
# --------------------------------------------------------------------------
class NullPhysics:
    """占位后端：关节瞬间到位、无动力学。用于验证 obs/reward/映射的管线。

    真机一致性来自 obs 构建与 reward 定义（都源自 spec），因此用占位后端
    就能先把「管线」跑通；接 MuJoCo 时只替换本类。
    """

    def __init__(self, model: RobotModel) -> None:
        self.m = model
        self.target = list(model.default_pos)
        # 占位后端把机器人固定在站立目标高度（来自 robot.yaml 的几何）
        self.stand_height = float(model.raw.get("geometry", {}).get("base_height_m", 0.12))

    def reset(self) -> None:
        self.target = list(self.m.default_pos)

    def step(self, target_pos: list[float], dt: float) -> None:
        self.target = list(target_pos)

    def state(self, committable: bool = True) -> DuckState:
        return DuckState(
            joint_pos=list(self.target),
            joint_vel=[0.0] * len(self.target),
            base_ang_vel=[0.0, 0.0, 0.0],
            projected_gravity=[0.0, 0.0, -1.0],
            base_lin_vel=[0.0, 0.0],
            base_height=self.stand_height,
            imu_euler_deg=[0.0, 0.0, 0.0],
            command=[0.0, 0.0, 0.0],
            contact=[1.0, 1.0],
            phase=[1.0, 0.0],
            quat_xy=[0.0, 0.0],
            joint_torque=[0.0] * len(self.target),
            self_collision=False,
        )


# --------------------------------------------------------------------------
# 环境
# --------------------------------------------------------------------------
class StandEnv:
    """stage_1_stand：命令恒为零，观测/奖励全部按 spec 实现。"""

    def __init__(
        self,
        robot_name: str = "miniduck-S",
        spec: Spec | None = None,
        backend: str = "mujoco",
        actuator_model: str = "pd",
    ) -> None:
        self.spec = spec or Spec()
        self.model = RobotModel.load(robot_name)
        self.action_scale = self.spec.action_scale()
        self.dt = 1.0 / self.spec.control_hz()
        self.gait_period_s = self.spec.gait_period_s()
        self.weights = self.spec.reward_terms()
        # 参考量：解析 @robot.* 引用（h_ref 等几何量来自 body/robot.yaml）
        self.refs = self.spec.reward_refs_for(self.model.raw)
        self.segments = self.spec.obs_segments()
        self.backend_name = backend
        # 执行器模型选择（配置层单一来源：robot.yaml 的 actuators.default）
        want = actuator_model
        if want == "auto":
            want = self.model.actuator_default
        if want == "bam":
            from actuators import bam_available

            if not bam_available():
                msg = (
                    "BAM 不可用（需 Python ≥3.12 且安装 better-actuator-models[mujoco]；"
                    "本机 python 3.11 不行）"
                )
                if self.model.actuator_fallback:
                    print(f"[warn] {msg}，按 actuators.fallback_to_pd_on_error 回退 pd")
                    want = "pd"
                else:
                    raise RuntimeError(msg)
        self.actuator_name = want
        if backend == "mujoco":
            try:
                self.physics = MujocoPhysics(
                    self.model, self.dt, self.gait_period_s, actuator_model=want
                )
            except Exception as e:  # noqa: BLE001
                print(f"[warn] MuJoCo 后端不可用（{e}），回退 NullPhysics")
                self.backend_name = "null"
                self.actuator_name = "n/a"
                self.physics = NullPhysics(self.model)
        else:
            self.actuator_name = "n/a"
            self.physics = NullPhysics(self.model)
        self.last_action = [0.0] * self.model.n_leg
        self.t = 0

    # ---------------- 维度 ----------------
    def _seg_dim(self, dim: int | str) -> int:
        return self.model.n_leg if dim == "N" else int(dim)

    @property
    def obs_dim(self) -> int:
        return sum(self._seg_dim(d) for _, d in self.segments)

    def check_dims(self) -> None:
        """spec 段表推出的维度必须与 robot.yaml 一致（跨单一来源一致性检查）。"""
        calc = sum(self._seg_dim(d) for _, d in self.segments)
        if calc != self.model.obs_dim:
            raise SpecError(
                f"obs 维度不一致：spec 段表推出 {calc}，robot.yaml 声明 {self.model.obs_dim}"
            )
        if self.model.act_dim != self.model.n_leg:
            raise SpecError(f"act_dim={self.model.act_dim} 与 leg_joints={self.model.n_leg} 不一致")

    def check_refs(self) -> None:
        """跨文件引用必须全部可解析，且数值要落在物理合理区间。"""
        refs = self.refs
        h = refs.get("h_ref_m")
        if not isinstance(h, (int, float)):
            raise SpecError(f"h_ref_m 未解析为数值：{h!r}")
        if not (0.0 < h < 0.5):
            raise SpecError(f"h_ref_m={h} 超出合理范围 (0, 0.5) m")
        hc = self.model.raw.get("geometry", {}).get("height_cm")
        if hc and h > min(hc) / 100.0:
            raise SpecError(
                f"h_ref_m={h} 大于整机身高下限 {min(hc)} cm，几何不自洽"
                "（检查 body/robot.yaml 的 geometry）"
            )

    # ---------------- 观测 ----------------
    def build_obs(self, s: DuckState, last_action: list[float]) -> list[float]:
        """严格按 spec.obs_segments 顺序拼接（真机 build_obs 同构）。"""
        src = {
            "base_ang_vel": s.base_ang_vel,
            "projected_gravity": s.projected_gravity,
            "command": s.command,
            "joint_pos": [s.joint_pos[i] for i in self.model.leg_index],
            "joint_vel": [s.joint_vel[i] for i in self.model.leg_index],
            "last_action": list(last_action),
            "contact": s.contact,
            "phase": s.phase,
            "base_height": [s.base_height],
            "imu_euler": s.imu_euler_deg,
            "base_lin_vel": s.base_lin_vel,
        }
        obs: list[float] = []
        for name, dim in self.segments:
            vals = src[name]
            want = self._seg_dim(dim)
            if len(vals) != want:
                raise SpecError(f"段 {name}: 实际 {len(vals)} 维 != 期望 {want} 维")
            obs.extend(float(v) for v in vals)
        return obs

    # ---------------- 奖励 ----------------
    @staticmethod
    def _exp_cost(err: float, sigma: float) -> float:
        return math.exp(-(err * err) / sigma)

    def reward(self, s: DuckState, action: list[float], prev_action: list[float]) -> dict[str, float]:
        refs = self.refs
        cmd = s.command
        terms: dict[str, float] = {}

        dvx = s.base_lin_vel[0] - cmd[0]
        dvy = s.base_lin_vel[1] - cmd[1]
        terms["track_lin_vel_xy"] = self._exp_cost(math.hypot(dvx, dvy), refs["sigma_track"])
        terms["track_ang_vel_z"] = self._exp_cost(
            s.base_ang_vel[2] - cmd[2], refs["sigma_track"]
        )
        terms["base_height"] = self._exp_cost(
            s.base_height - refs["h_ref_m"], refs["sigma_height"]
        )
        terms["orientation"] = self._exp_cost(
            math.hypot(s.quat_xy[0], s.quat_xy[1]), refs["sigma_orientation"]
        )
        terms["action_smoothness"] = -sum((a - p) ** 2 for a, p in zip(action, prev_action))
        tau = s.joint_torque or [0.0] * len(s.joint_pos)
        terms["joint_torque"] = -sum(t * t for t in tau)
        terms["alive"] = 1.0
        terms["self_collision"] = -1.0 if s.self_collision else 0.0
        if "lateral_stability" in self.weights:                 # S 版专用
            terms["lateral_stability"] = -(s.base_lin_vel[1] ** 2)
        return terms

    def reward_total(self, terms: dict[str, float]) -> float:
        return sum(self.weights.get(k, 0.0) * v for k, v in terms.items())

    def terminated(self, s: DuckState) -> bool:
        rules = {row[0]: row[1] for row in self.spec.get("termination")}
        if s.base_height < rules["base_height_min_m"]:
            return True
        if abs(s.imu_euler_deg[1]) > rules["pitch_max_deg"]:
            return True
        if abs(s.imu_euler_deg[0]) > rules["roll_max_deg"]:
            return True
        return bool(s.self_collision)

    # ---------------- 标准接口 ----------------
    def reset(self) -> list[float]:
        self.check_dims()
        self.check_refs()
        self.physics.reset()
        self.last_action = [0.0] * self.model.n_leg
        self.t = 0
        return self.build_obs(self.physics.state(), self.last_action)

    def step(self, action: list[float]):
        target = self.model.action_to_target(action, self.action_scale)
        self.physics.step(target, self.dt)
        s = self.physics.state()
        terms = self.reward(s, action, self.last_action)
        total = self.reward_total(terms)
        obs = self.build_obs(s, action)
        self.last_action = list(action)
        self.t += 1
        return obs, total, self.terminated(s), {"terms": terms, "state": s}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="miniduck stand 环境自检")
    ap.add_argument("--robot", default="miniduck-S", choices=["miniduck-L", "miniduck-S"])
    ap.add_argument("--backend", default="mujoco", choices=["mujoco", "null"])
    ap.add_argument("--actuator", default="auto", choices=["auto", "pd", "bam"],
                    help="执行器模型：auto=按 robot.yaml 的 actuators.default；"
                         "pd=力矩级PD基线；bam=电压+直流电机+M1-M6摩擦+掉压")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    env = StandEnv(args.robot, backend=args.backend, actuator_model=args.actuator)
    print(f"机型 {env.model.name}: 腿关节 {env.model.n_leg}, obs {env.obs_dim}, "
          f"act {env.model.act_dim}, 后端 {env.backend_name}, 执行器 {env.actuator_name}")
    if env.model.pose_solved_with and env.model.pose_solved_with != env.actuator_name:
        print(f"[warn] stance_pose 是在 {env.model.pose_solved_with} 下求解的，"
              f"当前执行器为 {env.actuator_name} —— 可能不是静态平衡姿态，"
              "建议重跑 solve_stance.py")
    from actuators import describe

    print(f"执行器配置: {describe(env.model.servo)}")
    print(f"参考量（引用解析后）: h_ref_m={env.refs['h_ref_m']}（来自 body/robot.yaml geometry）")

    obs = env.reset()
    print(f"重置观测: 长度 {len(obs)}, 前 12 项 {[round(x, 3) for x in obs[:12]]}")
    total, heights, contacts, done = 0.0, [], None, False
    for _ in range(args.steps):
        obs, r, done, info = env.step([0.0] * env.model.n_leg)
        s = info["state"]
        total += r
        heights.append(s.base_height)
        contacts = s.contact
        if done:
            break
    print(f"{env.t} 步零动作累计回报 {total:.3f}（每步 {total / max(env.t, 1):.5f}）"
          + ("，提前终止" if done else ""))
    print(f"躯干高度: 起 {heights[0] * 1000:.1f} mm → 终 {heights[-1] * 1000:.1f} mm "
          f"（区间 {min(heights) * 1000:.1f}–{max(heights) * 1000:.1f} mm，h_ref={env.refs['h_ref_m'] * 1000:.0f} mm）")
    print(f"触地: {contacts}  IMU 姿态: pitch={info['state'].imu_euler_deg[1]:.1f}°, "
          f"roll={info['state'].imu_euler_deg[0]:.1f}°")
    print("单步奖励分解:", {k: round(v, 4) for k, v in info["terms"].items()})
    print("权重（来自 spec/reward.yac）:", env.weights)
