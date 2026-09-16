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

    def __init__(self, robot_name: str = "miniduck-S", spec: Spec | None = None) -> None:
        self.spec = spec or Spec()
        self.model = RobotModel.load(robot_name)
        self.action_scale = self.spec.action_scale()
        self.dt = 1.0 / self.spec.control_hz()
        self.weights = self.spec.reward_terms()
        # 参考量：解析 @robot.* 引用（h_ref 等几何量来自 body/robot.yaml）
        self.refs = self.spec.reward_refs_for(self.model.raw)
        self.segments = self.spec.obs_segments()
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
    env = StandEnv("miniduck-S")
    print(f"机型 {env.model.name}: 腿关节 {env.model.n_leg}, obs {env.obs_dim}, act {env.model.act_dim}")
    print(f"参考量（引用解析后）: h_ref_m={env.refs['h_ref_m']}（来自 body/robot.yaml geometry）")
    obs = env.reset()
    print(f"重置观测: 长度 {len(obs)}, 前 12 项 {[round(x, 3) for x in obs[:12]]}")
    total = 0.0
    for _ in range(200):
        obs, r, done, info = env.step([0.0] * env.model.n_leg)
        total += r
        if done:
            print(f"提前终止于第 {env.t} 步")
            break
    print(f"200 步零动作累计回报: {total:.3f}（每步 {total / max(env.t, 1):.5f}）")
    print("单步奖励分解:", {k: round(v, 4) for k, v in info["terms"].items()})
    print("权重（来自 spec/reward.yac）:", env.weights)
