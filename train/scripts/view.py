# -*- coding: utf-8 -*-
"""train/scripts/view.py — miniduck 可视化：交互式 viewer 与离屏出图。

两种模式：

    interactive（默认，需要显示器）
        用 MuJoCo 的 passive viewer 实时看机器人动。控制律走环境侧
        （PD 或 BAM），因此看到的就是训练时用的执行器行为。
        --action zero  站立（验证 stance_pose / 执行器能否撑住）
        --action sine  给髋关节加正弦动作（看执行器跟随与摩擦差异）

    render（离屏，无需显示器）
        用 mujoco.Renderer 逐帧渲染成 PNG（可再存 GIF），
        适合在没有屏幕的机器上出图、或做 PR/文档素材。

用法：
    python train/scripts/view.py --robot miniduck-S --actuator bam
    python train/scripts/view.py --mode render --out train/runs/view --seconds 2 --gif
    python train/scripts/view.py --mode render --robot miniduck-L --actuator pd --action sine
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train" / "envs"))

from duck_env import StandEnv  # noqa: E402

# 渲染参数
WIDTH, HEIGHT = 960, 720
RENDER_FPS = 30
SINE_HZ = 0.5            # 正弦动作频率
SINE_AMP = 0.35          # 动作幅度（[-1,1] 归一化）


def scripted_action(env: StandEnv, t: float, kind: str) -> list[float]:
    """生成动作向量（维度 = 策略腿关节数）。"""
    n = env.model.n_leg
    if kind == "zero":
        return [0.0] * n
    # 让左右腿反相摆动，便于肉眼观察髋/膝关节跟随
    names = env.model.leg_joints
    out = [0.0] * n
    for i, name in enumerate(names):
        sign = 1.0 if name.startswith("L_") else -1.0
        if "hip" in name:
            out[i] = sign * SINE_AMP * math.sin(2 * math.pi * SINE_HZ * t)
        elif "knee" in name:
            out[i] = sign * 0.5 * SINE_AMP * math.sin(4 * math.pi * SINE_HZ * t)
    return out


#: 相机预设：从不同方向看才能看清不同问题
#:   side  侧视（看前后重心与脚掌位置）——判断前后失衡
#:   front 前视（看左右对称与髋距）——判断侧向失衡
#:   top   俯视（看支撑多边形与质心投影是否在里面）
CAMERAS = {
    "iso": (135.0, -20.0),
    "side": (90.0, -5.0),
    "front": (0.0, -5.0),
    "top": (90.0, -89.0),
}


def apply_camera(cam, env: StandEnv, kind: str, lookat_z: float) -> None:
    az, el = CAMERAS.get(kind, CAMERAS["iso"])
    cam.azimuth, cam.elevation = az, el
    cam.distance = camera_distance(env)
    cam.lookat[:] = [0.0, 0.0, lookat_z]


def balance_report(m, d, env: StandEnv) -> str:
    """质心投影 vs 支撑多边形：数值版的"为什么站不稳"。

    支撑点取与地面接触的接触点；质心取各 body 质量加权位置（世界系）。
    结论判据：质心投影必须落在支撑多边形内，越靠边越不稳。
    """
    import mujoco

    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pts = [d.contact[c].pos.copy() for c in range(d.ncon)
           if floor in (int(d.contact[c].geom1), int(d.contact[c].geom2))]
    mass = float(np.sum(m.body_mass))
    com = np.sum(m.body_mass[:, None] * d.xipos, axis=0) / mass
    if not pts:
        return f"COM=({com[0] * 1000:+.1f},{com[1] * 1000:+.1f})mm  无地面接触"
    P = np.array(pts)
    mx, Mx = P[:, 0].min(), P[:, 0].max()
    my, My = P[:, 1].min(), P[:, 1].max()
    cx, cy = com[0], com[1]
    inside = (mx - 1e-6 <= cx <= Mx + 1e-6) and (my - 1e-6 <= cy <= My + 1e-6)
    return (
        f"COM=({cx * 1000:+.1f},{cy * 1000:+.1f})mm  "
        f"支撑区 X[{mx * 1000:+.1f},{Mx * 1000:+.1f}] Y[{my * 1000:+.1f},{My * 1000:+.1f}]mm  "
        f"前后余量={min(cx - mx, Mx - cx) * 1000:+.1f}mm 侧向余量={min(cy - my, My - cy) * 1000:+.1f}mm  "
        f"{'在支撑区内 ✓' if inside else '★质心出界→必倒'}"
    )


def camera_distance(env: StandEnv) -> float:
    """按机身体积取景：机器人只有几厘米，固定 1 m 相机距离会小得看不清。"""
    h = float(env.refs["h_ref_m"])
    return max(0.25, 3.2 * (h + 0.05))


def log_setup(env: StandEnv, args) -> None:
    print(f"机型 {env.model.name} | 后端 {env.backend_name} | 执行器 {env.actuator_name} | "
          f"obs {env.obs_dim} / act {env.model.act_dim} | 动作 {args.action}")
    if args.action == "zero":
        print(f"stance_pose = {env.model.stance_pose}")
        print(f"h_ref = {env.refs['h_ref_m'] * 1000:.0f} mm")


# --------------------------------------------------------------------------
# 交互式
# --------------------------------------------------------------------------
def run_interactive(env: StandEnv, args) -> int:
    import mujoco
    import mujoco.viewer

    m, d = env.physics.m, env.physics.d
    env.reset()

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    apply_camera(cam, env, args.camera, env.refs["h_ref_m"])

    dt = env.dt
    print("窗口已打开：拖动鼠标旋转、滚轮缩放、右键平移，关闭窗口结束。")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        last_report = t0
        while viewer.is_running():
            t = time.time() - t0
            action = scripted_action(env, t, args.action)
            _, reward, done, info = env.step(action)
            viewer.sync()
            if time.time() - last_report > 1.0:
                s = info["state"]
                print(
                    f"t={t:5.2f}s 高度={s.base_height * 1000:6.1f}mm "
                    f"pitch={s.imu_euler_deg[1]:6.1f}° roll={s.imu_euler_deg[0]:6.1f}° "
                    f"reward={reward:5.2f}{'  [terminated]' if done else ''}"
                )
                last_report = time.time()
                if done:
                    env.reset()
            # 按控制周期节流（仿真步进在 env.step 内完成）
            time.sleep(max(0.0, dt - (time.time() - t)))
    return 0


# --------------------------------------------------------------------------
# 离屏渲染
# --------------------------------------------------------------------------
def run_render(env: StandEnv, args) -> int:
    import mujoco
    from PIL import Image

    m, d = env.physics.m, env.physics.d
    env.reset()

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    apply_camera(cam, env, args.camera, float(d.qpos[2]))

    # 离屏缓冲默认 640×480；MJCF 里已声明 offwidth/offheight=1280×960。
    # 若模型没带该声明（旧生成物），自动退到 640×480 并提示。
    try:
        renderer = mujoco.Renderer(m, HEIGHT, WIDTH)
        size = (WIDTH, HEIGHT)
    except ValueError as e:
        print(f"[warn] 离屏缓冲不足（{e}），退到 640×480；"
              "重新运行 body/tools/gen.py 可获得 1280×960 缓冲")
        renderer = mujoco.Renderer(m, 480, 640)
        size = (640, 480)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = int(args.seconds * RENDER_FPS)
    substeps = max(1, int(round((1.0 / RENDER_FPS) / env.dt)))   # 每帧推进几个控制周期
    frames: list[Image.Image] = []
    heights: list[float] = []
    t0 = time.time()

    for k in range(n_frames):
        t = k / RENDER_FPS
        for _ in range(substeps):
            action = scripted_action(env, t, args.action)
            _, _, done, info = env.step(action)
            if done:
                env.reset()
        s = info["state"]
        heights.append(s.base_height)
        apply_camera(cam, env, args.camera, s.base_height)
        renderer.update_scene(d, camera=cam)
        img = Image.fromarray(renderer.render())
        frames.append(img)
        img.save(out_dir / f"frame_{k:04d}.png")

    renderer.close()

    heights_mm = [h * 1000 for h in heights]
    print(f"渲染 {len(frames)} 帧 → {out_dir}")
    print(f"  尺寸 {size[0]}×{size[1]} @ {RENDER_FPS} fps，时长 {args.seconds}s，"
          f"耗时 {time.time() - t0:.1f}s")
    print(f"  躯干高度 {min(heights_mm):.1f}–{max(heights_mm):.1f} mm（h_ref "
          f"{env.refs['h_ref_m'] * 1000:.0f} mm）")
    if args.diag:
        print("  平衡诊断（末帧）:", balance_report(m, d, env))

    if args.gif:
        gif = out_dir / f"{env.model.name}_{env.actuator_name}_{args.action}.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 / RENDER_FPS), loop=0)
        print(f"  GIF → {gif}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="miniduck 可视化")
    ap.add_argument("--mode", default="interactive", choices=["interactive", "render"])
    ap.add_argument("--robot", default="miniduck-S", choices=["miniduck-L", "miniduck-S"])
    ap.add_argument("--actuator", default="auto", choices=["auto", "pd", "bam"])
    ap.add_argument("--action", default="zero", choices=["zero", "sine"],
                    help="zero=站立（验证姿态能否撑住）；sine=髋关节正弦摆动")
    ap.add_argument("--seconds", type=float, default=3.0, help="render 模式时长")
    ap.add_argument("--out", default="train/runs/view", help="render 模式输出目录")
    ap.add_argument("--gif", action="store_true", help="render 模式额外存 GIF")
    ap.add_argument("--camera", default="iso", choices=list(CAMERAS),
                    help="iso 总览 / side 侧视 / front 前视 / top 俯视")
    ap.add_argument("--diag", action="store_true",
                    help="打印质心投影与支撑多边形（数值版平衡诊断）")
    args = ap.parse_args()

    env = StandEnv(args.robot, backend="mujoco", actuator_model=args.actuator)
    log_setup(env, args)
    if args.mode == "interactive":
        return run_interactive(env, args)
    return run_render(env, args)


if __name__ == "__main__":
    sys.exit(main())
