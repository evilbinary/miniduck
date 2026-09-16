# -*- coding: utf-8 -*-
"""train/envs/tests/obs_parity.py — Python 与 yac 两侧 build_obs 的逐位对拍。

这是 M3.5 门禁的提前版：在训练开始前就验证「观测构建」在两端完全一致。

做法：
    1. 用一组**顺序递增**的夹具值填充各观测段（1,2,3,...）；
       若两侧段顺序或维度有任何偏差，比对必然失败（顺序敏感）。
    2. Python 侧用 StandEnv.build_obs 算出观测；
    3. 动态生成一个 yac 程序：同样的夹具 + DESIGN 6.1 的 build_obs，
       逐项打印观测分量；
    4. 用真实 yac 解释器执行，解析输出，与 Python 结果逐个比对。

用法：
    python train/envs/tests/obs_parity.py            # L 与 S 两版都跑
    YAC_BIN=/path/to/yac python train/envs/tests/obs_parity.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "envs"))

from duck_env import DuckState, StandEnv  # noqa: E402

YAC_BIN = os.environ.get("YAC_BIN") or str(ROOT.parent / "yac" / "yac.exe")
OUT_DIR = ROOT / "train" / "runs" / "parity"          # gitignored

YAC_BUILD_OBS = """
let clamp(x, lo, hi) = if x < lo then lo else if x > hi then hi else x in
let map2(f, xs, ys) =
  if len(xs) == 0 then []
  else cons(f(nth(xs, 0), nth(ys, 0)), map2(f, tail(xs), tail(ys))) in
let map3(f, xs, ys, zs) =
  if len(xs) == 0 then []
  else cons(f(nth(xs, 0), nth(ys, 0), nth(zs, 0)),
            map3(f, tail(xs), tail(ys), tail(zs))) in
let build_obs(ang_vel, grav, cmd, jpos, jvel, last_action,
              contact, phase, base_h, euler, lin_vel) =
  append(append(append(append(ang_vel, grav), cmd),
                append(append(append(jpos, jvel), last_action), contact)),
         append(append(append(phase, base_h), euler), lin_vel)) in
"""


def fmt(xs) -> str:
    return "[" + ", ".join(f"{float(x):g}" for x in xs) + "]"


def fixture(env: StandEnv):
    """顺序递增夹具：第 k 个观测分量期望恰为 k+1。

    注意：Python 与 yac 用的是同一组夹具值，且都按 spec 段顺序拼接，
    因此任何一侧的顺序/维度错误都会导致比对失败。
    """
    n = env.model.n_leg
    cur = 1.0

    def take(k: int) -> list[float]:
        nonlocal cur
        out = [cur + i for i in range(k)]
        cur += k
        return out

    ang = take(3)
    grav = take(3)
    cmd = take(3)
    jpos_leg = take(n)
    jvel_leg = take(n)
    last_action = take(n)
    contact = take(2)
    phase = take(2)
    base_h = take(1)
    euler = take(3)
    lin_vel = take(2)

    joint_pos = [0.0] * len(env.model.joints)
    joint_vel = [0.0] * len(env.model.joints)
    for k, idx in enumerate(env.model.leg_index):
        joint_pos[idx] = jpos_leg[k]
        joint_vel[idx] = jvel_leg[k]

    state = DuckState(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_ang_vel=ang,
        projected_gravity=grav,
        base_lin_vel=lin_vel,
        base_height=base_h[0],
        imu_euler_deg=euler,
        command=cmd,
        contact=contact,
        phase=phase,
    )
    yac_args = dict(
        ang_vel=ang,
        grav=grav,
        cmd=cmd,
        jpos=jpos_leg,
        jvel=jvel_leg,
        last_action=last_action,
        contact=contact,
        phase=phase,
        base_h=base_h,
        euler=euler,
        lin_vel=lin_vel,
    )
    return state, yac_args


def gen_yac_program(args: dict, dim: int, path: Path) -> None:
    call = (
        "build_obs("
        f"{fmt(args['ang_vel'])}, {fmt(args['grav'])}, {fmt(args['cmd'])}, "
        f"{fmt(args['jpos'])}, {fmt(args['jvel'])}, {fmt(args['last_action'])}, "
        f"{fmt(args['contact'])}, {fmt(args['phase'])}, {fmt(args['base_h'])}, "
        f"{fmt(args['euler'])}, {fmt(args['lin_vel'])})"
    )
    lines = [
        "-- 自动生成（train/envs/tests/obs_parity.py），勿手改",
        YAC_BUILD_OBS.strip(),
        f"let obs = {call} in",
    ]
    for k in range(dim):
        lines.append(f"let _ = print(nth(obs, {k})) in")
    lines.append("()")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(robot: str) -> bool:
    env = StandEnv(robot)
    dim = env.obs_dim
    state, args = fixture(env)
    py_obs = env.build_obs(state, args["last_action"])
    expected = [float(k + 1) for k in range(dim)]

    prog = OUT_DIR / f"obs_parity_{robot}.yac"
    gen_yac_program(args, dim, prog)

    if not Path(YAC_BIN).exists():
        print(f"FAIL  [{robot}] 找不到 yac 解释器：{YAC_BIN}")
        return False
    r = subprocess.run([YAC_BIN, str(prog)], capture_output=True, text=True)
    nums: list[float] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line == "()":
            continue
        try:
            nums.append(float(line))
        except ValueError:
            pass

    ok = True
    if r.returncode != 0:
        print(f"FAIL  [{robot}] yac 执行失败 rc={r.returncode}: {r.stderr.strip()[:200]}")
        ok = False
    if len(nums) != dim:
        print(f"FAIL  [{robot}] yac 输出 {len(nums)} 个数 != obs 维度 {dim}")
        ok = False
        return ok
    bad = [
        (k, nums[k], py_obs[k], expected[k])
        for k in range(dim)
        if abs(nums[k] - expected[k]) > 1e-6 or abs(nums[k] - py_obs[k]) > 1e-6
    ]
    if bad:
        print(f"FAIL  [{robot}] {len(bad)} 个分量不一致，前 5 个：")
        for k, y, p, e in bad[:5]:
            print(f"        obs[{k}]: yac={y} python={p} expected={e}")
        ok = False
    else:
        print(
            f"PASS  [{robot}] obs 维度 {dim}，yac 与 Python 逐位一致"
            f"（顺序敏感夹具 1..{dim}）"
        )
    return ok


def main() -> int:
    results = [run_one(r) for r in ("miniduck-L", "miniduck-S")]
    print(f"\n{sum(results)}/{len(results)} 通过" + ("" if all(results) else " —— 存在不一致"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
