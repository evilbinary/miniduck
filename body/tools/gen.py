#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""body/tools/gen.py — 从 body/robot.yaml 生成仿真描述与 yac 侧常量。

单一来源：body/robot.yaml
产物：
    body/urdf/{robot}.urdf                 训练仿真（生成物，不入库）
    body/mjcf/{robot}.xml                  MuJoCo（生成物，不入库）
    mind/spec/joints.generated.yac         yac 侧关节映射表
    mind/robotd/consts.generated.yac       真机运行时常量

用法：
    python body/tools/gen.py            # 校验 + 生成全部产物
    python body/tools/gen.py --check    # 只校验（CI 用），有 error 则返回 1

校验分为两级：errors 阻断生成；warnings 仅提示（TBD 占位等）。
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOT_YAML = ROOT / "body" / "robot.yaml"

ERRORS: list[str] = []
WARNINGS: list[str] = []


def err(msg: str) -> None:
    ERRORS.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


# --------------------------------------------------------------------------
# 读取与静态校验
# --------------------------------------------------------------------------
def load_cfg() -> dict:
    if not ROBOT_YAML.exists():
        err(f"找不到定义文件：{ROBOT_YAML}")
        return {}
    with ROBOT_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def obs_dim_from_formula(formula: str, n: int) -> int:
    """把 "3+3+3+N+N+N+2+2+1+3+2" 里的 N 替换为 n 后求和。"""
    expr = formula.replace("N", str(n))
    total = 0
    for term in expr.split("+"):
        total += int(term.strip())
    return total


def validate(cfg: dict) -> None:
    rules = cfg.get("validate", {})
    robots = cfg.get("robots", {})
    profiles = cfg.get("servo_profiles", {})

    for robot_name, robot in robots.items():
        joints = robot.get("joints", [])
        links = robot.get("links", [])
        policy = robot.get("policy", {})
        by_name = {j["name"]: j for j in joints}
        link_names = {l["name"] for l in links}

        # 1) 关节数量
        expect = rules.get("joint_count", {}).get(robot_name)
        if expect is not None and len(joints) != expect:
            err(f"[{robot_name}] 关节数 {len(joints)} != 预期 {expect}")

        # 2) ID 唯一 + 范围
        ids = [j["id"] for j in joints]
        if rules.get("id_unique", True) and len(set(ids)) != len(ids):
            err(f"[{robot_name}] 关节 ID 重复：{sorted(ids)}")
        lo_id, hi_id = rules.get("id_in_range", [1, 14])
        bad = [i for i in ids if not (lo_id <= i <= hi_id)]
        if bad:
            err(f"[{robot_name}] 关节 ID 越界 {bad}（允许 {lo_id}..{hi_id}）")

        # 3) 策略引用有效性 + 动作维度一致
        leg_joints = policy.get("leg_joints", [])
        if rules.get("policy_leg_joints_subset_of_joints", True):
            missing = [n for n in leg_joints if n not in by_name]
            if missing:
                err(f"[{robot_name}] policy.leg_joints 引用了不存在的关节：{missing}")
        if rules.get("act_dim_matches_leg_joints", True):
            if policy.get("act_dim") != len(leg_joints):
                err(f"[{robot_name}] act_dim={policy.get('act_dim')} != len(leg_joints)={len(leg_joints)}")

        # 4) obs 维度公式
        formula = rules.get("obs_dim_formula", "3+3+3+N+N+N+2+2+1+3+2")
        if policy.get("obs_dim") is not None:
            calc = obs_dim_from_formula(formula, policy.get("act_dim", len(leg_joints)))
            if calc != policy["obs_dim"]:
                err(f"[{robot_name}] obs_dim={policy['obs_dim']} != 公式计算 {calc}（N={len(leg_joints)}）")

        # 5) 限位有序 + 默认位在限位内
        for j in joints:
            if not (j["lo"] < j["hi"]):
                err(f"[{robot_name}.{j['name']}] 限位无序：lo={j['lo']} hi={j['hi']}")
            if rules.get("default_within_limits", True) and not (j["lo"] <= j["default"] <= j["hi"]):
                err(f"[{robot_name}.{j['name']}] default={j['default']} 不在 [{j['lo']}, {j['hi']}] 内")

        # 6) 连杆引用有效 + parent != child
        if rules.get("link_refs_valid", True):
            for j in joints:
                if j["parent"] not in link_names:
                    err(f"[{robot_name}.{j['name']}] parent 连杆不存在：{j['parent']}")
                if j["child"] not in link_names:
                    err(f"[{robot_name}.{j['name']}] child 连杆不存在：{j['child']}")
                if j["parent"] == j["child"]:
                    err(f"[{robot_name}.{j['name']}] parent == child（自环）：{j['parent']}")
            # 连杆的 parent_joint 必须与其关节的 child 一致
            for l in links:
                pj = l.get("parent_joint")
                if pj is None:
                    continue
                if pj not in by_name:
                    err(f"[{robot_name}] 连杆 {l['name']} 引用了不存在的关节 {pj}")
                elif by_name[pj]["child"] != l["name"]:
                    err(f"[{robot_name}] 连杆 {l['name']} 的 parent_joint={pj} 但该关节 child={by_name[pj]['child']}")

        # 7) safe_pose 长度
        sp = robot.get("safe_pose", [])
        if len(sp) != len(joints):
            err(f"[{robot_name}] safe_pose 长度 {len(sp)} != 关节数 {len(joints)}")

        # 8) 连杆树：单根、无环、连通
        tree_err = check_tree(robot_name, links, by_name)
        if tree_err:
            err(tree_err)

        # 9) 舵机档存在
        if robot.get("servo") not in profiles:
            err(f"[{robot_name}] servo={robot.get('servo')} 未在 servo_profiles 中定义")

        # warnings：TBD 占位
        n_tbd = 0
        for j in joints:
            if j.get("xyz") is None:
                n_tbd += 1
        if n_tbd:
            warn(f"[{robot_name}] {n_tbd} 个关节缺 xyz（安装偏移），将由 CAD 回填")
        for l in links:
            if l.get("mass_g") is None:
                warn(f"[{robot_name}.{l['name']}] 缺 mass_g，仿真惯量将为默认值")
        prof = profiles.get(robot.get("servo"), {})
        for k in ("baud", "velocity_rad_s"):
            if k not in prof:
                warn(f"[{robot_name}] 舵机档缺 {k}，使用默认值")


def check_tree(robot_name: str, links: list, by_name: dict) -> str | None:
    """校验连杆树：恰好一个根、无环、全连通。返回错误串或 None。"""
    parent_joint = {l["name"]: l.get("parent_joint") for l in links}
    roots = [n for n, pj in parent_joint.items() if pj is None]
    if len(roots) != 1:
        return f"[{robot_name}] 根连杆数量 {len(roots)} != 1（{roots}）"
    root = roots[0]
    children: dict[str, list[str]] = {l["name"]: [] for l in links}
    for name, pj in parent_joint.items():
        if pj is None:
            continue
        parent = by_name[pj]["parent"]
        children.setdefault(parent, []).append(name)
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n in seen:
            return f"[{robot_name}] 连杆树存在环：{n} 被重复访问"
        seen.add(n)
        stack.extend(children.get(n, []))
    unreachable = set(parent_joint) - seen
    if unreachable:
        return f"[{robot_name}] 连杆树不连通，孤立连杆：{sorted(unreachable)}"
    return None


# --------------------------------------------------------------------------
# 生成：URDF / MJCF
# --------------------------------------------------------------------------
def child_map(robot: dict) -> dict[str, list[dict]]:
    """parent 连杆名 -> [{'joint': j, 'link': l}]"""
    by_name = {j["name"]: j for j in robot["joints"]}
    out: dict[str, list[dict]] = {l["name"]: [] for l in robot["links"]}
    for l in robot["links"]:
        pj = l.get("parent_joint")
        if pj is None:
            continue
        out.setdefault(by_name[pj]["parent"], []).append({"joint": by_name[pj], "link": l})
    return out


def inertia_of(mass_g: float) -> float:
    """占位惯量：按半径 20 mm 的球体估算 I = 2/5·m·r²。待 CAD 回填。"""
    m = (mass_g or 10.0) / 1000.0
    return round(0.4 * m * (0.02 ** 2), 10)


def gen_urdf(robot_name: str, robot: dict, cfg: dict) -> str:
    prof = cfg["servo_profiles"][robot["servo"]]
    effort = prof["stall_torque_nm"]
    velocity = prof["velocity_rad_s"]
    cmap = child_map(robot)
    out = [
        '<?xml version="1.0"?>',
        f'<!-- 自动生成：body/tools/gen.py，勿手改。来源 body/robot.yaml v{cfg["version"]} -->',
        f'<robot name="{robot_name}">',
    ]
    for l in robot["links"]:
        mass = (l.get("mass_g") or 10.0) / 1000.0
        i = inertia_of(l.get("mass_g"))
        out += [
            f'  <link name="{l["name"]}">',
            "    <inertial>",
            f'      <mass value="{mass:.5f}"/>  <!-- TBD: 由 CAD 回填 -->',
            f'      <inertia ixx="{i}" ixy="0" ixz="0" iyy="{i}" iyz="0" izz="{i}"/>',
            "    </inertial>",
            "  </link>",
        ]
    for parent, kids in cmap.items():
        for k in kids:
            j, l = k["joint"], k["link"]
            jtype = "revolute"
            xyz = j.get("xyz") or [0.0, 0.0, 0.0]
            out += [
                f'  <joint name="{j["name"]}" type="{jtype}">',
                f'    <parent link="{parent}"/>',
                f'    <child link="{l["name"]}"/>',
                f'    <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="0 0 0"/>  <!-- TBD: 由 CAD 回填 -->',
                f'    <axis xyz="{j["axis"][0]} {j["axis"][1]} {j["axis"][2]}"/>',
                f'    <limit lower="{j["lo"]}" upper="{j["hi"]}" effort="{effort}" velocity="{velocity}"/>',
            ]
            if j.get("passive"):
                out.append(f'    <dynamics damping="0.05" friction="0.02"/>  <!-- 被动关节：无驱动 -->')
            out.append("  </joint>")
    out.append("</robot>")
    return "\n".join(out) + "\n"


def gen_mjcf(robot_name: str, robot: dict, cfg: dict) -> str:
    prof = cfg["servo_profiles"][robot["servo"]]
    kp = 8.0
    cmap = child_map(robot)
    by_link = {l["name"]: l for l in robot["links"]}
    roots = [n for n, l in by_link.items() if l.get("parent_joint") is None]

    out = [
        f'<mujoco model="{robot_name}">',
        f'  <!-- 自动生成：body/tools/gen.py，勿手改。来源 body/robot.yaml v{cfg["version"]} -->',
        '  <compiler angle="radian" autolimits="true"/>',
        '  <option timestep="0.002"/>',
        '  <default>',
        '    <joint damping="0.05" armature="0.01"/>',
        f'    <position kp="{kp}" kv="0.5"/>',
        '    <geom type="capsule" size="0.008" density="800"/>',
        '  </default>',
        '  <worldbody>',
    ]

    def emit_body(link_name: str, indent: int) -> None:
        l = by_link[link_name]
        pad = "  " * indent
        pos = [0.0, 0.0, 0.0]
        # 关节的安装偏移记在关节上：此处用父关节的 xyz 作为 body pos
        pj = l.get("parent_joint")
        if pj:
            by_name = {j["name"]: j for j in robot["joints"]}
            pos = by_name[pj].get("xyz") or pos
        out.append(f'{pad}<body name="{link_name}" pos="{pos[0]} {pos[1]} {pos[2]}">')
        if pj is None:
            out.append(f"{pad}  <freejoint/>")
        else:
            by_name = {j["name"]: j for j in robot["joints"]}
            j = by_name[pj]
            out.append(
                f'{pad}  <joint name="{j["name"]}" type="hinge" '
                f'axis="{j["axis"][0]} {j["axis"][1]} {j["axis"][2]}" '
                f'range="{j["lo"]} {j["hi"]}"/>'
            )
        mass = (l.get("mass_g") or 10.0) / 1000.0
        out.append(f'{pad}  <geom name="{link_name}_geom" mass="{mass:.5f}"/>  <!-- TBD: 由 CAD 回填 -->')
        for k in cmap.get(link_name, []):
            if k["link"]["name"] != link_name:
                emit_body(k["link"]["name"], indent + 1)
        out.append(f"{pad}</body>")

    for r in roots:
        emit_body(r, 3)

    out += ["  </worldbody>", "  <actuator>"]
    for j in robot["joints"]:
        if j.get("passive"):
            continue
        out.append(
            f'    <position name="{j["name"]}_pos" joint="{j["name"]}" '
            f'ctrlrange="{j["lo"]} {j["hi"]}"/>  <!-- 输入 = target_pos（见 DESIGN 4.2） -->'
        )
    out += ["  </actuator>", "</mujoco>"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# 生成：yac 侧常量
# --------------------------------------------------------------------------
def fmt_list(xs) -> str:
    return "[" + ", ".join(f"{x:g}" if isinstance(x, (int, float)) else f'"{x}"' for x in xs) + "]"


def gen_joints_yac(cfg: dict) -> str:
    out = [
        "-- 自动生成：body/tools/gen.py，勿手改。来源 body/robot.yaml v" + str(cfg["version"]),
        "-- 关节映射表：真机与训练共用。名称/ID/限位/默认位/角色。",
        "",
    ]
    for robot_name, robot in cfg["robots"].items():
        var = robot_name.replace("-", "_")
        rows = []
        for j in robot["joints"]:
            rows.append(
                f'  ["{j["name"]}", {j["id"]}, "{j["role"]}", {j["lo"]:g}, {j["hi"]:g}, '
                f'{j["default"]:g}, {str(j.get("passive", False)).lower()}, {str(j.get("expressive", False)).lower()}]'
            )
        out.append(f"-- {robot_name}：{len(robot['joints'])} 关节")
        out.append(f'-- [name, id, role, lo, hi, default_pos, passive, expressive]')
        out.append(f"let joints_{var} = [")
        out.append(",\n".join(rows))
        out.append("] in")
        out.append(f'let policy_leg_joints_{var} = {fmt_list(robot["policy"]["leg_joints"])} in')
        out.append(f'let obs_dim_{var} = {robot["policy"]["obs_dim"]} in')
        out.append(f'let act_dim_{var} = {robot["policy"]["act_dim"]} in')
        out.append("")
    out.append("-- 顶层尾表达式")
    out.append("()")
    return "\n".join(out) + "\n"


def gen_consts_yac(cfg: dict) -> str:
    out = [
        "-- 自动生成：body/tools/gen.py，勿手改。来源 body/robot.yaml v" + str(cfg["version"]),
        "-- 真机运行时常量。robotd 按机型选择后缀（_miniduck_L / _miniduck_S）加载。",
        "-- 与 DESIGN 6.1 的对应关系：",
        "--   joint_ids    → loop 里 io_dxl_sync_read/write 的 ID 列表",
        "--   default_pos  → from_action 的 d 参数（14 维全关节）",
        "--   policy_default_pos / policy_joint_ids → S 版策略子集（6 维）",
        "--   safe_pose    → 异常分支的下发目标",
        "--   leg_lo/leg_hi→ safe_joint 的限位（取腿关节限位的最紧值）",
        "",
    ]
    for robot_name, robot in cfg["robots"].items():
        var = robot_name.replace("-", "_")
        joints = robot["joints"]
        by_name = {j["name"]: j for j in joints}
        leg = [by_name[n] for n in robot["policy"]["leg_joints"]]
        out.append(f"-- ===== {robot_name} =====")
        out.append(f'let joint_ids_{var} = {fmt_list([j["id"] for j in joints])} in')
        out.append(f'let joint_names_{var} = {fmt_list([j["name"] for j in joints])} in')
        out.append(f'let default_pos_{var} = {fmt_list([j["default"] for j in joints])} in')
        out.append(f'let safe_pose_{var} = {fmt_list(robot["safe_pose"])} in')
        out.append(f'let leg_lo_{var} = {min(j["lo"] for j in leg):g} in')
        out.append(f'let leg_hi_{var} = {max(j["hi"] for j in leg):g} in')
        out.append(f'let policy_joint_ids_{var} = {fmt_list([j["id"] for j in leg])} in')
        out.append(f'let policy_default_pos_{var} = {fmt_list([j["default"] for j in leg])} in')
        exp = [j for j in joints if j.get("expressive")]
        if exp:
            out.append(f'let expressive_joint_ids_{var} = {fmt_list([j["id"] for j in exp])} in')
        out.append("")
    out.append("-- 顶层尾表达式")
    out.append("()")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="robot.yaml → 仿真描述 + yac 常量")
    ap.add_argument("--check", action="store_true", help="只校验，不生成")
    args = ap.parse_args()

    cfg = load_cfg()
    if not cfg:
        for e in ERRORS:
            print(f"ERROR  {e}")
        return 1

    validate(cfg)

    for w in WARNINGS:
        print(f"WARN   {w}")
    for e in ERRORS:
        print(f"ERROR  {e}")

    if ERRORS:
        print(f"\n校验失败：{len(ERRORS)} 个 error，{len(WARNINGS)} 个 warning → 拒绝生成")
        return 1

    print(f"\n校验通过：0 error，{len(WARNINGS)} warning（TBD 占位）")
    if args.check:
        return 0

    gen = cfg["generate"]
    written = []
    for robot_name, robot in cfg["robots"].items():
        p = ROOT / gen["urdf"]["target"].format(robot=robot_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(gen_urdf(robot_name, robot, cfg), encoding="utf-8")
        written.append(p)

        p = ROOT / gen["mjcf"]["target"].format(robot=robot_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(gen_mjcf(robot_name, robot, cfg), encoding="utf-8")
        written.append(p)

    p = ROOT / gen["spec_joints"]["target"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(gen_joints_yac(cfg), encoding="utf-8")
    written.append(p)

    p = ROOT / gen["runtime_consts"]["target"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(gen_consts_yac(cfg), encoding="utf-8")
    written.append(p)

    print("生成产物：")
    for p in written:
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
