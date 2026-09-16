#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""body/tools/verify_gen.py — 校验 gen.py 的生成产物是否自洽（回归测试）。

检查项：
    URDF    XML 良构；link/joint 数量与 robot.yaml 一致；parent/child 均可解析
    MJCF    XML 良构；body 数量一致；hinge 关节数量一致；执行器数 = 非被动关节数
            （注意排除 <default> 块中的模板元素）
    yc      生成的两个 .yac 文件能被 **yc 编译器**编译执行，且通过 LIR 不变量校验
            （yac 是解释器，生产路径用 yc：ANF → LIR → 机器码）

用法：
    python body/tools/verify_gen.py            # 自动探测 ../yac/yc[.exe]
    YC_BIN=/path/to/yc python body/tools/verify_gen.py

返回 0 = 全部通过；1 = 有失败项。
"""

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOT_YAML = ROOT / "body" / "robot.yaml"
def _default_yc() -> str:
    """按平台探测兄弟目录里的 yc 编译器。"""
    for name in ("yc.exe", "yc"):
        p = ROOT.parent / "yac" / name
        if p.exists():
            return str(p)
    return str(ROOT.parent / "yac" / "yc.exe")


YC_BIN = os.environ.get("YC_BIN") or os.environ.get("YAC_BIN") or _default_yc()

FAILS: list[str] = []
OKS: list[str] = []


def ok(msg: str) -> None:
    OKS.append(msg)


def fail(msg: str) -> None:
    FAILS.append(msg)


def check(cond: bool, msg: str) -> None:
    ok(msg) if cond else fail(msg)


#: MuJoCo 执行器标签（我们是 <motor>：力矩语义，BAM 的硬性要求；见 DESIGN 3.4）
ACTUATOR_TAGS = ("motor", "position", "velocity", "general")


def body_and_joint_elements(root: ET.Element):
    """返回 (body 列表, 机器人关节列表, 执行器列表)，排除 <default> 模板块。"""
    default = root.find("default")

    def in_default(el: ET.Element) -> bool:
        if default is None:
            return False
        return any(c is el for c in default.iter())

    bodies = [b for b in root.iter("body") if not in_default(b)]
    joints = [j for j in root.iter("joint") if not in_default(j)]
    acts = [a for a in root.iter() if a.tag in ACTUATOR_TAGS and not in_default(a)]
    return bodies, joints, acts


def main() -> int:
    cfg = yaml.safe_load(ROBOT_YAML.read_text(encoding="utf-8"))
    gen = cfg["generate"]

    for robot_name, robot in cfg["robots"].items():
        n_joints = len(robot["joints"])
        n_links = len(robot["links"])
        n_passive = sum(1 for j in robot["joints"] if j.get("passive"))

        # ---- URDF ----
        up = ROOT / gen["urdf"]["target"].format(robot=robot_name)
        if not up.exists():
            fail(f"{up.relative_to(ROOT)} 不存在（先运行 gen.py）")
            continue
        urdf = ET.parse(up).getroot()
        links = urdf.findall("link")
        joints = urdf.findall("joint")
        check(len(links) == n_links, f"[{robot_name}] URDF link {len(links)} == {n_links}")
        check(len(joints) == n_joints, f"[{robot_name}] URDF joint {len(joints)} == {n_joints}")
        names = {l.get("name") for l in links}
        dangling = [
            j.get("name")
            for j in joints
            if j.find("parent").get("link") not in names or j.find("child").get("link") not in names
        ]
        check(not dangling, f"[{robot_name}] URDF parent/child 全部可解析（悬空={dangling}）")

        # ---- MJCF ----
        mp = ROOT / gen["mjcf"]["target"].format(robot=robot_name)
        mjcf = ET.parse(mp).getroot()
        bodies, mjoints, acts = body_and_joint_elements(mjcf)
        check(len(bodies) == n_links, f"[{robot_name}] MJCF body {len(bodies)} == {n_links}")
        check(len(mjoints) == n_joints, f"[{robot_name}] MJCF hinge joint {len(mjoints)} == {n_joints}")
        check(
            len(acts) == n_joints - n_passive,
            f"[{robot_name}] MJCF actuator {len(acts)} == 非被动关节 {n_joints - n_passive}",
        )
        jnames = {j.get("name") for j in mjoints}
        bad_actor = [a.get("joint") for a in acts if a.get("joint") not in jnames]
        check(not bad_actor, f"[{robot_name}] MJCF actuator 引用的关节都存在（坏={bad_actor}）")

    # ---- yac 产物 ----
    yac_files = [
        ROOT / gen["spec_joints"]["target"],
        ROOT / gen["runtime_consts"]["target"],
    ]
    yac_available = Path(YC_BIN).exists()
    if not yac_available:
        fail(f"找不到 yc 编译器：{YC_BIN}（可用 YC_BIN 指定；注意 yc 无参数只编译不运行）")
    else:
        for f in yac_files:
            # --both：ANF→CPS→ANF 往返后编译执行，额外覆盖 CPS 转换路径
            r = subprocess.run([YC_BIN, "--both", str(f)], capture_output=True, text=True)
            check(
                r.returncode == 0,
                f"{f.relative_to(ROOT)} yc --both 编译并执行通过（覆盖 CPS 往返路径）",
            )
            # --verify-lir：校验 LIR 不变量（docs/LIR.md §8）
            # 输出形如 "verify-lir: ok -- 0 error(s), 14 warning(s)"
            # 注意：不能子串匹配 "error"，摘要行自身就含 "0 error(s)"
            r2 = subprocess.run([YC_BIN, "--verify-lir", str(f)], capture_output=True, text=True)
            out = (r2.stdout or "") + (r2.stderr or "")
            m = re.search(r"(\d+)\s+error", out)
            n_err = int(m.group(1)) if m else -1
            check(
                r2.returncode == 0 and n_err == 0,
                f"{f.relative_to(ROOT)} LIR 不变量校验通过（error={n_err}）",
            )

    for m in OKS:
        print(f"PASS  {m}")
    for m in FAILS:
        print(f"FAIL  {m}")
    print(f"\n{len(OKS)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
