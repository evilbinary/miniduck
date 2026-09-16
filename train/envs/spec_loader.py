# -*- coding: utf-8 -*-
"""train/envs/spec_loader.py — 以「受限子集」方式读取 mind/spec/*.yac。

设计约束（见 DESIGN 第 4 章）：
    spec/*.yac 只允许 `let` 绑定 + 标量/列表字面量 + 顶层尾表达式 `()`。
    yac 侧直接执行这些文件（真机 build_obs）；Python 侧用本模块**安全解析**
    同一份定义——只做字面量解析，不执行任何代码，因此 spec 里不可能有
    任意代码被训练侧执行。

解析失败一律抛 SpecError 并给出文件与偏移，便于定位。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "mind" / "spec"

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class SpecError(Exception):
    """spec 文件不满足受限子集约定。"""


#: 跨文件引用前缀：`"@robot.geometry.base_height_m"` 表示取
#: body/robot.yaml 中 robots.<当前机型> 下的字段。几何量归 robot.yaml，
#: spec 只引用不重复硬编码（见 DESIGN 4.3）。
ROBOT_REF = "@robot."


def resolve_refs(value: Any, robot_cfg: dict, where: str = "") -> Any:
    """递归解析 spec 值中的跨文件引用；解析不到即报错（不静默取默认值）。"""
    if isinstance(value, str) and value.startswith(ROBOT_REF):
        path = value[len(ROBOT_REF):].split(".")
        cur: Any = robot_cfg
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                raise SpecError(
                    f"{where}: 引用 {value!r} 无法解析"
                    f"（在 robots.<机型>.{k} 处断裂，请检查 body/robot.yaml）"
                )
            cur = cur[k]
        return cur
    if isinstance(value, list):
        return [resolve_refs(v, robot_cfg, where) for v in value]
    return value


class SpecParser:
    def __init__(self, src: str, filename: str) -> None:
        self.s = src
        self.i = 0
        self.n = len(src)
        self.f = filename

    # ---------------- 基础 ----------------
    def error(self, msg: str) -> None:
        line = self.s.count("\n", 0, self.i) + 1
        raise SpecError(f"{self.f}:{line}: {msg}")

    def skip(self) -> None:
        """跳过空白、`--` 行注释与 `/* */` 块注释。"""
        while self.i < self.n:
            c = self.s[self.i]
            if c.isspace():
                self.i += 1
            elif self.s.startswith("--", self.i):
                j = self.s.find("\n", self.i)
                self.i = self.n if j < 0 else j + 1
            elif self.s.startswith("/*", self.i):
                j = self.s.find("*/", self.i + 2)
                if j < 0:
                    self.error("未闭合的块注释")
                self.i = j + 2
            else:
                return

    def peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    # ---------------- 字面量 ----------------
    def parse_literal(self) -> Any:
        self.skip()
        c = self.peek()
        if c == "":
            self.error("表达式意外结束")
        if c == '"':
            return self.parse_string()
        if c == "[":
            return self.parse_list()
        if c == "-" or c.isdigit():
            return self.parse_number()
        word = self.parse_name()
        if word == "true":
            return True
        if word == "false":
            return False
        if word == "nil":
            return None
        self.error(f"受限子集不支持字面量 {word!r}（只允许标量 / 列表字面量）")

    def parse_string(self) -> str:
        assert self.peek() == '"'
        self.i += 1
        start = self.i
        while self.i < self.n and self.s[self.i] != '"':
            if self.s[self.i] == "\\":
                self.i += 1
            self.i += 1
        if self.i >= self.n:
            self.error("未闭合的字符串")
        out = self.s[start:self.i]
        self.i += 1
        return out

    def parse_number(self) -> float | int:
        m = _NUM_RE.match(self.s, self.i)
        if not m:
            self.error("无法解析的数字")
        self.i = m.end()
        txt = m.group(0)
        return float(txt) if ("." in txt or "e" in txt or "E" in txt) else int(txt)

    def parse_list(self) -> list:
        assert self.peek() == "["
        self.i += 1
        out: list = []
        while True:
            self.skip()
            if self.peek() == "]":
                self.i += 1
                return out
            out.append(self.parse_literal())
            self.skip()
            if self.peek() == ",":
                self.i += 1
                continue
            if self.peek() == "]":
                self.i += 1
                return out
            self.error("列表中期望 ',' 或 ']'")

    def parse_name(self) -> str:
        m = _NAME_RE.match(self.s, self.i)
        if not m:
            self.error("期望标识符")
        self.i = m.end()
        return m.group(0)

    # ---------------- 顶层 ----------------
    def parse_bindings(self) -> dict[str, Any]:
        """解析全部 `let name = <literal> in` 绑定；遇到函数绑定即报错。"""
        out: dict[str, Any] = {}
        while True:
            self.skip()
            if self.i >= self.n:
                return out
            if self.s.startswith("()", self.i):          # 顶层尾表达式
                self.i += 2
                continue
            if not self.s.startswith("let", self.i):
                self.error("受限子集只允许 `let` 绑定与顶层尾表达式 `()`")
            self.i += 3
            self.skip()
            name = self.parse_name()
            self.skip()
            if self.peek() != "=":
                self.error(
                    f"受限子集不支持函数绑定 `let {name}(...)`；"
                    "spec 文件只允许字面量绑定"
                )
            self.i += 1
            val = self.parse_literal()
            self.skip()
            if not self.s.startswith("in", self.i):
                self.error(f"绑定 {name} 之后期望 `in`")
            self.i += 2
            out[name] = val


def load_spec_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SpecError(f"spec 文件不存在：{path}")
    return SpecParser(path.read_text(encoding="utf-8"), path.name).parse_bindings()


class Spec:
    """按文件名聚合的全部 spec 绑定，提供带默认值的读取接口。"""

    def __init__(self, spec_dir: Path = SPEC_DIR) -> None:
        self.dir = spec_dir
        self.files: dict[str, dict[str, Any]] = {}
        for p in sorted(spec_dir.glob("*.yac")):
            if p.name.endswith(".generated.yac"):
                continue                                  # 生成物由 robot.yaml 派生
            self.files[p.stem] = load_spec_file(p)

    def get(self, key: str, default: Any = None) -> Any:
        """跨文件按键查找；同名键冲突时报错（避免静默取错值）。"""
        hits = [(f, v[key]) for f, v in self.files.items() if key in v]
        if not hits:
            if default is not None:
                return default
            raise SpecError(f"spec 中找不到键 {key!r}")
        if len(hits) > 1:
            raise SpecError(f"键 {key!r} 在多处定义：{ [f for f, _ in hits] }")
        return hits[0][1]

    def has(self, key: str) -> bool:
        return any(key in v for v in self.files.values())

    def obs_segments(self) -> list[tuple[str, int | str]]:
        return [(n, d) for n, d, *_ in self.get("obs_segments")]

    def reward_terms(self) -> dict[str, float]:
        return {row[0]: row[2] for row in self.get("reward_terms")}

    def reward_refs(self) -> dict[str, Any]:
        """原始值（可能含 @robot.* 引用）。"""
        return {row[0]: row[1] for row in self.get("reward_refs")}

    def reward_refs_for(self, robot_cfg: dict) -> dict[str, Any]:
        """解析跨文件引用后的参考量。robot_cfg = robot.yaml 的 robots.<机型>。"""
        raw = self.reward_refs()
        return {
            k: resolve_refs(v, robot_cfg, where=f"reward_refs.{k}")
            for k, v in raw.items()
        }

    def refs_pending(self) -> dict[str, str]:
        """返回仍含 @robot.* 引用的键 → 便于校验是否全部可解析。"""
        return {
            k: v
            for k, v in self.reward_refs().items()
            if isinstance(v, str) and v.startswith(ROBOT_REF)
        }

    def randomization(self) -> dict[str, Any]:
        return {row[0]: row[1] for row in self.get("domain_randomization")}

    def ppo(self) -> dict[str, Any]:
        return {row[0]: row[1] for row in self.get("ppo")}

    def action_scale(self) -> float:
        return float(self.get("action_scale_rad"))

    def control_hz(self) -> int:
        return int(self.get("control_hz"))


if __name__ == "__main__":
    sp = Spec()
    print("spec 文件:", ", ".join(sorted(sp.files)))
    print("观测段:", sp.obs_segments())
    print("奖励项:", list(sp.reward_terms()))
    print("action_scale:", sp.action_scale(), " control_hz:", sp.control_hz())
    print("PPO envs:", sp.ppo()["envs"], " 随机化项数:", len(sp.randomization()))
