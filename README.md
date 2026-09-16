# miniduck

双足仿生鸭子机器人：3D 建模身体 + 14 个总线舵机 + 强化学习步态，**用 yac 语言
在 yiyiya OS 上跑真机闭环**。

设计文档见 [`docs/DESIGN.md`](docs/DESIGN.md)（唯一权威，改了设计先改文档）。

| 版本 | 体格 | 舵机 | 关节分配 | 策略 |
|------|------|------|----------|------|
| **miniduck-L** 大型版 | 同 microduck 量级（规格待补） | 14 × XL330 | 双腿 14 DOF 全给腿 | obs 61 / act 14 |
| **miniduck-S** 桌面版 | 身高 12–18 cm，整机 200–400 g | 14 × STS3032 | 腿 6 + 颈/头 4 + 喙 1 + 尾/翅 3 | obs 37 / act 6（表情关节走脚本） |

两版共用同一套软件栈，只有 `body/robot.yaml` 与关节分配不同。

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | **3.12**（推荐项目 venv） | 生成器、校验、训练侧 |
| PyYAML | 6.x | 读 `body/robot.yaml` |
| **yc 编译器**（必需） | 兄弟目录 `../yac` | 编译执行 `mind/**/*.yac`：ANF → LIR → 机器码 |
| yac 解释器（可选） | 同上 | 仅在需要 ANF/CPS **双解释器互证**时使用 |
| MuJoCo | 3.2.2 | 物理后端、`check_physics.py` |
| **BAM**（Better Actuator Models） | `better-actuator-models[mujoco]` | 执行器拟真模型（电压控制 + 直流电机 + M1–M6 摩擦 + 掉压） |

### 项目虚拟环境（推荐）

**BAM 要求 Python ≥3.12,<3.13**，因此项目用独立 venv：

```sh
cd miniduck
python3.12 -m venv .venv                  # 或改用本机任意 3.12 解释器
./.venv/Scripts/python -m pip install numpy pyyaml mujoco==3.2.2 "better-actuator-models[mujoco]"
```

之后所有命令把 `python` 换成 `./.venv/Scripts/python`（Linux/macOS 为 `.venv/bin/python`）。
不装 BAM 也能跑：环境会自动回退到 `pd` 模型。

> **本机已知坑**：
> 1. `pip` 与 `python` 可能指向不同环境（如 Anaconda 的 pip + Windows Store 的 python）
>    → 一律用 `python -m pip install ...`。
> 2. 在 Anaconda 的 Python 3.12 下，**mujoco ≥3.3 的 DLL 加载失败**
>    （`WinError 1114`），3.2.2 正常 → venv 里固定 3.2.2。
> 3. BAM 在 Python 3.11 下装不上（requires-python 限制）。

先构建一次（`make` 会同时产出 `yac` 与 `yc`）：

```sh
cd ../yac
make              # 产物 yac + yc（Windows 下为 yac.exe / yc.exe）
make yc           # 只构建编译器
```

> **注意 `yc` 的调用方式**：`yc file.yac` 只编译不运行；要执行必须带
> `--both`（ANF→CPS→ANF 后编译执行）或 `--cps`。另外 `yc --verify-lir`
> 可校验 LIR 不变量，`yc --dump-lir` / `--dump-asm` 可看中间产物与机器码。

如果 yc 不在默认位置，用环境变量指定（`verify_gen.py`、`obs_parity.py` 会读它）：

```sh
export YC_BIN=/path/to/yc          # Git Bash；PowerShell 用 $env:YC_BIN="..."
```

---

## 快速开始

在仓库根目录（`miniduck/`）执行：

```sh
# 1) 校验关节定义（不生成任何文件）
python body/tools/gen.py --check

# 2) 生成仿真描述与 yac 常量
python body/tools/gen.py

# 3) 校验生成产物（URDF/MJCF 结构 + yc 编译执行 + LIR 不变量）
python body/tools/verify_gen.py

# 3.5) 物理健全性检查（需 MuJoCo；结构校验抓不到的问题在这里暴露）
python body/tools/check_physics.py

# 3.6) 可选：重新求解静态站立姿态（改了几何/质量后需要）
python body/tools/solve_stance.py

# 4) 纯函数核心冒烟测试（yc 编译执行；--both 覆盖 ANF→CPS→ANF 往返）
../yac/yc.exe --both mind/tests/robotd_core_smoke.yac           # 或 "$YC_BIN"
../yac/yc.exe --verify-lir mind/tests/robotd_core_smoke.yac     # 可选：LIR 不变量

# 5) Python 与 yc（编译执行）两侧 build_obs 逐位对拍
python train/envs/tests/obs_parity.py

# 6) 训练环境自检（MuJoCo 后端，零动作站立 200 步）
cd train/envs && python duck_env.py --backend mujoco
python duck_env.py --robot miniduck-L --backend mujoco    # L 版
python duck_env.py --backend null                         # 占位后端（无 MuJoCo 时）

# 7) 执行器模型切换（见 DESIGN 3.4）
python duck_env.py --actuator pd     # 力矩级 PD 基线（快，算法调试）
python duck_env.py --actuator bam    # BAM：电压+直流电机+M1-M6 摩擦+掉压（训练默认）
python duck_env.py --actuator auto   # 按 body/robot.yaml 的 actuators.default
python ../../body/tools/check_physics.py --actuator all   # 两种模型都验静置稳定性
```

### 一键跑全部

```sh
cd /path/to/miniduck && \
python body/tools/gen.py && \
python body/tools/verify_gen.py && \
python body/tools/check_physics.py > /dev/null && \
"$YC_BIN" --both mind/tests/robotd_core_smoke.yac > /dev/null && \
python train/envs/tests/obs_parity.py && \
python -c "import sys; sys.path.insert(0,'train/envs'); from duck_env import StandEnv; StandEnv('miniduck-S').reset()" && \
echo "ALL GREEN"
```

### 期望输出

```
校验通过：0 error，0 warning（TBD 占位）           # gen.py
18 passed, 0 failed                               # verify_gen.py（14 结构 + 2 编译执行 + 2 LIR）
PASS × 12                                         # 冒烟测试（yc --both）
PASS  [miniduck-L] obs 维度 61，yc 编译执行结果与 Python 逐位一致
PASS  [miniduck-S] obs 维度 37，yc 编译执行结果与 Python 逐位一致
机型 miniduck-S: 腿关节 6, obs 37, act 6           # duck_env.py
```

---

## 目录结构

```
body/                 「身体」硬件与建模
  robot.yaml          ★ 关节定义单一来源（DOF/限位/零位/舵机 ID/几何）
  tools/gen.py        robot.yaml → URDF/MJCF/joints.yac/consts.yac（含 10 条校验）
  tools/verify_gen.py 生成产物回归测试（结构层）
  tools/check_physics.py MuJoCo 物理健全性检查（质量账/站立高度/落地）
  urdf/  mjcf/        生成物（不入库，由 gen.py 重建）
mind/                 「心智」真机软件（yac，运行在 yiyiya OS）
  spec/               ★ 规范单一来源：obs / reward / action / randomize / ppo
  robotd/             50 Hz 控制循环（纯函数核心 + 宿主边界原语）
  tests/              yc 侧测试（robotd_core_smoke.yac）
  mindd/  updaterd/   高层意图、OTA（待实现）
train/                训练侧（Python）
  envs/spec_loader.py 受限子集解析 mind/spec/*.yac（不执行代码）
  envs/actuators.py   ★ 执行器模型层：PdTorque ↔ BAM（含上游垫片，见 DESIGN 3.4）
  envs/duck_env.py    环境：MujocoPhysics 后端 + build_obs / reward / StandEnv
  envs/tests/         obs 双侧对拍（发布门禁的提前版）
  ppo/  export.py     待实现
verify/               发布门禁：自研 runtime vs ONNX 逐帧比对（待实现）
docs/DESIGN.md        设计文档（唯一权威）
```

### 两处「单一来源」

1. **`body/robot.yaml`** — 关节名称/顺序/限位/零位/几何。生成 URDF、MJCF、
   `mind/spec/joints.generated.yac`、`mind/robotd/consts.generated.yac`。
2. **`mind/spec/*.yac`** — 观测段表、奖励权重、动作映射、域随机化、PPO 超参。
   yc 侧编译执行、Python 侧受限解析，一处定义两端消费。

> 改任何数值都改这两处，然后重跑 `gen.py`；**不要**手改 `*.generated.yac`、
> `body/urdf/`、`body/mjcf/`（它们在 `.gitignore` 里，改了也会被覆盖）。

---

## 训练（Python + GPU）

训练侧尚在搭建中，规划如下（见 DESIGN 第 4、7 章）：

```sh
# 目标形态（尚未实现）
python train/ppo/train.py --robot miniduck-S --stage stage_1_stand
python train/export.py --run train/runs/<run>       # 双产物：ONNX + policy_blob
python verify/policy_forward.py --blob ...          # 与 ONNX golden 帧逐帧比对
```

当前已可用的部分：环境骨架、观测构建、奖励计算（均由 spec 驱动）。
已完成：环境骨架、观测构建、奖励计算（均由 spec 驱动）、**MuJoCo 后端**
（零动作可稳站 4 s）、静态姿态求解、**执行器模型层**（PD ↔ BAM 可切换，
两版机型 × 两种模型静置稳定性均通过）。

未完成的：**MJX/Warp GPU 并行后端**（大规模并行时替换 MuJoCo）、PPO 训练循环、
ONNX/blob 导出、真机 `robotd` 宿主运行时、**执行器台架辨识**
（BAM 内置参数：XL330 有 M1–M6 但对象是 M288；STS3215 只有 M1，STS3032 无内置）。

---

## 真机部署（规划）

1. 悬挂测试确认关节方向与限位（`robot.yaml` 的 `sign_check`），
2. 回填实测值：`xyz`、`mass_g`、`baud`、`safe_pose`、`geometry.base_height_m`，
3. 重跑 `gen.py` → 更新 `mind/robotd/consts.generated.yac`，
4. 低增益站立 → 闭环行走 → OTA（`updaterd`，含回滚槽位）。

---

## 常见问题

**`找不到 yc 编译器`** — 先 `cd ../yac && make`，或设 `YC_BIN` 指向可执行文件。

**`yc` 跑完没有任何输出** — `yc file.yac` 只编译不运行（这是它的语义），
执行要加 `--both` 或 `--cps`。同理，`yc --both` 不会像 `yac` 那样回显顶层
表达式的值（`()`），脚本里不要依赖这一点判成功，判退出码即可。

**生成的 URDF 里质量/惯量都是占位值** — 正常，`robot.yaml` 中相关字段标了 `TBD`
（等 CAD 回填）；生成器只对**结构**（拓扑、限位、维度、几何自洽）做强校验。

**改了 `robot.yaml` 但测试还在用旧值** — 生成物需要重建：`python body/tools/gen.py`。

**测试报 `SpecError`** — spec 文件违反了受限子集约定（只允许 `let` 绑定 +
标量/列表字面量，且必须以尾表达式 `()` 结束），错误信息里带文件与行号。

**`h_ref_m` 引用解析失败** — 检查 `body/robot.yaml` 的
`robots.<机型>.geometry.base_height_m` 是否存在；spec 里写的是
`"@robot.geometry.base_height_m"`，几何量不在 spec 里硬编码。

**`[warn] BAM 不可用…回退 pd`** — 当前解释器 <3.12 或未装 BAM。
用项目 venv（`./.venv/Scripts/python`）运行；若确实不需要拟真模型，
把 `robot.yaml` 的 `actuators.default` 改成 `pd` 即可，不必每次都回退。

**`[warn] stance_pose 是在 xx 下求解的，当前执行器为 yy`** — 静态站立姿态
与执行器模型相关（实测 L 版：PD 下需 hip/ankle=0.10/−0.10，BAM 下需 0.20/−0.20）。
执行 `python body/tools/solve_stance.py --actuator <当前模型>` 重解并更新
`stance_pose` 与腿关节 `default`，再把 `actuators.pose_solved_with` 改成同一模型。
