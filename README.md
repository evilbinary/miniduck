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
| Python | 3.11+ | 生成器、校验、训练侧 |
| PyYAML | 6.x（`pip install pyyaml`） | 读 `body/robot.yaml` |
| **yc 编译器**（必需） | 兄弟目录 `../yac` | 编译执行 `mind/**/*.yac`：ANF → LIR → 机器码 |
| yac 解释器（可选） | 同上 | 仅在需要 ANF/CPS **双解释器互证**时使用 |

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

# 4) 纯函数核心冒烟测试（yc 编译执行；--both 覆盖 ANF→CPS→ANF 往返）
../yac/yc.exe --both mind/tests/robotd_core_smoke.yac           # 或 "$YC_BIN"
../yac/yc.exe --verify-lir mind/tests/robotd_core_smoke.yac     # 可选：LIR 不变量

# 5) Python 与 yc（编译执行）两侧 build_obs 逐位对拍
python train/envs/tests/obs_parity.py

# 6) 训练环境自检（占位物理后端，跑 200 步 stand）
cd train/envs && python duck_env.py
```

### 一键跑全部

```sh
cd /path/to/miniduck && \
python body/tools/gen.py && \
python body/tools/verify_gen.py && \
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
  tools/verify_gen.py 生成产物回归测试
  urdf/  mjcf/        生成物（不入库，由 gen.py 重建）
mind/                 「心智」真机软件（yac，运行在 yiyiya OS）
  spec/               ★ 规范单一来源：obs / reward / action / randomize / ppo
  robotd/             50 Hz 控制循环（纯函数核心 + 宿主边界原语）
  tests/              yc 侧测试（robotd_core_smoke.yac）
  mindd/  updaterd/   高层意图、OTA（待实现）
train/                训练侧（Python）
  envs/spec_loader.py 受限子集解析 mind/spec/*.yac（不执行代码）
  envs/duck_env.py    环境骨架：build_obs / reward / StandEnv
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
未完成的：MuJoCo/MJX 物理后端（替换 `envs/duck_env.py` 里的 `NullPhysics`）、
PPO 训练循环、ONNX/blob 导出、真机 `robotd` 宿主运行时。

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
