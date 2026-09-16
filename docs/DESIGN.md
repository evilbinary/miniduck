# miniduck 设计文档

## 1. 项目概述

miniduck 是一台双足/双舵机驱动的小型仿生鸭子机器人。项目目标是：

1. 使用 3D 建模完成鸭子身体结构与关节布局设计；
2. 采用多关节舵机（XL330 系列总线舵机）驱动全身 14 个关节；
3. 采用 **yac + Python 双模式**：Python 生态承担 GPU 训练（MJX/Warp 并行仿真 + PPO），yac 承担规范定义（单一来源）、交叉验证与真机部署；
4. 训练得到的策略导出为 ONNX，最终用 yac 运行在 **yiyiya OS** 上，实现真机闭环行走。

核心思路：数学上是标准 RL——

- 策略 π(aₜ | oₜ)，观测 oₜ 来自仿真状态；
- 动作 aₜ 输出舵机目标（位置 / 增益 / Offset，具体看任务）；
- PPO 优化期望回报 Σ γᵗ rₜ；
- 仿真收集 → TD 误差 → 更新网络 → 域随机化（Domain Randomization）防止过拟合到仿真器。

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    yiyiya OS                             │
│  ┌───────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │  robotd   │  │  mindd   │  │ 策略推理 runtime(yac) │  │
│  │ 50Hz 控制环│←─│ (意图)   │←─│  策略推理 (obs→action) │  │
│  └─────┬─────┘  └──────────┘  └───────────────────────┘  │
│        │ SyncRead/SyncWrite (TTL 总线)                    │
│  ┌─────┴─────┐   ┌──────┐   ┌──────┐                     │
│  │ 14× XL330 │   │ IMU  │   │ ToF  │                     │
│  └───────────┘   └──────┘   └──────┘                     │
└─────────────────────────────────────────────────────────┘

训练侧（离线，PC / GPU，Python）：
┌─────────────────────────────────────────────────────────┐
│  spec（yac 单一来源）──翻译/加载──→ Python 环境           │
│  MuJoCo / MJX / Warp 并行仿真 (4096 envs)                 │
│  PPO 训练（PyTorch）→ ONNX 导出（归一化烘焙进图内）        │
└─────────────────────────────────────────────────────────┘

验证侧（yac，CPU）：
┌─────────────────────────────────────────────────────────┐
│  yac 迷你闭环验证：加载 ONNX 权重 → 手写 MLP 前向          │
│  → 与 Python 侧策略输出逐帧比对（交叉验证）                │
└─────────────────────────────────────────────────────────┘
```

### 2.1 双模式分工（yac + Python）

| 模块 | 职责 | 语言 | 运行位置 |
|------|------|------|----------|
| `spec/` | **单一来源规范**：obs 定义、reward 公式、action_scale、域随机化参数、关节映射 | yac | 训练 PC + 真机共用 |
| `train/` | PPO 训练主循环、MJX/Warp 并行仿真、超参调度 | Python | 训练 PC（GPU） |
| `export.py` | ONNX 导出（**中间格式**），并转换为部署端权重 blob | Python | 训练 PC |
| `verify/` | 迷你闭环验证：加载 ONNX 权重 → yac 手写 MLP 前向 → 与 Python 侧输出比对 | yac | 训练 PC（CPU） |
| `robotd` | 50 Hz 实时控制循环：读传感器 → 推理 → 安全层 → 下发 | yac（yiyiya OS） | 真机 |
| `mindd` | 高层意图（vx, vy, yaw_rate 命令）来源 | yac | 真机 |
| `updaterd` | OTA 打包 + 回滚槽位管理 | yac | 真机 |
| `cad/` | 鸭子身体 3D 建模文件与关节定义 | 建模工具 | 设计阶段 |

### 2.2 双模式的依据

训练侧性能瓶颈在于 GPU 并行仿真与自动微分（PPO 反向传播），这部分依赖
PyTorch/MJX 生态，自研语言无法在合理成本内补齐（yac 目前无张量库、无
autodiff、无 FFI 仿真接口）。因此：

- **Python 只出现在训练 PC 上**：吃 MJX/Warp 的 GPU 红利，1–2 小时出步态；
- **yac 承担所有"必须两端一致"的逻辑**（obs 构建、reward、动作映射）：
  规范用 yac 写成单一来源，训练侧翻译加载、真机侧直接执行，
  消除 sim2real 实现漂移——这是最常见的翻车点；
- **yac 纯函数式 + checkpoint/resume 特性**天然适合：
  `build_obs`/`servo_from_action`/安全层等纯逻辑，以及真机故障恢复；
- 推理本身不由 yac 计算：宿主运行时内置**自研 MLP 前向 runtime**（见 5.3，
  无外部依赖，单次 < 0.1 ms），ONNX 仅作为训练侧中间格式。

## 3. 硬件设计

### 3.0 两个尺寸版本

| | **miniduck-L（大型版）** | **miniduck-S（桌面版）** |
|------|------|------|
| 体格 | 与 microduck 相同量级 | 桌面小尺寸 |
| 目标身高 | 同 microduck | **12–18 cm** |
| 整机质量 | 同 microduck | **200–400 g** |
| 总线舵机 | 14 × XL330（或 XC330 提扭矩） | **14 × STS3032** |
| 关节分配 | 双腿 14 DOF（7×2，全预算给腿） | 腿 6 + 颈/头 3–4 + 喙 1 + 尾/翅/配重 3–4 |
| RL 策略 | 控制 14 腿关节，obs 61 / act 14 | 只控制 6 腿关节，obs 37 / act 6；表情关节走脚本 |
| 50 Hz 位置环 | 无压力 | 无压力（STS3032 内置位置环，总线 115200 起步） |

两版共用同一套软件栈（yac 规范 / robotd / 自研 runtime / 训练管线），
仅 `cad/robot.yaml` 与关节分配不同。

### 3.1 3D 建模与结构

**miniduck-L**：双腿各 7 个关节，共 **14 DOF** 全部用于行走：
每腿 Hip Yaw/Roll/Pitch + Knee Pitch + Ankle Pitch/Roll + 脚趾（被动/传感）；
重心低、靠近髋轴。

**miniduck-S** 关节分配（14 × STS3032）：

| 部位 | 数量 | 关节 | 说明 |
|------|------|------|------|
| 腿 ×2 | 6 | 髋 Pitch ×1 + 膝 Pitch ×1 + 踝 Pitch ×1（每腿 3） | 矢状面行走；桌面版牺牲髋侧摆/偏航，靠步宽与脚掌形状补偿 |
| 颈/头 | 3–4 | 颈 Pitch/Yaw、头 Tilt | 朝向交互、配合表情 |
| 喙 | 1 | 张合 | 交互反馈/鸣叫动作 |
| 尾巴/翅膀/配重 | 3–4 | 尾 Pitch、双翅扇动或髋部配重块 | 动态平衡辅助 + 表现力 |

结构要点：

- 躯干中空容纳主控板、IMU、电池，重心尽量靠下、靠近髋轴；
- S 版腿部只有矢状面自由度，**质心横向位置**必须靠结构保证
  （脚掌加宽、髋距拉大、翅/尾配重微调），这是 6-DOF 腿能站住的关键；
- 建模保证关节轴线正交、走线空间与过线孔；
- 材质：3D 打印（PLA/PETG），承重连接处碳板/金属加强；
- 脚底：刚性脚掌 + 可选触地传感（ToF / 电流估计接触）；
- 导出 MJCF/URDF 时必须与训练侧关节命名、顺序、限位完全一致
  （单一来源：`cad/robot.yaml` 生成两端描述文件，避免手改漂移）。

### 3.2 驱动与传感器

| 部件 | L 版选型 | S 版选型 | 说明 |
|------|----------|----------|------|
| 舵机 | 14 × XL330 / XC330 | **14 × STS3032** | TTL 总线串联，位置控制模式，同步读/写；STS3032：23.2×12.1×28.5 mm、~12 g、0.44 N·m，比 XL330 轻近一半 |
| IMU | 6/9 轴 | 6/9 轴 | 角速度、重力投影、欧拉角 |
| ToF | 左右脚 ×1（可选） | 左右脚 ×1（可选） | 脚触地估计；无则用电流/虚拟估计 |
| 主控 | yiyiya OS 兼容板 | yiyiya OS 兼容板 | 运行 robotd + 自研策略推理 runtime |

> 舵机协议差异（Dynamixel 协议 2.0 vs Feetech STS 协议）由宿主运行时的
> `io_dxl_sync_read/write` 边界原语适配，yac 侧控制核心零改动；
> S 版整机舵机总重 ~170 g，占 200–400 g 质量预算的一半，为电池与结构留出空间。

### 3.3 电气要点

- 舵机总线供电与逻辑电隔离，母线电容缓冲；
- 每关节可读：位置、速度、温度、负载（电流），供安全层使用；
- 电池容量按 50 Hz 全速控制 + 推理功耗估算，留 30% 余量；
- S 版 14 × STS3032 堵转电流按 1.5 A/个峰值预算，总线电源需 ≥ 8 A 峰值（桌面版实际运行电流远低于此）。

## 4. 强化学习环境（Python 训练 + yac 规范）

> obs 定义、reward 公式、动作映射、域随机化参数以 yac 规范（`spec/`）为单一来源；
> Python 训练环境从规范翻译/加载，保证与真机 `build_obs` 逐位一致。

### 4.1 观测向量（原始值，61 维示例）

| 段 | 维度 | 来源/说明 |
|----|------|-----------|
| base_ang_vel | 3 | IMU 陀螺仪，rad/s |
| projected_gravity | 3 | 重力向量在机体坐标系 |
| command | 3 | vx, vy, yaw_rate 目标 |
| joint_pos | 14 | XL330 当前位置 |
| joint_vel | 14 | 舵机速度/差分估计 |
| last_action | 14 | 上一帧策略输出 |
| contact | 2 | 左右脚触地估计（可选 ToF/电流/虚拟估计） |
| phase | 2 | cos(2πt/T), sin(2πt/T) |
| base_height | 1 | 躯干离地估计 |
| imu_euler | 3 | roll, pitch, yaw（用于状态对齐） |
| base_lin_vel | 2 | vx, vy 滤波/观测器输出 |
| **合计** | **61** | 3+3+3+14+14+14+2+2+1+3+2 |

> 若没有可靠线速度估计，该段填 0；ONNX 内的归一化层会自己处理，训练时同样置 0 + 域随机化。

**维度随版本参数化**（N = 策略控制的腿关节数）：

| 版本 | N | obs 维度 | 动作维度 |
|------|---|----------|----------|
| miniduck-L | 14（上表） | 3+3+3+14+14+14+2+2+1+3+2 = **61** | 14 |
| miniduck-S | 6（髋/膝/踝 ×2） | 3+3+3+6+6+6+2+2+1+3+2 = **37** | 6 |

> S 版的表情关节（颈/头、喙、尾/翅）**不进策略**：由表情/行为系统在
> 独立总线任务上以 ~20 Hz 驱动，与 50 Hz 控制环互不阻塞；翅膀/尾在
> RL 训练中建模为固定质量块（配重），不参与观测。

### 4.2 动作空间

- 维度：N（L 版 14，S 版 6），输出范围 [-1, 1]（策略原始输出）。
- 部署映射：

  ```
  target_pos = default_pos + action_scale * action
  ```

- 典型 `action_scale`：0.25 ~ 0.5，单位 rad。
- 控制频率：**50 Hz**。
- 舵机控制：位置控制模式，电流/速度限制由安全层约束。

### 4.3 奖励项与权重（示例组）

| 项 | 公式 | 权重 |
|----|------|------|
| 速度跟踪 | exp(-‖v_xy - v_cmd‖² / 0.25) | 1.5 |
| 偏航跟踪 | exp(-(ωz - ωz_cmd)² / 0.25) | 0.5 |
| 躯干高度 | exp(-(h - h_ref)² / 0.04) | 2.0 |
| 姿态稳定 | exp(-‖quat_xy‖² / 0.01) | 1.0 |
| 动作平滑 | -‖a_t - a_prev‖² | 0.01 |
| 关节力矩惩罚 | -‖τ‖² | 0.0005 |
| 存活奖励 | 1.0 | 1.0 |
| 自撞惩罚 | 若两腿相撞则 -1.0 | 1.0 |

> 权重是一组能跑出稳定步态的起点，仍需按真机行为调。

### 4.4 域随机化（关键项）

- 质量/质心随机（模拟电池、线缆扰动）；
- 关节位置/速度观测噪声；
- 舵机响应延迟（1~2 帧随机延迟）与控制增益抖动；
- 摩擦系数、地面起伏随机；
- 推送/踢腿扰动（随机外力脉冲）；
- 观测缺失：base_lin_vel 段训练时按概率置 0，与真机部署保持一致。

### 4.5 PPO 超参（典型）

| 参数 | 值 |
|------|-----|
| envs | 4096 |
| steps_per_env | 24 |
| mini_batch_size | 4096 |
| update_epochs | 5 |
| clip_range | 0.2 |
| gamma | 0.99 |
| lambda | 0.95 |
| entropy_coef | 0.01 |
| value_loss_coef | 0.5 |
| learning_rate | 0.0003 |
| schedule | cosine |
| max_grad_norm | 0.5 |

训练时长：单张 A100/4090 约 **1–2 小时**出可部署步态。若不开 MJX/Warp，用 CPU MuJoCo 则慢一个量级以上。

## 5. 策略导出与部署端推理（ONNX 中间格式 + 自研 runtime）

### 5.1 输入输出

```
输入：
  name: "obs"
  shape: [1, 61]        -- L 版；S 版为 [1, 37]
  dtype: float32
  含义: 4.1 节表格按顺序拼接的原始观测

输出：
  name: "actions"
  shape: [1, 14]        -- L 版；S 版为 [1, 6]
  dtype: float32
  含义: N 个腿关节 target position（已含 default_pos 偏移和 scale）
```

### 5.2 烘焙归一化

导出时把归一化、反归一化全部烤进图内：

```python
# 伪代码
norm_obs = (obs - mean) / std
action = policy(norm_obs)             # [-1, 1]
target_pos = default_pos + scale * action
onnx_export(model, args=(obs,), output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}})
```

因此真机只需提供**原始观测**，模型直接输出最终舵机目标角。

### 5.3 部署端：自研最小推理 runtime（不依赖 ONNX Runtime）

策略网络只是 61 → H → 14 的小 MLP（H=256 时约 3.3 万参数，float32 约
130 KB），前向就是两次矩阵-向量乘 + tanh，**没有任何理由引入一个通用推理
引擎**。部署端自研 runtime 全部计算核心：

```c
/* policy.c — 自研推理 runtime 的全部计算核心 */
typedef struct {
    uint32_t magic;                 /* "MDPK"（或 LLF 扩展 arch=MLP-POLICY） */
    uint32_t n_in, n_hid, n_out;    /* L: 61,256,14；S: 37,256,6 */
    /* 之后紧跟权重数组（mmap 直接读，无解析步骤）：
       float w1[n_hid*n_in]; float b1[n_hid];
       float w2[n_out*n_hid]; float b2[n_out];  */
} policy_blob;

void policy_run(const float obs[N_IN], float act[N_OUT]) {
    float h[N_HID];
    for (int i = 0; i < N_HID; i++) {
        float s = b1[i];
        for (int j = 0; j < N_IN; j++) s += w1[i * N_IN + j] * obs[j];
        h[i] = tanhf(s);
    }
    for (int i = 0; i < N_OUT; i++) {
        float s = b2[i];
        for (int j = 0; j < N_HID; j++) s += w2[i * N_HID + j] * h[j];
        act[i] = s;                 /* [-1,1]；归一化已烘焙，输入即原始 obs */
    }
}
```

要点：

- **计算量**：L 版 61×256 + 256×14 ≈ 1.9 万 MAC，S 版 37×256 + 256×6 ≈ 1.1 万 MAC，单核 < 0.1 ms，50 Hz 周期余量巨大；
- **零依赖**：无动态内存、无第三方库，看门狗/掉电恢复后立即可用，OTA 制品 ~130 KB；
- **等价性门禁**：`export.py` 同时产出 ONNX（给 `verify/` 交叉验证）与权重
  blob（给真机），两路输出逐帧比对，误差 < 1e-5（float32 求和顺序差异）；
- **与 yllm 生态的关系（可选）**：若希望统一模型格式，可按 LLF 扩展
  `arch=MLP-POLICY` 并自定义 slot 集，复用其 mmap/O(1) 寻址设施；但 MLP
  体量太小，裸 blob 已经足够，扩展 LLF 属于锦上添花。

## 6. 真机部署（yiyiya OS）

### 6.1 robotd 50 Hz 控制循环（yac 描述）

> **yac 是纯函数式语言**：没有命令式循环，"循环" = 尾递归（yac 真 TCO，
> 10⁷ 层不爆栈）；没有可变状态，`last_action` / 相位等全部作为参数显式传递；
> 没有内建 IO，所有副作用隔离在带 `io_` / `st_` 前缀的**宿主扩展原语**里
> （由 yiyiya OS 运行时提供）。纯函数核心（观测构建、安全层、动作映射）
> 不含任何副作用，可用 `yac --both` 做 ANF/CPS 双解释器一致性校验。
> 14 维向量统一用不可变 list 表示；步态相位用旋转递推，避免三角原语。

```yac
-- =====================================================================
-- robotd.yac — miniduck 50 Hz 控制循环
-- 分层：[边界原语 io_*/st_*（宿主提供）] ← 调用 ← [纯函数核心（本文件主体）]
-- =====================================================================

-- ---------------- 常量 ----------------
let joint_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] in
let cycle_ms  = 20 in                    -- 50 Hz
let action_scale = 0.35 in               -- rad，典型 0.25 ~ 0.5
let joint_lo  = -2.6 in                  -- 关节软限位（rad）
let joint_hi  =  2.6 in
-- safe_pose / default_pos：各 14 个角，装配标定后写入
-- （S 版：总线仍挂 14 个 STS3032 全量读；策略只下发 6 个腿目标，
--   颈/喙/尾/翅 8 个表情舵机由表情任务单独下发，二者在宿主层合流）
let safe_pose   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] in
let default_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] in
-- 相位旋转步长：步态周期 T 秒 → 每步转 θ = 2π / (50·T)，rc=cos θ, rs=sin θ
-- （启动时算好存为常量，运行期只需加减乘，不需要三角函数）
let rc = 0.998 in let rs = 0.0628 in     -- 示例：T = 1 s

-- ---------------- 纯函数工具 ----------------
let clamp(x, lo, hi) = if x < lo then lo else if x > hi then hi else x in
let max2(a, b) = if a > b then a else b in

-- 两/三列表同位映射（动作映射、安全层逐关节过滤用）
let map2(f, xs, ys) =
  if len(xs) == 0 then []
  else cons(f(nth(xs, 0), nth(ys, 0)),
            map2(f, tail(xs), tail(ys))) in

let map3(f, xs, ys, zs) =
  if len(xs) == 0 then []
  else cons(f(nth(xs, 0), nth(ys, 0), nth(zs, 0)),
            map3(f, tail(xs), tail(ys), tail(zs))) in

-- ---------------- 观测构建（4.1 节 61 维，与训练 spec 同一来源） ----------------
let build_obs(ang_vel, grav, cmd, jpos, jvel, last_action,
              contact, phase, base_h, euler, lin_vel) =
  append(append(append(append(ang_vel, grav), cmd),
                append(append(append(jpos, jvel), last_action), contact)),
         append(append(append(phase, cons(base_h, [])), euler), lin_vel)) in
-- 维度核对：3+3+3 + 14+14+14+2 + 2+1+3+2 = 61

-- ---------------- 安全层（纯函数） ----------------
let over60(x)   = x > 60.0 or x < -60.0 in
let is_fallen(e) = over60(nth(e, 0)) or over60(nth(e, 1)) in      -- roll/pitch
let is_stuck(js) = foldl(max2, 0.0, js) > 0.9 in                  -- 负载峰值

-- 逐关节过滤：过热(>70°C) → 保持当前位置；再做限位截断
let safe_joint(target, cur, temp) =
  let held = if temp > 70.0 then cur else target in
  clamp(held, joint_lo, joint_hi) in

-- ---------------- 动作 → 舵机目标（4.2 节映射） ----------------
let from_action(a, d) = d + action_scale * a in

-- ---------------- 主循环：尾递归，状态全部显式 ----------------
-- 状态：(t 周期计数, last_action 上一帧输出, pc/ps 相位 cos/sin)
let loop(t, last_action, pc, ps) =
  let cycle_start = io_now_ms() in

  -- 1. 读取总线状态（同步读；失败帧 ok=false）
  let frame  = io_dxl_sync_read(joint_ids) in
  let bus_ok = st_ok(frame) in
  let jpos   = st_pos(frame) in
  let jvel   = st_vel(frame) in
  let jtemp  = st_temp(frame) in
  let jload  = st_load(frame) in

  -- 2~4. 构建观测 → 策略推理（自研 runtime，见 5.3） → 安全层
  -- 任一异常（总线超时/跌倒/卡死/推理失败）→ 本帧动作 = safe_pose
  let action14 =
    if not bus_ok then
      let _ = io_log_warn("bus timeout") in safe_pose
    else if is_fallen(imu_euler_of(frame)) then
      let _ = io_log_warn("fall: cut motion, relax hips") in safe_pose
    else if is_stuck(jload) then
      let _ = io_log_warn("stuck: enter safe pose") in safe_pose
    else
      let imu     = io_imu_read() in
      let euler   = imu_euler(imu) in
      let cmd     = latest_cmd() in          -- 缓存为空 → 返回零速命令
      let contact = foot_contact_estimate() in    -- ToF 缓存/电流估计
      let phase   = cons(pc, cons(ps, [])) in     -- [cos, sin]
      let obs = build_obs(imu_ang_vel(imu), imu_grav(imu), cmd,
                          jpos, jvel, last_action,
                          contact, phase, base_height_estimate(),
                          euler, base_lin_vel_estimate()) in
      let raw = io_policy_run("obs", obs) in -- 推理失败返回 nil
      if raw == nil then
        let _ = io_log_warn("inference error") in safe_pose
      else
        -- 先映射到目标角（14 维），再逐关节安全过滤（过热保持 + 限位）
        let targets = map2(from_action, raw, default_pos) in
        map3(safe_joint, targets, jpos, jtemp)
  in

  -- 5. 下发目标（同步写；safe_pose 分支同样下发，保证回位）
  let _ = io_dxl_sync_write(joint_ids, action14) in

  -- 6. 状态发布
  let _ = io_publish(jpos, jvel, action14) in

  -- 7. 节奏控制（20 ms）：超期告警，睡到下一周期
  let elapsed = io_now_ms() - cycle_start in
  let _ = if elapsed > cycle_ms
          then io_log_warn("control loop late") else () in
  let _ = io_sleep_until(cycle_start + cycle_ms) in

  -- 相位旋转递推：pc' = pc·rc − ps·rs；ps' = pc·rs + ps·rc
  loop(t + 1, action14, pc * rc - ps * rs, pc * rs + ps * rc)
in
loop(0, safe_pose, 1.0, 0.0)    -- 初始相位 (cos, sin) = (1, 0)
```

> **实现注记**：真实 `robotd` 中，外层 forever 循环、看门狗与总线驱动位于
> yiyiya OS 宿主运行时；yac 描述的是其中**确定性的控制核心**。这样切分后：
> - 纯函数核心可离线用 ANF/CPS 双解释器校验，也可在 `verify/` 交叉验证中
>   直接复用同一份 `build_obs` / `safe_joint`；
> - 相位递推、限位截断等只依赖标量算术，完全落在 yac 现有原语能力内；
> - `io_policy_run` 内部调自研 MLP 前向 runtime（见 5.3，< 0.1 ms），yac 不自己算矩阵。

### 6.2 安全层要点

- **关节角度软/硬限位**：超限则截断并告警；
- **关节温度 > 70°C**：该关节进入 position 保持，禁止大动作；
- **负载电流持续超阈值**：判定卡死，立即回 safe pose；
- **IMU 姿态 > 60°**：判定跌倒，切断运动，松髋关节；
- **推理失败 / 总线超时**：下一帧必须回到 safe pose；
- **命令缓存为空**：使用零速度命令，而不是上次命令。

### 6.3 部署顺序

1. 仿真训策略 → ~2h（Python + MJX/Warp）
2. 导出 ONNX（中间格式）+ 权重 blob（部署端）→ `export.py`
3. 交叉验证 → `verify/`：自研 runtime 输出与 ONNX 逐帧比对（误差 < 1e-5）
4. 机架悬挂测试 → 用手托住，确认关节方向/限位
5. 低增益地面测试 → 能站则逐步提高增益
6. 闭环行走验证 → 按需改 reward / domain_random
7. OTA 打包 + 回滚槽位 → `updaterd`

## 7. 开发路线图

| 阶段 | 内容 | 产出 |
|------|------|------|
| M1 | 3D 建模 + 关节定义 + URDF/MJCF 导出 | `cad/`，可仿真模型 |
| M2 | yac 规范层（obs/reward/scale/随机化）+ Python 仿真环境加载规范 | `spec/`，`train/` 单机可跑 |
| M3 | PPO 训练 + ONNX 导出 | 可部署策略 `policy.onnx` |
| M3.5 | yac 交叉验证：迷你 MLP 前向比对 ONNX 输出 | `verify/`，逐帧一致 |
| M4 | 真机硬件装配 + robotd 50 Hz 循环（yac） | 悬挂测试通过 |
| M5 | 安全层完善 + 低增益站立/行走 | 地面步态验证 |
| M6 | OTA + 回滚 + 上线 | `updaterd` 发布流程 |

## 8. 风险与开放问题

- 舵机力矩有限（L: XL330 0.5 N·m；S: STS3032 0.44 N·m），粗调 reward 时容易抖动烧舵机 → 先低增益、短时测试；
- S 版腿部仅矢状面 3 DOF，横向稳定性靠结构（脚掌/髋距/配重）而非控制 → 先在悬挂上验证静态站立再上地面；
- 线速度估计不可靠时跟踪 reward 会失真 → 训练与部署都置 0 + 域随机化兜底；
- 仿真与真机关节方向/零位不一致是常见事故源 → 用 `cad/robot.yaml` 单一来源生成两端描述；
- 自研 runtime 与 ONNX 输出需逐帧比对（M3.5 门禁，误差 < 1e-5）；若 M4 实测延迟异常，再考虑 INT8 量化或裁剪隐层；
- yac 规范 → Python 训练环境的翻译环节可能引入偏差 → M3.5 的逐帧比对作为门禁，规范变更必须重跑验证。

## 9. 工程目录结构

```
miniduck/
├── docs/                        # 设计文档
│   └── DESIGN.md
│
├── body/                        # 「身体」—— 硬件与 3D 建模（L/S 两版）
│   ├── robot.yaml               # ★ 关节定义单一来源：DOF/限位/零位/舵机 ID/版本区分
│   ├── cad/                     # 3D 模型源文件（STEP / Fusion / Onshape 导出）
│   ├── urdf/                    # 由 robot.yaml 生成 → 训练仿真用（生成物，可重建）
│   ├── mjcf/                    # 由 robot.yaml 生成 → MuJoCo 用（生成物，可重建）
│   ├── electrical/              # 接线图、供电架构、总线拓扑
│   └── bom.md                   # 物料清单（STS3032/XL330、IMU、ToF、主控板）
│
├── mind/                        # 「心智」—— 真机软件（yac，运行在 yiyiya OS）
│   ├── spec/                    # ★ yac 单一来源规范：obs 定义/reward/action_scale/域随机化
│   ├── robotd/                  # 50 Hz 控制循环（robotd.yac + 宿主边界原语）
│   │   ├── robotd.yac           #   纯函数核心：build_obs / safe_joint / 动作映射
│   │   └── host/                #   宿主扩展原语：总线读写、IMU、时钟、watchdog（C）
│   ├── expression/              # 表情任务（S 版）：颈/喙/尾/翅 ~20 Hz，与控制环合流
│   ├── policy/                  # 部署端推理：policy_blob 权重 + 自研 MLP runtime（C）
│   └── tests/                   # ANF/CPS 双解释器一致性校验、安全层单元测试
│
├── train/                       # 训练侧（Python + GPU，只在训练 PC 上）
│   ├── envs/                    # MuJoCo/MJX/Warp 环境，从 spec/ 翻译加载规范
│   ├── ppo/                     # PPO 主循环、超参（L: obs61/act14；S: obs37/act6）
│   ├── export.py                # 双产物：ONNX（验证用）+ policy_blob（部署用）
│   └── scripts/                 # train.sh / eval.py / 回放可视化
│
├── verify/                      # ★ 交叉验证（M3.5 门禁）：自研 runtime vs ONNX 逐帧比对
│   ├── policy_forward.yac       # yac 手写 MLP 前向
│   └── golden/                  # 金标准帧数据（obs → 期望 action）
│
├── README.md
└── LICENSE
```

> ★ 为架构关键：`body/robot.yaml` 与 `mind/spec/` 是两处"单一来源"
> （关节定义、观测/奖励规范只写一份，两端各自生成/加载）；
> `verify/` 是发布门禁，独立于训练与真机两侧。
> `body/urdf/`、`body/mjcf/` 为生成物，不入库（见 `.gitignore`）。
