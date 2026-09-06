<p align="right"><sub><a href="GETTING_STARTED.md">English</a> · <strong>中文</strong></sub></p>

# 快速上手：让 Go2 在 OrcaLab 中听懂导航指令

本实验不是“把模型跑起来”就结束。你要观察一条完整的机器人决策链：**看见什么、语言模型说了什么、四足机器人怎样执行、场景中发生了什么**。

方案 A 将 OrcaLab、NaVILA server 和导航进程分别放在终端 1、2、3。方案 B
把 OrcaLab 与导航终端留在客户端，将 NaVILA 服务移到远程推理服务器。方案 C
同样在参与者本机运行 OrcaLab 和导航，但通过 AWS SSM 端口转发连接主办方
托管的 NaVILA 服务。各层保持独立，便于定位问题。

## 部署方式：三选一

安装前只选择一种部署方式：

| 部署方式 | 结构 | 操作指南 |
| --- | --- | --- |
| **方案 A（默认）— 单机部署** | OrcaLab、NaVILA 与导航进程位于同一台机器 | 继续阅读下文的[方案 A 安装](#option-a-single-host) |
| **方案 B — 远程推理** | OrcaLab 与导航进程位于客户端，NaVILA 位于独立 GPU 服务器 | 按照[远程推理指南](REMOTE_INFERENCE_zh.md)操作 |
| **方案 C — 托管远程推理（AWS SSM）** | OrcaLab 与导航进程位于参与者本机，NaVILA 位于主办方托管的 AWS 实例 | 按照[托管访问指南](ACCESS_GUIDE_zh.md)操作 |

下文的安装与首次运行流程描述方案 A。方案 B 使用相同的场景与导航行为，
但两台机器的安装、服务启动、SSH 隧道和 NaVILA 协议端到端检查只在独立
远程指南中说明。方案 C 也使用相同的客户端场景与导航行为，其临时 SSO
凭据、AWS SSM 隧道、健康检查和客户端启动流程只在托管访问指南中说明。
这些连通性检查不执行模型推理。

## 一、实验目标与成功标准

默认任务保存在
[`prompts/orcalab_scene_locomotion.txt`](../prompts/orcalab_scene_locomotion.txt)：

> Walk toward the red waste bin and pass close by it without stopping. Continue toward the blue barrels and pass them. Then turn right and follow the open aisle beside the white safety fence toward the red fire extinguisher. Keep outside the fenced work cell and avoid the boxes. When the white industrial robotic arm mounted on a gray pedestal is visible, approach the open floor directly in front of the pedestal. Stop about 1.5 meters away from the arm.

成功不只等于终端没有报错。完成一次有效实验时，应同时满足：

- OrcaLab 中已打开 `VLN_Presentation` 场景，并有一个完整 Go2、红色垃圾桶、蓝色油桶、红色灭火器，以及灰色底座上的白色工业机械臂。
- `mujococamera1080` 的图像会随着 Go2 移动而改变。
- NaVILA server 收到 8 帧图像和任务文本，并返回一条可解析动作。
- Go2 动作平稳，结束后 `outputs/scene_locomotion_smoke/` 内有结果 JSON 与 RGB 帧。

## 二、先理解四个角色

| 角色 | 输入 | 输出 | 不负责什么 |
| --- | --- | --- | --- |
| NaVILA | 8 帧 RGB + 自然语言 | 文本动作 | 关节控制、碰撞求解 |
| 导航循环 | 文本动作 + 当前状态 | 速度命令和持续时间 | 生成视觉语言答案 |
| Go2 locomotion | 速度命令 | 12 关节动作 | 理解“蓝桶”或“左转”语义 |
| OrcaLab | Go2 位姿 | 场景 RGB 与可视化 | 训练或求解低层步态 |

例如，NaVILA 说 `turn left 15 degrees` 后，导航循环把它解析为固定角速度和 0.5 秒持续时间；Go2 策略在 50 Hz 下连续执行，OrcaLab 相机再采集新画面。这就是高层 VLM 与低层控制的分工。

<a id="option-a-single-host"></a>

## 三、方案 A：单机安装

先安装 [Miniconda 或 Anaconda](https://docs.anaconda.com/miniconda/install/)、Git，以及至少 RTX 4090 级别的 NVIDIA GPU 与驱动。必须先确认 `nvidia-smi` 成功，再克隆本仓库并按顺序执行：

```bash
git clone https://github.com/openverse-orca/Orca_VLN.git
cd Orca_VLN
```

安装 Python 3.12 OrcaLab、Python 3.10 NaVILA 和经过验证的模型：

```bash
./NaVILA-Orca/scripts/setup_all.sh
```

单独检查安装结果。只要出现 `FAIL` 就先停止，不要继续启动：

```bash
./NaVILA-Orca/scripts/doctor.sh
```

安装器会在 `Orca_VLN/.conda/envs/` 下创建彼此隔离的 `orcalab` 与
`navila` 前缀环境。它会锁定包版本、NaVILA 源码提交、Transformers
提交和 FlashAttention wheel 哈希，并对 OrcaLab 执行 `pip check`、
对两套环境执行导入验证。OrcaLab 官方原生 viewport 和场景 pak 也会在
安装阶段完成准备，不再推迟到第一次 GUI 进程中安装。

脚本根据自身位置解析环境。无需激活 Conda 环境，也无需导出仓库根目录
变量；即使当前终端激活了另一套环境，也不会选错 Python。

任意时候可单独验证或修复各层：

```bash
./NaVILA-Orca/scripts/setup_orcalab_env.sh --verify
./NaVILA-Orca/scripts/setup_navila_env.sh --verify
./NaVILA-Orca/scripts/doctor.sh
```

`setup_all.sh` 默认下载体积较大的模型。离线准备环境时可暂用
`--skip-model`，但启动服务前必须执行
`./NaVILA-Orca/scripts/download_navila_model.sh`。

## 四、方案 A：第一次运行

<a id="scene-setup"></a>

### 步骤 1：打开默认场景

终端 1：

```bash
./NaVILA-Orca/scripts/start_orcalab_gui.sh
```

GUI 中执行：

1. 在 OrcaLab 资产浏览器中订阅 `VLN_Presentation` 和 `unitree_robots`，等待两个
   订阅均显示为最新。
2. 选择 `VLN_Presentation` 场景。
3. 依次选择 **文件 → 打开布局 → `NaVILA-Orca/factory.json`**。
4. 在场景树中确认只有一个完整 Go2 actor。
5. 目视确认红色垃圾桶、蓝色油桶、红色灭火器和白色工业机械臂按任务所需方位可见。

`VLN_Presentation` 提供工厂场景本体，`factory.json` 保存叠加在其上的 actor
布局。该布局引用 `vln_presentation` 工厂资源和 `unitree_robots` 的 Go2；两个
订阅未完成时导入会出现缺失 actor。

启动脚本只打开 OrcaLab 的普通编辑器，不会强制选择地图、布局、全屏视图或
外部仿真。终端 3 的导航命令会在当前场景运行后应用并校验
`orca-train` profile。

### 步骤 2：启动 NaVILA

终端 2：

```bash
./NaVILA-Orca/scripts/start_navvlm_server.sh
```

当日志出现服务正在 `127.0.0.1:54321` 监听时，保持此终端运行。服务适配器
已包含在项目的 `scripts/navila_vlm_server.py` 中，用户不需要寻找或导出
额外脚本。模型缺失或下载不完整时，启动器会在加载前失败并给出恢复命令。

<a id="run-navigation"></a>

### 步骤 3：运行导航

运行终端 3 前，保持 OrcaLab GUI 打开，并依次选择：**运行 → 开始模拟 →
无仿真程序 → 启动**。等待仿真进入运行状态；终端 3 只连接这个已启动的
OrcaLab 会话，不会自行打开或启动仿真。

终端 3：

```bash
./NaVILA-Orca/scripts/run_orcalab_scene_locomotion.sh
```

该命令使用上面列出的默认 prompt。若要在本次运行中显式指定同一条指令，可使用：

```bash
./NaVILA-Orca/scripts/run_orcalab_scene_locomotion.sh \
  --instruction "Walk toward the red waste bin and pass close by it without stopping. Continue toward the blue barrels and pass them. Then turn right and follow the open aisle beside the white safety fence toward the red fire extinguisher. Keep outside the fenced work cell and avoid the boxes. When the white industrial robotic arm mounted on a gray pedestal is visible, approach the open floor directly in front of the pedestal. Stop about 1.5 meters away from the arm."
```

脚本的关键默认项：

| 参数 | 默认行为 | 教学含义 |
| --- | --- | --- |
| `--robot-actor-name auto` | 要求场景中恰有一台完整 Go2 | 避免控制到错误 actor |
| `--camera-asset-path prefabs/mujococamera1080` | 创建一次、持续采集 PNG | 看见的是机器人视角，不是 viewport |
| 默认相机安装位置 `0.1 0 1.0`，向下俯视 20 度 | 在站立场景中约为离地 1.4 米 | 为 VLM 提供更高且朝向桌面的视角 |
| `--warmup-steps 100` | 起步前零速度执行 100 个策略步 | 让策略状态稳定后再接收 VLM 命令 |
| `--scene-profile orca-train` | 200 Hz 物理、50 Hz 控制 | 动作距离可以按 tick 精确复现 |

## 五、读懂输出

结果目录为 `outputs/scene_locomotion_smoke/`。每次实验至少保存：

- RGB 帧：检查视角、图像是否随机器人移动而变化。
- 运行 JSON：记录输入 instruction、解析后的动作、时间和轨迹。
- scene alignment 文件：出现坐标或 actor 问题时用于核对 OrcaLab combined XML。

建议每组建立一张实验表：指令、首次模型动作、最终位置、是否经过红桶和蓝桶、是否沿安全围栏到达红色灭火器、是否在白色工业机械臂前停下、是否出现误转向、截图文件名。不要只记录“成功/失败”。

## 六、三项递进任务

### 任务 1：复现实验

保持所有默认参数不变，跑两次默认案例。比较两次的动作序列与最终轨迹，讨论模型推理是否完全确定，以及仿真初始化是否可重复。

### 任务 2：语言消融

```bash
./NaVILA-Orca/scripts/run_orcalab_scene_locomotion.sh \
  --instruction 'Pass the red bin, then turn right and stop at the blue barrel.'
```

再尝试“先向左转，再靠近蓝桶”。记录不同表达是否导致不同动作。注意：这不是测语言模型的常识题，而是观察语言、图像和几何关系是否共同影响决策。

### 任务 3：相机消融

将相机略微提高：

```bash
./NaVILA-Orca/scripts/run_orcalab_scene_locomotion.sh \
  --camera-mount-position 0.35 0 0.58
```

比较两组 RGB 帧和 NaVILA 动作。相机位置改变的不是物理控制器，而是 VLM 的观察；因此若结果变化，应该从视觉信息变化解释。

## 七、接入自定义 Go2 policy（进阶）

默认 checkpoint 已足够复现 VLN baseline。若要使用自行训练的 low-level policy，可以选择 [OrcaLocomotion](https://github.com/openverse-orca/OrcaLocomotion)、IsaacLab 或其他训练平台；训练平台不属于本项目的限制范围。

MJLab 在 Orca_VLN 中只负责运行当前 baseline 和输出对齐报告。自定义模型的接入重点是：Go2 关节顺序/符号、根部位姿、动作顺序、控制频率，以及 `vx / vy / wz / duration` 的速度命令接口。详细的直接加载与 adapter 路径见[低层运动控制](LOW_LEVEL_LOCOMOTION_zh.md)。高层 NaVILA 的 SFT/LoRA 路径见[VLN 微调](VLN_FINE_TUNING_zh.md)。

## 八、常见错误：先判断哪一层出了问题

| 现象 | 优先检查 | 常见原因 |
| --- | --- | --- |
| `Actor does not exist` | OrcaLab 场景树 | 未通过“文件 → 打开布局”载入 JSON、Go2 被删除或 actor 名不匹配 |
| `Failed to initialize NVML: Driver/library version mismatch` | 宿主 NVIDIA 驱动 | 系统更新了用户态驱动，但内核仍加载旧模块；保留 `.conda/`，重启电脑后依次运行 `nvidia-smi` 和 `setup_all.sh` |
| Qt 无法加载 `xcb` platform plugin | Ubuntu 系统库 | 重新运行 `setup_all.sh`，或单独执行 `setup_system_deps.sh` 安装 Qt/XCB 系统包 |
| `libOpenGL.so.0: undefined symbol: _glapi_tls_Current` | OrcaLab 的 OpenGL 前端与另一套 GLVND dispatcher 混用 | 拉取最新分支并重新运行 `setup_orcalab_env.sh`；项目会按 ELF 实际声明，将 `libGL.so.1` 或 `libOpenGL.so.0` 绑定到匹配的宿主库 |
| OrcaLab 首启安装 `orcalab-pyside` 并要求重启 | 使用了旧安装流程 | 拉取最新代码后重新运行 `setup_orcalab_env.sh`；Doctor 会检查原生 viewport、`patchelf` 及其环境专用 RPATH |
| `No module named 'deepspeed'` | NaVILA 环境 | 重新运行 `setup_navila_env.sh`；Doctor 现在会验证真实 model-builder import |
| 找到 0/多个 Go2 | 当前 scene | 没有完整 Go2 或重复导入了 setting |
| 相机属性缺失 | `orca-lab` 与 `orca-gym` 版本 | 未使用 26.7.1 或错误使用旧 `agentcamera` |
| VLM 无法连接 | 终端 2、端口 54321 | NaVILA server 未启动或端口不一致；方案 B 应按远程指南执行端到端检查；方案 C 应检查 SSM 隧道和托管服务健康状态 |
| 模型加载失败 | `NAVVLM_MODEL_PATH` | 指向了错误目录或 NaVILA 环境不完整 |
| Go2 抖动/跌倒 | checkpoint、warmup、场景初始位置 | checkpoint 不匹配、起点穿模、尚未稳定 |

排错顺序永远是：场景/actor → 相机 → NaVILA server → 动作文本 → Go2 策略。这样不会把一个连接错误误判为“模型不会导航”。
