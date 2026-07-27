# BeyondMimic G1：训练、Sim2Sim 与 Sim2Real

本项目面向宇树 29 自由度 G1，将以下两条官方/上游链路整合为同一套接口：

- BeyondMimic `whole_body_tracking`：动作预处理、Isaac Lab + RSL-RL 训练、评估与 ONNX 导出；
- Unitree `unitree_rl_gym`：MuJoCo sim2sim 与 SDK2 真机部署流程。

默认任务为 `Tracking-Flat-G1-Wo-State-Estimation-v0`。它使用 154 维观测，
不需要根位置和根线速度估计，适合当前真机部署实现。

> 真机部署会发送底层关节指令。必须先完成 Isaac Lab 评估和 MuJoCo
> sim2sim，并在吊装、急停可用和现场有人监护的条件下测试。

## 项目结构

```text
training/whole_body_tracking/   BeyondMimic 训练源码（已支持本地 NPZ）
beyondmimic_g1/                 训练产物校验、共享观测、sim2sim、sim2real
configs/g1_beyondmimic.yaml     训练部署统一配置
unitree_description/            29-DoF G1 URDF、MJCF 与网格
deploy_real/                    Unitree SDK2 通信辅助代码及兼容入口
scripts/                        训练安装、训练、导出快捷脚本
tests/                          不依赖 Isaac/MuJoCo 的运行时单元测试
```

## 1. 环境

训练环境和部署环境建议分开创建。

### 训练环境

- Ubuntu 22.04
- NVIDIA GPU 与匹配的驱动
- Isaac Sim / Isaac Lab 2.1.0
- Python 3.10

在已经能运行 Isaac Lab 示例的环境中执行：

```bash
bash scripts/setup_training.sh
```

脚本以 editable 模式安装本仓库内的 `whole_body_tracking`。G1 资产默认直接
使用仓库根目录的 `unitree_description`；也可以通过
`BEYONDMIMIC_ASSET_DIR` 指向其他资产根目录。

### Sim2Sim 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim]"
```

### Sim2Real 环境

建议在连接 G1 的 Ubuntu 控制电脑上安装：

```bash
python -m pip install -r requirements-real.txt
```

另外必须按照 Unitree 官方说明安装 `unitree_sdk2py`。训练环境中的 Isaac
Lab、部署环境中的 SDK2 不要求安装在同一个 Python 环境。

## 2. 动作预处理

输入 CSV 使用 Unitree 重定向数据格式：

```text
root_position_xyz(3), root_quaternion_xyzw(4), G1_joint_positions(29)
```

转换为包含全部刚体状态、50 Hz 的 BeyondMimic NPZ：

```bash
python training/whole_body_tracking/scripts/csv_to_npz.py \
  --input_file data/my_motion.csv \
  --input_fps 30 \
  --output_name my_motion \
  --output_file motions/my_motion.npz \
  --output_fps 50 \
  --headless
```

默认只保存本地文件，不需要 WandB。若需要同步到 WandB Registry，可额外加
`--upload_to_wandb`。

在 Isaac Sim 中检查重定向结果：

```bash
python training/whole_body_tracking/scripts/replay_npz.py \
  --motion_file motions/my_motion.npz
```

## 3. G1 强化学习训练

本地 NPZ 训练：

```bash
bash scripts/train_g1.sh motions/my_motion.npz \
  --num_envs 4096 \
  --max_iterations 30000 \
  --run_name my_motion \
  --logger tensorboard
```

等价的完整命令为：

```bash
python training/whole_body_tracking/scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --motion_file motions/my_motion.npz \
  --headless
```

训练实现包括动作跟踪奖励、域随机化、随机推力、关节限位惩罚、异常接触惩罚
及 PPO 自适应采样。日志和 checkpoint 位于：

```text
logs/rsl_rl/g1_flat/<date>_<run_name>/
```

原版 WandB Registry 流程仍然可用，将 `--motion_file` 替换为
`--registry_name <registry-artifact>` 即可。

## 4. 评估和导出部署包

```bash
bash scripts/export_g1.sh motions/my_motion.npz \
  --load_run <训练运行目录名> \
  --checkpoint <checkpoint编号> \
  --headless
```

导出脚本使用同一个 Isaac Lab 环境恢复 checkpoint，生成：

```text
logs/rsl_rl/g1_flat/<run>/exported/
├── policy.onnx
└── motion.npz
```

ONNX 内包含关节名称、默认关节角、PD 参数、动作缩放、观测项和 anchor body
等元数据。sim2sim 与 sim2real 会优先读取这些元数据，不再依赖手工复制训练
参数。

## 5. 训练产物检查

先对 ONNX、动作、关节顺序、观测布局、控制频率和 G1 URDF 做一致性检查：

```bash
python -m beyondmimic_g1.validate \
  --config configs/g1_beyondmimic.yaml \
  --policy logs/rsl_rl/g1_flat/<run>/exported/policy.onnx \
  --motion logs/rsl_rl/g1_flat/<run>/exported/motion.npz
```

154 维观测顺序为：

```text
reference joint position  29
reference joint velocity  29
anchor orientation         6
base angular velocity      3
joint position            29
joint velocity            29
previous action           29
                         ---
                         154
```

## 6. MuJoCo Sim2Sim

先做无界面的短时冒烟测试：

```bash
python -m beyondmimic_g1.sim2sim \
  --config configs/g1_beyondmimic.yaml \
  --policy logs/rsl_rl/g1_flat/<run>/exported/policy.onnx \
  --motion logs/rsl_rl/g1_flat/<run>/exported/motion.npz \
  --headless \
  --steps 5000
```

再打开 MuJoCo Viewer 完整检查：

```bash
python -m beyondmimic_g1.sim2sim \
  --config configs/g1_beyondmimic.yaml \
  --policy <policy.onnx> \
  --motion <motion.npz>
```

兼容入口 `python deploy_mujoco_1.py ...` 仍然保留。

## 7. G1 Sim2Real

支持范围：

- 宇树 29 自由度 G1；
- `hg` DDS 消息；
- 骨盆 IMU；
- 50 Hz 策略；
- `Tracking-Flat-G1-Wo-State-Estimation-v0` 的 154 维 ONNX。

仓库自带的 `policy_zuiwu_48000.onnx` 包含倒地、头颈接近地面等高风险舞蹈
动作，仅用于检查数据和 sim2sim 链路，禁止直接作为首次真机测试策略。

运行前确认机器人处于吊装状态、零力矩/调试模式，网络和急停正常。命令必须
显式提供 `--enable-real` 才会发布电机指令：

```bash
python -m beyondmimic_g1.sim2real enp3s0 \
  --config configs/g1_beyondmimic.yaml \
  --policy <policy.onnx> \
  --motion <motion.npz> \
  --enable-real
```

遥控器流程：

1. `START`：从零力矩缓动到训练默认姿态；
2. `A`：启动动作策略；
3. `SELECT`：立即退出策略并进入阻尼模式。

真机运行时包含以下保护：

- ONNX/NPZ/关节顺序及维度启动校验；
- 动作裁剪和 URDF 关节限位；
- IMU 倾倒阈值；
- 关节速度阈值；
- DDS 状态超时；
- 连续控制周期超时；
- 动作结束、异常和 Ctrl+C 均通过 `finally` 进入阻尼模式。

所有阈值位于 `configs/g1_beyondmimic.yaml` 的 `safety` 节中。它们不是功能
安全认证，不能代替吊装、物理急停和现场风险评估。

## 8. 测试

纯 Python 运行时测试：

```bash
python -m unittest discover -s tests -v
```

完整验收顺序应为：

```text
动作回放 → Isaac Lab play → 导出校验 → MuJoCo headless
→ MuJoCo Viewer → 吊装真机 → 低难度落地动作
```

## 上游与许可证

- BeyondMimic 训练源码：
  <https://github.com/HybridRobotics/whole_body_tracking>
- Unitree RL Gym：
  <https://github.com/unitreerobotics/unitree_rl_gym>

固定的上游 commit 和许可证信息见 `THIRD_PARTY_NOTICES.md`。
