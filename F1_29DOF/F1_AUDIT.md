# F1_29DOF 资产审计与 kp/kd 定稿

日期：2026-11-05（rev2：timestep 0.001 对齐 X1、qfrc_bias 重力矩、I+armature 对账、X1 基线门标定）｜ 来源：`F1_29DOF/urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf`（commit c0e0ef6）
工具：`tools_audit_f1.py`（审计）、`tools_convert_f1_meshes.py` + 扁平化导出（网格）、`tools_build_f1_mjcf.py`（资产构建）、`tools_f1_kp_kd.py`（增益推导）

## 1. 结构审计

- **29 个旋转关节，与 X1 完全同名同序**（无 F1-only / X1-only 关节）；23 个 fixed 装饰 link（腰电机、腕球头、颈部、相机、雷达）在仿真资产中剔除（30 body 树，与 Isaac PhysX merge-fixed 后一致）。
- **上身（lumbar + 双臂 17 关节）与 X1 逐字一致**：关节 origin/axis/limit/effort/velocity 全同，网格直接引用 `meshes/x1/`。
- **腿部 12 关节全新**：

| 关节 | F1 轴 | F1 限位 | F1 力矩 | X1 限位 | X1 力矩 |
|---|---|---|---|---|---|
| hip_pitch | y | ±1.571 | 130 | -1~2 | 180 |
| hip_roll | x | L:-0.349~1.571 R:-1.571~0.349 | 130 | L:-1.5~0.2 | 150 |
| hip_yaw | z | ±1.571 | 130 | ±1.5 | 150 |
| knee_pitch | y | 0~1.571 | 130 | 0~2 | 180 |
| ankle_pitch | y | ±0.524 | **241.9** | -0.41~0.35 | 80 |
| ankle_roll | x | ±0.524 | **218.4** | ±0.64 | 80 |

注意：F1 hip_roll 限位不对称（仅向外 ~20°，向内 ~90°）；URDF 轴为真实 x/y/z（X1 全用 z 轴+转角约定）。

## 2. 质量/几何

- 总质量 **35.03 kg**（上身 lumbar_yaw 子树 16.85 kg，腿+骨盆 18.19 kg）；惯量全部正定、三角不等式通过。
- 腿链：hip_pitch→hip_roll(0.085y)→hip_yaw(0.107)→knee(**0.139** m 短大腿)→ankle(**0.331** m 长小腿)；零位髋踝距离 0.573 m，髋间距 0.139 m。
- 足底（vendor collision STL AABB，ankle_roll 系）：长 0.204 m（x: -0.097~+0.107），宽 0.076 m，底面低于踝 0.034 m。
- 站立关键帧（数值求解，对称）：hip_pitch -0.292, knee 0.450, ankle_pitch -0.158, hip_roll≈0, lumbar≈0；base z **0.588 m**，CoM 过支撑中心（残差≈0）。
- 网格：visual STL 为 ASCII（最大 66 MB/25 万面）→ 转 binary 并抽稀（visual ≤40k 面，collision ≤20k 面）；三个自包含扁平 meshes 目录（`F1_29DOF/mjcf/meshes`、`whole_body_tracking/.../assets/f1/meshes`、`GMR/assets/f1/meshes`，collision 加 `c_` 前缀）。X1 验证过的布局（扁平文件名 + meshdir）。

## 3. 生成的仿真资产

| 文件 | 用途 |
|---|---|
| `F1_29DOF/mjcf/robot/f1/f1_serial.xml` | MuJoCo serial（timestep 0.001、29 motor、63 sensor 按 X1 顺序、8 足底角点接触球 + 腿/骨盆 collision mesh、keyframe stand） |
| `F1_29DOF/mjcf/f1_flat.xml` | 场景（含 floor） |
| `whole_body_tracking/.../assets/f1/mjcf/` | 训练/sim2sim 用副本（play_mujoco 自动探测 `assets/f1/mjcf/f1_flat.xml`） |
| `whole_body_tracking/.../assets/f1/urdf/f1.urdf` | Isaac URDF（30 link 树 + sim 网格） |
| `GMR/assets/f1/f1_mocap.xml` | GMR 重定向 IK 模型 |
| `F1_29DOF/mjcf/robot/f1/f1_stand.json` | 站立位姿（sensor 序 / URDF 序） |

渲染目检（render_f1_flat.png / render_f1_front.png）：装配正确、双腿连续、双脚贴地、双臂自然下垂。碰撞网格归 group 3（隐藏）避免与视觉网格混叠。

## 4. kp/kd 定稿（f1_kp_kd.json）

方法：上身沿用 X1 辨识方案 B（link 逐字一致，辨识惯性相同）；腿部按 F1 URDF 站立位子树惯量 I_stand，kp 取 X1 验证带宽带（髋 1.4–1.8 Hz、膝 3.2 Hz），kd 按 ζ 目标（0.4–0.7）反推、上限 8（沿用 X1 主机栈 MIT 编码约束）；armature 无台架辨识，按力矩等级 130/180 从 X1 辨识值缩放（未验证估计，如实标注）。

| 关节 | kp | kd | armature | I_stand | ζ | fn(Hz) |
|---|---|---|---|---|---|---|
| hip_pitch | 70 | 8.0 | 0.15 | 0.839 | 0.52 | 1.45 |
| hip_roll | 60 | 8.0 | 0.15 | 0.789 | 0.58 | 1.39 |
| hip_yaw | 90 | 2.0 | 0.02 | 0.047 | 0.41 | 6.96 |
| knee_pitch | 90 | 6.0 | 0.20 | 0.229 | 0.66 | 3.15 |
| ankle_pitch | 60 | 1.0 | 0.01 | 0.004 | 1.02 | 19.6 |
| ankle_roll | 45 | 0.6 | 0.01 | 0.0013 | 1.26 | 30.1 |
| lumbar_yaw/roll/pitch | 90/70/100 | 8.0 | 0.022/0.021/0.012 | — | 0.41–0.49 | ~1.1–1.8 |
| shoulder_p/r/y | 40 | 4/4/2 | 0.010 | — | 0.71–1.08 | ~2.3–6.9 |
| elbow_p/y | 30/20 | 1.9/1.0 | 0.010/0.005 | — | ~0.7–1.1 | ~3.5–6.8 |
| wrist_p/r | 15 | 1.0 | 0.005 | — | ~1.2 | ~5.9 |

**验证（门校准后）**：
- 焊接基座保持测试（3° 扰动，PD 保持站立目标，2 s @1 ms）→ 收敛到最大下垂 1.23°（hip_pitch），均值 0.29°，**PASS**（机械保形门）。
- 自由站立 PD-hold 门**经 X1 基线标定为无效**：X1 模型 + X1 sim2sim 验证过的方案 B 增益同样 3 s 内倒下（0.610→0.073 m）。自由站立是策略职责（线性化 mgh：X1≈183、F1≈216 N·m/rad ≫ 双踝 kp），"增益是 F1 倒塌根因"的假设被证伪。
- 站立位 qfrc_bias 重力矩（修正零位误差）：hip_pitch 2.23、knee 1.30 Nm → PD 静态下垂 ≤1.83°，与焊接基座实测一致。
- ζ/fn 按 I_eff = I_stand + armature 对账：ζ 0.41–0.55，fn 髋 1.3 / 膝 2.3 / 踝 ~10 Hz；kd·dt/I_eff < 0.8 在 1 ms（max 0.07）与 sim2sim 5 ms 物理（max 0.36）下均满足。
- 踝 kd 与 X1 差 ~5 倍的对账：X1 I_tot 0.15/0.035 含实验室辨识转子 armature；F1 裸惯量 0.004/0.0013 + 估计 armature 0.01 → I_eff ~10× 小，kd 相应小，两者 ζ 同带。armature 为未验证估计，sim2sim 时由 play_mujoco 注入（与 X1 配方一致，MJCF 本体只带厂商阻尼）。

## 5. 已知限制

- F1 无台架辨识：腿部 armature 为缩放估计；若后续拿到辨识数据，重训前应替换。
- 被动阻尼沿用 URDF `<dynamics>`（髋/膝 0.3，踝 0.1，上身 1.0），摩擦 0.5/0.2/0。
- hip_roll 不对称限位在重定向中须作为硬约束（mocap 模型 range 已按 URDF）。
