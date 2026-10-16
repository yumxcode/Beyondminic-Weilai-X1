# =============================================================================
# F1/X1 29DOF AMP 重训参数（定稿：方案 B，kd ≤ 8，不改电机 MIT 量程）
# 生成日期 2026-10-14；依据：实验室 12DOF 扭矩辨识 + 2026-10-13 两份真机行走日志估计
#
# 约束
#   - R86/R52 主机 MIT kd 编码上限 8.0，电机量程不可改 -> 所有下发 kd ≤ 8。
#   - 腰 LumbarParallelTransmission 为 1:1 直传（pitch->右电机，roll->左电机），关节 kd = 电机 kd。
#   - 髋/膝为 SimpleTransmission 1:1；踝为主机软件 PD（不经 MIT kd 字段，不受此限）。
#   - 日志拟合显示 kd 兑现约 0.9–1.0，说明固件 kd 量程与主机一致（8）。
#
# 设计原则
#   1) armature：实验室已辨识的用辨识值；髋 roll / 腰 / 右臂用日志估计；左臂 = 右臂；
#      肩 yaw、肘、腕 低于估计方法分辨率，维持 0.01。
#   2) kd 需求超过 8 的关节（髋 pitch/roll、膝、腰）：kd 取 8，降低 kp 使阻尼比达标：
#        kp = (kd / 2ζ)^2 / I_tot
#      目标 ζ：髋 pitch、膝 0.7；髋 roll 0.6；腰 0.4–0.5（带整个上半身，惯量 0.7–1.35）。
#   3) 其余关节 kp 不变；kd 按 ζ≈0.7 反推（≤ 4），已 ≥0.7 的不变。
#   4) Isaac 200 Hz 显式 PD 稳定性：kd·dt / I_min 全部 < 0.8。
#
#   关节            I_tot   kp/kd 旧      kp/kd 新     ζ 旧→新     fn 新
#   hip_pitch       0.49   120 / 4.0     65 / 8.0    0.26→0.71   1.8 Hz
#   hip_roll        0.74   100 / 3.3     60 / 8.0    0.19→0.60   1.4 Hz
#   hip_yaw         0.042  100 / 3.3    100 / 3.3    0.81        7.8 Hz
#   knee_pitch      0.39   150 / 5.0     85 / 8.0    0.33→0.70   2.4 Hz
#   ankle_pitch     0.15    50 / 2.5     50 / 4.0    0.45→0.73   2.9 Hz
#   ankle_roll      0.036   50 / 2.5     50 / 2.5    0.93        5.9 Hz
#   lumbar_yaw      0.73   120 / 4.0     90 / 8.0    0.21→0.49   1.8 Hz
#   lumbar_roll     1.35   120 / 4.0     70 / 8.0    0.16→0.41   1.1 Hz
#   lumbar_pitch    0.76   150 / 5.0    100 / 8.0    0.23→0.46   1.8 Hz
#   shoulder_pitch  0.19    40 / 2.0     40 / 4.0    0.36→0.72   2.3 Hz
#   shoulder_roll   0.20    40 / 2.0     40 / 4.0    0.36→0.71   2.3 Hz
#   shoulder_yaw    0.021   40 / 2.0     40 / 2.0    1.08        6.9 Hz
#   elbow_pitch     0.064   30 / 1.5     30 / 1.9    0.54→0.69   3.5 Hz
#   elbow_yaw       0.011   20 / 1.0     20 / 1.0    1.06        6.8 Hz
#   wrist_*         0.011   15 / 1.0     15 / 1.0    1.24        5.9 Hz
#   参照：main 12DOF 真机可用增益 膝 100/8、髋 pitch 30/3（真实惯量下 ζ 0.67 / 0.42）
# =============================================================================

# ---------- 1. 训练：roboparty_train/robolab/robolab/assets/robots/x1.py ----------
# 替换 X1_CFG.actuators；effort / velocity / delay 不变。

actuators = {
    "waist_legs": DelayedPDActuatorCfg(
        joint_names_expr=[
            "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
            ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_pitch_joint",
        ],
        effort_limit_sim=180.0,
        velocity_limit_sim=8.9,
        stiffness={
            "lumbar_yaw_joint": 90.0,     # 120 -> 90
            "lumbar_roll_joint": 70.0,    # 120 -> 70
            "lumbar_pitch_joint": 100.0,  # 150 -> 100
            ".*_hip_pitch_joint": 65.0,   # 120 -> 65
            ".*_hip_roll_joint": 60.0,    # 100 -> 60
            ".*_hip_yaw_joint": 100.0,    # 不变
            ".*_knee_pitch_joint": 85.0,  # 150 -> 85
        },
        damping={
            "lumbar_yaw_joint": 8.0,      # 4.0 -> 8.0
            "lumbar_roll_joint": 8.0,     # 4.0 -> 8.0
            "lumbar_pitch_joint": 8.0,    # 5.0 -> 8.0
            ".*_hip_pitch_joint": 8.0,    # 4.0 -> 8.0
            ".*_hip_roll_joint": 8.0,     # 3.3 -> 8.0
            ".*_hip_yaw_joint": 3.3,      # 不变
            ".*_knee_pitch_joint": 8.0,   # 5.0 -> 8.0
        },
        armature={
            "lumbar_yaw_joint": 0.022,    # 日志估计
            "lumbar_roll_joint": 0.021,   # 日志估计
            "lumbar_pitch_joint": 0.012,  # 日志估计
            ".*_hip_pitch_joint": 0.208,  # 实验室辨识
            ".*_hip_roll_joint": 0.25,    # 日志估计（建议有空补辨识）
            ".*_hip_yaw_joint": 0.0148,   # 实验室辨识
            ".*_knee_pitch_joint": 0.273, # 实验室辨识
        },
        min_delay=0,
        max_delay=2,
    ),
    "ankles": DelayedPDActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=80.0,
        velocity_limit_sim=13.61,
        stiffness=50.0,
        damping={
            ".*_ankle_pitch_joint": 4.0,  # 2.5 -> 4.0
            ".*_ankle_roll_joint": 2.5,   # 不变
        },
        armature={
            ".*_ankle_pitch_joint": 0.15,  # 实验室辨识
            ".*_ankle_roll_joint": 0.035,  # 实验室辨识
        },
        min_delay=0,
        max_delay=2,
    ),
    "arms": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
            ".*_elbow_pitch_joint", ".*_elbow_yaw_joint",
        ],
        effort_limit_sim=27.0,
        velocity_limit_sim=13.61,
        stiffness={
            ".*_shoulder_pitch_joint": 40.0,
            ".*_shoulder_roll_joint": 40.0,
            ".*_shoulder_yaw_joint": 40.0,
            ".*_elbow_pitch_joint": 30.0,
            ".*_elbow_yaw_joint": 20.0,
        },
        damping={
            ".*_shoulder_pitch_joint": 4.0,  # 2.0 -> 4.0
            ".*_shoulder_roll_joint": 4.0,   # 2.0 -> 4.0
            ".*_shoulder_yaw_joint": 2.0,    # 不变
            ".*_elbow_pitch_joint": 1.9,     # 1.5 -> 1.9
            ".*_elbow_yaw_joint": 1.0,       # 不变
        },
        armature={
            ".*_shoulder_pitch_joint": 0.02,   # 右臂估计，左臂镜像
            ".*_shoulder_roll_joint": 0.013,   # 右臂估计，左臂镜像
            ".*_shoulder_yaw_joint": 0.01,     # 维持
            ".*_elbow_pitch_joint": 0.01,      # 维持
            ".*_elbow_yaw_joint": 0.01,        # 维持
        },
        min_delay=0,
        max_delay=2,
    ),
    "wrists": DelayedPDActuatorCfg(
        joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint"],
        effort_limit_sim=10.0,
        velocity_limit_sim=2.5,
        stiffness=15.0,
        damping=1.0,
        armature=0.01,   # 维持；真机 L28 为内部位置环，不使用下发的 kp/kd
        min_delay=0,
        max_delay=2,
    ),
}

# ---------- 2. 训练随机化（amp_env_cfg.py EventCfg）----------
# scale_joint_parameters.armature_distribution_params: (0.8, 1.2) -> (0.8, 1.3)   # 乘在新 armature 上
# scale_actuator_gains: stiffness/damping (0.8, 1.2) 保持
#   注意：damping 随机化上沿 1.2 × 8 = 9.6 > 8，真机到不了；建议 damping 改为 (0.8, 1.0)，
#         或者训练名义 kd 取 7 再 ×(0.8,1.15)。stiffness 保持 (0.8, 1.2)。
# 动作延迟 / DelayedPD 0–2 步延迟 保持

# ---------- 3. sim2sim：sim2sim/mujoco_rollout.py 与 F1 scripts/amp_sim_bisect/rollout_bisect.py ----------
# - build_model() 里 "model.dof_armature[adr] = 0.01" 改为按上面的 armature 表逐关节设置；
# - PD 表 (PD dict) 的 kp/kd 同步改成上面的新值；
# 否则 sim2sim 仍是"轻腿 + 旧增益"，无法验证新策略。

# ---------- 4. 部署：motion_control/module/control_module/cfg/rl_x1.yaml -> rl_walk_amp ----------
# 关节顺序 = 策略顺序（Isaac Lab 交错顺序，见 rl_walk_amp.joint_list）。踝为主机软件 PD，同样读这里。
#
# stiffness: [65.0, 90.0, 65.0, 60.0, 70.0, 60.0,
#             100.0, 100.0, 100.0, 85.0, 40.0, 40.0,
#             85.0, 50.0, 40.0, 40.0, 50.0, 50.0,
#             40.0, 40.0, 50.0, 30.0, 30.0, 20.0,
#             20.0, 15.0, 15.0, 15.0, 15.0]
# damping:   [8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
#             3.3, 8.0, 3.3, 8.0, 4.0, 4.0,
#             8.0, 4.0, 4.0, 4.0, 4.0, 2.5,
#             2.0, 2.0, 2.5, 1.9, 1.9, 1.0,
#             1.0, 1.0, 1.0, 1.0, 1.0]
#
# 顺序对照：
#   left_hip_pitch, lumbar_yaw, right_hip_pitch, left_hip_roll, lumbar_roll, right_hip_roll,
#   left_hip_yaw, lumbar_pitch, right_hip_yaw, left_knee_pitch, left_shoulder_pitch, right_shoulder_pitch,
#   right_knee_pitch, left_ankle_pitch, left_shoulder_roll, right_shoulder_roll, right_ankle_pitch, left_ankle_roll,
#   left_shoulder_yaw, right_shoulder_yaw, right_ankle_roll, left_elbow_pitch, right_elbow_pitch, left_elbow_yaw,
#   right_elbow_yaw, left_wrist_pitch, right_wrist_pitch, left_wrist_roll, right_wrist_roll
#
# 站立 / 零位 / keep 控制器（pd_stand 等，kp 300 kd 5）不在本次范围内，保持不变。
