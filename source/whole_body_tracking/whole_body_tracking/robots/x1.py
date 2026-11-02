import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

##
# X1 (LimX Dynamics) 29-DoF humanoid.
#
# PD gains for the 6 leg joint types follow czy/config_gym/x1_dh_stand_config.py
# (hip_pitch 30/3, hip_roll 40/3, hip_yaw 35/4, knee_pitch 100/8, ankle_pitch 35/1.5,
# ankle_roll 35/1.5). Waist and arm gains are estimated from the URDF effort limits
# (waist 150-180 Nm -> 120/8, shoulder/elbow 20 Nm -> 35/2, wrist 10 Nm -> 20/1).
# Effort / velocity limits are taken directly from the URDF <limit> tags.
# Armature is a placeholder estimate (0.01) pending measured values.
##

X1_URDF_PATH = f"{ASSET_DIR}/x1/urdf/X1_29DOF_physically_mirrored.urdf"

# Default pose = frame 0 of F1_B1_stand_to_walk (the reference standing pose)
X1_DEFAULT_JOINT_POS = {
    "lumbar_yaw_joint": -0.018,
    "lumbar_roll_joint": -0.031,
    "lumbar_pitch_joint": 0.391,
    "left_shoulder_pitch_joint": -0.171,
    "left_shoulder_roll_joint": -0.129,
    "left_shoulder_yaw_joint": -0.223,
    "left_elbow_pitch_joint": 0.529,
    "left_elbow_yaw_joint": 0.000,
    "left_wrist_pitch_joint": 0.274,
    "left_wrist_roll_joint": 0.103,
    "right_shoulder_pitch_joint": -0.198,
    "right_shoulder_roll_joint": -0.240,
    "right_shoulder_yaw_joint": -0.310,
    "right_elbow_pitch_joint": 0.642,
    "right_elbow_yaw_joint": -0.073,
    "right_wrist_pitch_joint": -0.142,
    "right_wrist_roll_joint": -0.028,
    "left_hip_pitch_joint": -0.143,
    "left_hip_roll_joint": 0.016,
    "left_hip_yaw_joint": 0.089,
    "left_knee_pitch_joint": 0.087,
    "left_ankle_pitch_joint": -0.180,
    "left_ankle_roll_joint": -0.039,
    "right_hip_pitch_joint": 0.109,
    "right_hip_roll_joint": 0.011,
    "right_hip_yaw_joint": -0.142,
    "right_knee_pitch_joint": 0.154,
    "right_ankle_pitch_joint": -0.195,
    "right_ankle_roll_joint": 0.119,
}

X1_ARMATURE = 0.01

X1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=X1_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.60),
        joint_pos=X1_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit_sim=180.0,
            velocity_limit_sim=8.9,
            stiffness=30.0,
            damping=3.0,
            armature=X1_ARMATURE,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=27.23,
            stiffness=40.0,
            damping=3.0,
            armature=X1_ARMATURE,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=27.23,
            stiffness=35.0,
            damping=4.0,
            armature=X1_ARMATURE,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_pitch_joint"],
            effort_limit_sim=180.0,
            velocity_limit_sim=8.9,
            stiffness=100.0,
            damping=8.0,
            armature=X1_ARMATURE,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=13.61,
            stiffness=35.0,
            damping=1.5,
            armature=X1_ARMATURE,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=13.61,
            stiffness=35.0,
            damping=1.5,
            armature=X1_ARMATURE,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint"],
            effort_limit_sim=180.0,
            velocity_limit_sim=8.9,
            stiffness=120.0,
            damping=8.0,
            armature=X1_ARMATURE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=13.61,
            stiffness=35.0,
            damping=2.0,
            armature=X1_ARMATURE,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.5,
            stiffness=20.0,
            damping=1.0,
            armature=X1_ARMATURE,
        ),
    },
)

# Action scale follows the X1 locomotion stack (czy/config_gym), uniform 0.5 rad offset.
X1_ACTION_SCALE = 0.5
