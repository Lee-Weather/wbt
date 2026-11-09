"""
F1 gait-specific robot configurations.

Walking: 0.9× baseline KP/KD — softer tracking reduces overshoot for slower motions.
Running: 1.1× baseline KP/KD — stiffer tracking improves responsiveness for faster motions.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

F1_URDF_PATH = f"{ASSET_DIR}/f1/urdf/f1_29dof.urdf"
F1_ARMATURE = 0.01

# Default pose = frame 0 of B1_stand_to_walk (the reference standing pose)
F1_DEFAULT_JOINT_POS = {
    "lumbar_yaw_joint": -0.041,
    "lumbar_roll_joint": -0.056,
    "lumbar_pitch_joint": 0.020,
    "left_shoulder_pitch_joint": 0.107,
    "left_shoulder_roll_joint": -0.273,
    "left_shoulder_yaw_joint": -0.350,
    "left_elbow_pitch_joint": 0.563,
    "left_elbow_yaw_joint": 0.223,
    "left_wrist_pitch_joint": 0.307,
    "left_wrist_roll_joint": -0.010,
    "right_shoulder_pitch_joint": 0.076,
    "right_shoulder_roll_joint": -0.270,
    "right_shoulder_yaw_joint": 0.339,
    "right_elbow_pitch_joint": 0.670,
    "right_elbow_yaw_joint": 0.022,
    "right_wrist_pitch_joint": -0.178,
    "right_wrist_roll_joint": -0.010,
    "left_hip_pitch_joint": 0.034,
    "left_hip_roll_joint": -0.063,
    "left_hip_yaw_joint": -0.101,
    "left_knee_pitch_joint": 0.000,
    "left_ankle_pitch_joint": -0.060,
    "left_ankle_roll_joint": 0.009,
    "right_hip_pitch_joint": 0.011,
    "right_hip_roll_joint": 0.095,
    "right_hip_yaw_joint": 0.000,
    "right_knee_pitch_joint": 0.046,
    "right_ankle_pitch_joint": -0.071,
    "right_ankle_roll_joint": 0.009,
}


def _make_actuators(kp_scale: float, kd_scale: float) -> dict[str, ImplicitActuatorCfg]:
    """Build actuator dict with scaled gains from the baseline F1 config."""
    # Baseline values from f1.py (SYS-SPEC-0002-V1.0 locomotion tuning)
    baseline = {
        "hip_pitch": dict(kp=70.0, kd=5.8, effort=130.0, velocity=14.6608),
        "hip_roll": dict(kp=60.0, kd=4.5, effort=130.0, velocity=14.6608),
        "hip_yaw": dict(kp=35.0, kd=4.0, effort=130.0, velocity=14.6608),
        "knee": dict(kp=115.0, kd=11.5, effort=130.0, velocity=14.6608),
        "ankle_pitch": dict(kp=75.0, kd=1.25, effort=241.875, velocity=7.57244),
        "ankle_roll": dict(kp=50.0, kd=0.75, effort=218.4, velocity=8.05007),
        "lumbar": dict(kp=120.0, kd=8.0, effort=180.0, velocity=8.9),
        "arms": dict(kp=35.0, kd=2.0, effort=20.0, velocity=13.61),
        "wrists": dict(kp=20.0, kd=1.0, effort=10.0, velocity=2.5),
    }

    return {
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit_sim=baseline["hip_pitch"]["effort"],
            velocity_limit_sim=baseline["hip_pitch"]["velocity"],
            stiffness=baseline["hip_pitch"]["kp"] * kp_scale,
            damping=baseline["hip_pitch"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            effort_limit_sim=baseline["hip_roll"]["effort"],
            velocity_limit_sim=baseline["hip_roll"]["velocity"],
            stiffness=baseline["hip_roll"]["kp"] * kp_scale,
            damping=baseline["hip_roll"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            effort_limit_sim=baseline["hip_yaw"]["effort"],
            velocity_limit_sim=baseline["hip_yaw"]["velocity"],
            stiffness=baseline["hip_yaw"]["kp"] * kp_scale,
            damping=baseline["hip_yaw"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_pitch_joint"],
            effort_limit_sim=baseline["knee"]["effort"],
            velocity_limit_sim=baseline["knee"]["velocity"],
            stiffness=baseline["knee"]["kp"] * kp_scale,
            damping=baseline["knee"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=baseline["ankle_pitch"]["effort"],
            velocity_limit_sim=baseline["ankle_pitch"]["velocity"],
            stiffness=baseline["ankle_pitch"]["kp"] * kp_scale,
            damping=baseline["ankle_pitch"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim=baseline["ankle_roll"]["effort"],
            velocity_limit_sim=baseline["ankle_roll"]["velocity"],
            stiffness=baseline["ankle_roll"]["kp"] * kp_scale,
            damping=baseline["ankle_roll"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
        "lumbar": ImplicitActuatorCfg(
            joint_names_expr=["lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint"],
            effort_limit_sim=baseline["lumbar"]["effort"],
            velocity_limit_sim=baseline["lumbar"]["velocity"],
            # weighted average for effort limit across all lumbar joints
            stiffness=baseline["lumbar"]["kp"] * kp_scale,
            damping=baseline["lumbar"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim=baseline["arms"]["effort"],
            velocity_limit_sim=baseline["arms"]["velocity"],
            stiffness=baseline["arms"]["kp"] * kp_scale,
            damping=baseline["arms"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=baseline["wrists"]["effort"],
            velocity_limit_sim=baseline["wrists"]["velocity"],
            stiffness=baseline["wrists"]["kp"] * kp_scale,
            damping=baseline["wrists"]["kd"] * kd_scale,
            armature=F1_ARMATURE,
        ),
    }


def _make_cfg(actuators: dict[str, ImplicitActuatorCfg]) -> ArticulationCfg:
    """Build an ArticulationCfg with the given actuators."""
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=False,
            asset_path=F1_URDF_PATH,
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
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.61),
            joint_pos=F1_DEFAULT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=actuators,
    )


# --- Public singletons --------------------------------------------------------

# Walking: 0.9× baseline Kp/Kd — softer tracking reduces overshoot
F1_WALK_ACTUATORS = _make_actuators(kp_scale=0.9, kd_scale=0.9)
F1_CFG_WALK = _make_cfg(F1_WALK_ACTUATORS)
F1_ACTION_SCALE_WALK = 0.5

# Running: 1.1× baseline Kp/Kd — stiffer tracking for faster motions
F1_RUN_ACTUATORS = _make_actuators(kp_scale=1.1, kd_scale=1.1)
F1_CFG_RUN = _make_cfg(F1_RUN_ACTUATORS)
F1_ACTION_SCALE_RUN = 0.5
