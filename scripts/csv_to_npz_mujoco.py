"""Convert a Unitree-style motion CSV into the tracking npz format using MuJoCo forward kinematics.

This is the Isaac-free counterpart of ``csv_to_npz.py``: the CSV rows are
``root XYZ + root quaternion XY ZW + joint angles`` (the LAFAN1 retargeting
dataset convention), and body poses/velocities are recovered with MuJoCo FK
instead of replaying the motion in Isaac Sim.

.. code-block:: bash

    python scripts/csv_to_npz_mujoco.py \
        --input_file lafan1_g1/dance2_subject5.csv \
        --output_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/dance2_subject5_22s.npz \
        --xml path/to/g1_scene.xml \
        --input_fps 30 --output_fps 50 --start_time 0.6 --duration 22.0
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

# Unitree G1 29-DoF joint order used by both the LAFAN1 CSV files and the MJCF model.
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# ----------------------------- quaternion helpers -----------------------------

def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    out = s0 * q0 + s1 * q1
    return out / np.linalg.norm(out)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation vector of a unit wxyz quaternion."""
    if q[0] < 0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-8:
        return np.zeros(3)
    return q[1:4] / s * angle


# ----------------------------- motion loading -----------------------------

def load_csv(path: str, input_fps: float, start_time: float, duration: float | None) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    start = int(round(start_time * input_fps))
    if duration is None:
        window = data[start:]
    else:
        window = data[start : start + int(round(duration * input_fps))]
    if len(window) < 2:
        raise ValueError(f"motion window is too short: {len(window)} frames")
    return window


def resample(window: np.ndarray, input_fps: float, output_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly/slerp interpolate the CSV rows to the output frame rate."""
    n = len(window)
    duration = (n - 1) / input_fps
    times = np.arange(0.0, duration, 1.0 / output_fps)
    phase = times / duration
    idx0 = np.floor(phase * (n - 1)).astype(int)
    idx1 = np.minimum(idx0 + 1, n - 1)
    blend = phase * (n - 1) - idx0

    root_pos = window[idx0, 0:3] * (1 - blend[:, None]) + window[idx1, 0:3] * blend[:, None]
    joints = window[idx0, 7:] * (1 - blend[:, None]) + window[idx1, 7:] * blend[:, None]

    q0 = np.array([quat_xyzw_to_wxyz(q) for q in window[idx0, 3:7]])
    q1 = np.array([quat_xyzw_to_wxyz(q) for q in window[idx1, 3:7]])
    root_quat = np.array([quat_slerp(a, b, t) for a, b, t in zip(q0, q1, blend)])
    return root_pos, root_quat, joints


def fk_body_states(
    model: mujoco.MjModel,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joints: np.ndarray,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = mujoco.MjData(model)
    nq = model.nq
    if nq != 7 + len(joint_names):
        raise ValueError(f"expected a 7 + {len(joint_names)} DoF MJCF, got nq={nq}")

    body_ids = [
        i
        for i in range(1, model.nbody)
        if not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i).endswith("_foot_contact_point")
    ]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in body_ids]

    # Map CSV joint columns to MJCF qpos addresses by name so the order cannot silently drift.
    qpos_adr = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} not found in MJCF")
        qpos_adr.append(int(model.jnt_qposadr[jid]))

    T = len(root_pos)
    body_pos = np.zeros((T, len(body_ids), 3))
    body_quat = np.zeros((T, len(body_ids), 4))
    for t in range(T):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_quat[t]
        data.qpos[qpos_adr] = joints[t]
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[body_ids]
        body_quat[t] = data.xquat[body_ids]
    return body_pos, body_quat, body_names


def central_diff(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(values)
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    if len(values) > 1:
        out[0] = (values[1] - values[0]) / dt
        out[-1] = (values[-1] - values[-2]) / dt
    return out


def angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    T = len(quats)
    out = np.zeros((T, 3))
    for t in range(1, T - 1):
        rel = quat_mul(quats[t + 1], quat_conj(quats[t - 1]))
        out[t] = quat_to_rotvec(rel) / (2.0 * dt)
    if T > 1:
        out[0] = quat_to_rotvec(quat_mul(quats[1], quat_conj(quats[0]))) / dt
        out[-1] = quat_to_rotvec(quat_mul(quats[-1], quat_conj(quats[-2]))) / dt
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="Unitree-style motion CSV.")
    parser.add_argument("--output_file", required=True, help="Output tracking npz path.")
    parser.add_argument("--xml", required=True, help="G1 MJCF scene XML used for FK.")
    parser.add_argument("--input_fps", type=float, default=30.0)
    parser.add_argument("--output_fps", type=float, default=50.0)
    parser.add_argument("--start_time", type=float, default=0.0, help="Window start in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Window duration in seconds.")
    parser.add_argument("--no_center", action="store_true", help="Keep the original world XY offset.")
    args = parser.parse_args()

    window = load_csv(args.input_file, args.input_fps, args.start_time, args.duration)
    root_pos, root_quat, joints = resample(window, args.input_fps, args.output_fps)
    dt = 1.0 / args.output_fps

    model = mujoco.MjModel.from_xml_path(args.xml)
    body_pos, body_quat, body_names = fk_body_states(model, root_pos, root_quat, joints, G1_JOINT_NAMES)

    if not args.no_center:
        offset = body_pos[0, 0, :2].copy()
        body_pos[:, :, 0] -= offset[0]
        body_pos[:, :, 1] -= offset[1]

    joint_vel = central_diff(joints, dt)
    body_lin_vel = central_diff(body_pos, dt)
    body_ang_vel = np.zeros_like(body_lin_vel)
    for i in range(len(body_names)):
        body_ang_vel[:, i] = angular_velocity(body_quat[:, i], dt)

    np.savez(
        args.output_file,
        fps=np.array([args.output_fps], dtype=np.float32),
        joint_pos=joints.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(body_names),
    )
    print(
        f"saved {args.output_file}: {len(joints)} frames @ {args.output_fps:.0f} Hz, "
        f"{len(G1_JOINT_NAMES)} joints, {len(body_names)} bodies"
    )


if __name__ == "__main__":
    main()