"""Convert GMR/UMR qpos npz motions into the tracking npz format without Isaac Sim.

This script uses MuJoCo forward kinematics, so it can run on a plain Python environment.
The exported npz stores ``joint_names`` and ``body_names``; the tracking MotionLoader reorders
them to the Isaac articulation at runtime.

.. code-block:: bash

    python mujoco_to_npz.py \
        --input_dir source/whole_body_tracking/whole_body_tracking/assets/x1/motions/accad_120hz \
        --output_dir source/whole_body_tracking/whole_body_tracking/assets/x1/motions/converted
"""

import argparse
import glob
import os
import re

import mujoco
import numpy as np

TRACKING_FPS = 50.0


def quat_mul(a, b):
    """Hamilton product for wxyz quaternions, broadcasting over the last axis."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conj(q):
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_to_axis_angle(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm[..., 0], w)
    axis = np.where(norm > 1e-9, v / np.maximum(norm, 1e-9), 0.0)
    return axis * angle[..., None]


def angular_velocity(quats, dt):
    """World-frame angular velocity from a sequence of wxyz quaternions."""
    q_rel = quat_mul(quats[2:], quat_conj(quats[:-2]))
    omega = quat_to_axis_angle(q_rel) / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def slerp(q0, q1, t):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.maximum(sin_theta, 1e-6))
    w1 = np.where(small, t, np.sin(t * theta) / np.maximum(sin_theta, 1e-6))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample(qpos, fps_in, fps_out):
    times_in = np.arange(len(qpos)) / fps_in
    times_out = np.arange(0.0, times_in[-1], 1.0 / fps_out)
    phase = times_out * fps_in
    idx0 = np.floor(phase).astype(int)
    idx1 = np.minimum(idx0 + 1, len(qpos) - 1)
    blend = (phase - idx0)[:, None]

    pos = (1.0 - blend) * qpos[idx0, :3] + blend * qpos[idx1, :3]
    quat = slerp(qpos[idx0, 3:7], qpos[idx1, 3:7], blend)
    dof = (1.0 - blend) * qpos[idx0, 7:] + blend * qpos[idx1, 7:]
    return np.concatenate([pos, quat, dof], axis=1)


def convert(model, data, qpos, body_names, joint_names):
    t = len(qpos)
    dt = 1.0 / TRACKING_FPS
    body_pos = np.zeros((t, len(body_names), 3), dtype=np.float32)
    body_quat = np.zeros((t, len(body_names), 4), dtype=np.float32)

    for i in range(t):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        body_pos[i] = data.xpos[1:]
        body_quat[i] = data.xquat[1:]

    # Center the motion on the XY plane using the first frame.
    offset = body_pos[0, 0, :2].copy()
    body_pos[:, :, :2] -= offset
    qpos = qpos.copy()
    qpos[:, :2] -= offset

    joint_pos = qpos[:, 7:].astype(np.float32)
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang_vel = angular_velocity(body_quat.astype(np.float64), dt).astype(np.float32)

    return {
        "fps": np.array([TRACKING_FPS], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "joint_names": np.array(joint_names),
        "body_names": np.array(body_names),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert qpos npz motions to tracking npz via MuJoCo.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=float, default=50.0, help="Target tracking fps (fixed at 50 by default).")
    args = parser.parse_args()

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "source/whole_body_tracking/whole_body_tracking/assets/x1/mjcf/x1_29dof_scene.xml",
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]

    os.makedirs(args.output_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(args.input_dir, "*.npz"))):
        raw = np.load(path, allow_pickle=True)
        fps_in = float(raw["fps"][0])
        qpos = raw["qpos"].astype(np.float64)
        if fps_in != args.fps:
            qpos = resample(qpos, fps_in, args.fps)
        out = convert(model, data, qpos, body_names, joint_names)
        name = re.sub(r"\.npz$", "", os.path.basename(path))
        out_path = os.path.join(args.output_dir, name + ".npz")
        np.savez(out_path, **out)
        print(f"[INFO]: {os.path.basename(path)} ({fps_in:.0f} Hz) -> {out_path} ({len(qpos)} frames)")


if __name__ == "__main__":
    main()
