"""Convert a GMR pkl clip into the tracking npz format using MuJoCo forward kinematics.

This is the Isaac-free counterpart of `pkl_to_npz.py`: instead of replaying the motion
in Isaac Sim, it computes body poses with MuJoCo FK and derives velocities by central
differencing. Joint/body names are carried over, so `MotionLoader` reorders them to the
articulation order at load time.

.. code-block:: bash

    python scripts/pkl_to_npz_mujoco.py \
        --input_file source/whole_body_tracking/whole_body_tracking/assets/x1/motions/F1_B1_stand_to_walk.pkl \
        --output_file source/whole_body_tracking/whole_body_tracking/assets/x1/motions/converted/B1_stand_to_walk_x1_v13.npz
"""

import argparse
import os
import pickle

import mujoco
import numpy as np

DEFAULT_XML = (
    "source/whole_body_tracking/whole_body_tracking/assets/x1/mjcf/x1_29dof_scene.xml"
)


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_mul(a, b):
    """Hamilton product for wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotvec(q):
    """Shortest-arc rotation vector of a unit wxyz quaternion."""
    if q[0] < 0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-8:
        return np.zeros(3)
    return q[1:4] / s * angle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--no_center", action="store_true", help="Keep the original world XY offset.")
    args = ap.parse_args()

    with open(args.input_file, "rb") as f:
        d = pickle.load(f)

    fps = float(d["fps"])
    dt = 1.0 / fps
    dof_names = [str(n) for n in d["dof_names"]]
    body_names = [str(n) for n in d["body_names"]]
    T = len(d["root_pos"])

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in dof_names])
    body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names])

    body_pos = np.zeros((T, len(body_ids), 3))
    body_quat = np.zeros((T, len(body_ids), 4))
    for t in range(T):
        data.qpos[:7] = np.concatenate([d["root_pos"][t], quat_xyzw_to_wxyz(d["root_rot"][t])])
        data.qpos[qadr] = d["dof_pos"][t]
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[body_ids]
        body_quat[t] = data.xquat[body_ids]

    joint_pos = np.asarray(d["dof_pos"], dtype=np.float64)

    # central differences; one-sided at the ends
    def diff(arr):
        out = np.empty_like(arr)
        out[0] = (arr[1] - arr[0]) / dt
        out[-1] = (arr[-1] - arr[-2]) / dt
        out[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
        return out

    joint_vel = diff(joint_pos)
    body_lin_vel = diff(body_pos)

    body_ang_vel = np.zeros((T, len(body_ids), 3))
    for i in range(len(body_ids)):
        q = body_quat[:, i, :]
        body_ang_vel[0, i] = quat_to_rotvec(quat_mul(q[1], quat_conj(q[0]))) / dt
        body_ang_vel[-1, i] = quat_to_rotvec(quat_mul(q[-1], quat_conj(q[-2]))) / dt
        for t in range(1, T - 1):
            body_ang_vel[t, i] = quat_to_rotvec(quat_mul(q[t + 1], quat_conj(q[t - 1]))) / (2 * dt)

    if not args.no_center:
        offset = body_pos[0, 0, :2].copy()
        body_pos[:, :, 0] -= offset[0]
        body_pos[:, :, 1] -= offset[1]

    out = dict(
        fps=np.array([fps], dtype=np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        joint_names=np.array(dof_names),
        body_names=np.array(body_names),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    np.savez(args.output_file, **out)
    print(f"[INFO]: {T} frames @ {fps:.0f} fps -> {args.output_file}")
    print(f"[INFO]: joints={len(dof_names)} bodies={len(body_ids)} "
          f"|joint_vel|max={np.abs(joint_vel).max():.2f} rad/s "
          f"|body_ang_vel|max={np.abs(body_ang_vel).max():.2f} rad/s")


if __name__ == "__main__":
    main()
