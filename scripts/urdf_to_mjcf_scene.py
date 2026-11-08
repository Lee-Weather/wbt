"""Compile a URDF into a floating-base MuJoCo scene that keeps the original meshes.

MuJoCo can load URDF directly, but the result is fixed-base and has no scene. This
script saves MuJoCo's own URDF import as MJCF, injects a free joint into the root
body, and wraps the robot in a floor/lights scene — the meshes stay exactly the
ones the URDF references (no simplified proxies).

.. code-block:: bash

    python scripts/urdf_to_mjcf_scene.py \
        --urdf source/whole_body_tracking/whole_body_tracking/assets/f1/urdf/f1_29dof.urdf \
        --output source/whole_body_tracking/whole_body_tracking/assets/f1/mjcf/f1_29dof_full_scene.xml
"""

import argparse
import os
import xml.etree.ElementTree as ET

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=os.path.abspath, required=True)
    parser.add_argument("--output", type=os.path.abspath, required=True)
    parser.add_argument("--timestep", type=float, default=0.005)
    args = parser.parse_args()

    # 1. Let MuJoCo convert the URDF (meshes keep the URDF scale/origin conventions).
    model = mujoco.MjModel.from_xml_path(args.urdf)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    raw_path = args.output + ".raw"
    mujoco.mj_saveLastXML(raw_path, model)

    # 2. Wrap into a scene: rename the model, add a free joint to the root body and a
    #    floor + lights. muJoCo writes mesh paths relative to the .raw file location, so
    #    the final file keeps the same directory and the same relative meshdir.
    tree = ET.parse(raw_path)
    root = tree.getroot()
    root.set("model", os.path.splitext(os.path.basename(args.output))[0])

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", str(args.timestep))

    worldbody = root.find("worldbody")
    robot_body = worldbody.find("body")
    if robot_body is None:
        raise ValueError("compiled MJCF has no root body")
    for joint in robot_body.findall("joint"):
        if joint.get("type") == "free":
            raise ValueError("root body already has a free joint")
    free = ET.Element("freejoint", {"name": "floating_base"})
    robot_body.insert(0, free)

    floor = ET.SubElement(
        worldbody,
        "geom",
        {"name": "floor", "type": "plane", "size": "0 0 20", "pos": "0 0 0", "condim": "3"},
    )
    floor.tail = "\n    "
    ET.SubElement(
        worldbody,
        "light",
        {"name": "light0", "diffuse": "0.9 0.9 0.9", "specular": "0.3 0.3 0.3", "pos": "0 -3 4", "dir": "0 0 -1"},
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "light1",
            "mode": "targetbodycom",
            "target": robot_body.get("name"),
            "pos": "-2 3 4",
            "diffuse": "0.7 0.7 0.7",
        },
    )
    visual = ET.Element("visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})
    root.insert(list(root).index(worldbody), visual)

    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    os.remove(raw_path)

    check = mujoco.MjModel.from_xml_path(args.output)
    print(
        f"[INFO]: {args.output} nq={check.nq} nv={check.nv} nbody={check.nbody} ngeom={check.ngeom}",
        flush=True,
    )


if __name__ == "__main__":
    main()