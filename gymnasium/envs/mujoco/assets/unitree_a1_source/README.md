# Unitree A1 source description

The source URDF and meshes in this directory are from
[unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros),
revision `5994d4faef0a9cadd3287f8de0199a67eeb2a259`. They are distributed
under the BSD 3-Clause License included here.

`../unitree_a1.xml` is an MJCF model based on this description. It uses the
URDF's link inertias, joint locations and limits, and primitive collision
geometries. Its twelve motors are torque actuators with controls limited to
the URDF's 33.5 Nm effort limit. The MJCF uses simple collision geometry
rather than the source Collada visual meshes.

Its `standing` keyframe provides the nominal initial pose. Environments using
this model should initialize from that keyframe; loading the MJCF alone starts
from MuJoCo's zero configuration.
