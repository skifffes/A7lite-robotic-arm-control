#!/usr/bin/env python3
"""
Read-only right A7lite full-hand trajectory probe v1.

Runs in the linker_a7 Conda environment.

Important corrections:
  - receives the official XRoboToolkit-transformed full 6DoF hand pose;
  - uses A7lite world_frame="urdf", not "maestro";
  - applies both hand translation and hand orientation relative to Grip press;
  - uses only the public A7lite IK API;
  - never enables the arm and never sends motor targets.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rotation

from linkerbot.arm.a7_lite.a7_lite import A7lite
from linkerbot.arm.common import Pose


def fmt_cm(values: np.ndarray) -> str:
    return (
        "["
        + ", ".join(f"{100.0 * value:+.2f}" for value in values)
        + "] cm"
    )


def fmt_deg(values: list[float]) -> str:
    return (
        "["
        + ", ".join(
            f"{math.degrees(value):+7.2f}"
            for value in values
        )
        + "]"
    )


def pose_to_components(pose: Any):
    position = np.array(
        [pose.x, pose.y, pose.z],
        dtype=float,
    )
    rotation = Rotation.from_euler(
        "zyx",
        [pose.rz, pose.ry, pose.rx],
        degrees=False,
    )
    return position, rotation


def components_to_pose(
    position: np.ndarray,
    rotation: Rotation,
) -> Pose:
    rz, ry, rx = rotation.as_euler(
        "zyx",
        degrees=False,
    )
    return Pose(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        rx=float(rx),
        ry=float(ry),
        rz=float(rz),
    )


def hand_pose_components(values: list[float]):
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (7,)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("Invalid transformed hand pose")
    quaternion_norm = float(np.linalg.norm(array[3:7]))
    if not (0.95 <= quaternion_norm <= 1.05):
        raise ValueError(
            f"Invalid hand quaternion norm {quaternion_norm}"
        )
    return (
        array[:3],
        Rotation.from_quat(array[3:7]),
        quaternion_norm,
    )


def clamp_translation(
    delta: np.ndarray,
    max_axis: float,
    max_radius: float,
) -> np.ndarray:
    bounded = np.clip(delta, -max_axis, max_axis)
    radius = float(np.linalg.norm(bounded))
    if radius > max_radius and radius > 1e-12:
        bounded = bounded * (max_radius / radius)
    return bounded


def clamp_rotation(
    delta: Rotation,
    max_angle: float,
    gain: float,
) -> Rotation:
    rotvec = delta.as_rotvec() * gain
    angle = float(np.linalg.norm(rotvec))
    if angle > max_angle and angle > 1e-12:
        rotvec = rotvec * (max_angle / angle)
    return Rotation.from_rotvec(rotvec)


def solve_closest(
    arm: A7lite,
    target_pose: Pose,
    seeds: list[list[float]],
    reference: list[float],
):
    candidates = []
    errors = []

    for seed_number, seed in enumerate(seeds, start=1):
        try:
            solution = list(
                arm.inverse_kinematics(
                    target_pose,
                    current_angles=seed,
                )
            )
            score = (
                max(
                    abs(a - b)
                    for a, b in zip(solution, reference)
                ),
                sum(
                    (a - b) ** 2
                    for a, b in zip(solution, reference)
                ),
            )
            candidates.append(
                (score, solution, seed_number)
            )
        except Exception as exc:
            errors.append(
                f"seed {seed_number}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not candidates:
        return None, 0, "; ".join(errors)

    candidates.sort(key=lambda item: item[0])
    _, solution, seed_number = candidates[0]
    return solution, seed_number, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="can2")
    parser.add_argument("--listen-port", type=int, default=48090)
    parser.add_argument("--source-port", type=int, default=48091)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rotation-gain", type=float, default=1.0)
    parser.add_argument("--max-axis-cm", type=float, default=5.0)
    parser.add_argument("--max-radius-cm", type=float, default=7.0)
    parser.add_argument("--max-rotation-deg", type=float, default=30.0)
    parser.add_argument("--report-hz", type=float, default=5.0)
    args = parser.parse_args()

    expected_source = ("127.0.0.1", args.source_port)
    max_axis = args.max_axis_cm / 100.0
    max_radius = args.max_radius_cm / 100.0
    max_rotation = math.radians(
        args.max_rotation_deg
    )

    udp_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )
    udp_socket.bind(("127.0.0.1", args.listen_port))
    udp_socket.settimeout(0.10)

    arm = None
    latest_sequence = None
    last_receive_time = time.monotonic()

    grip_was_active = False
    hand_position_anchor = None
    hand_rotation_anchor = None
    robot_position_anchor = None
    robot_rotation_anchor = None
    joint_anchor = None
    last_good_solution = None
    last_report = 0.0

    print("=" * 86)
    print("RIGHT A7lite full-hand trajectory probe v1 — READ ONLY")
    print(f"CAN:          {args.interface}")
    print("Robot frame:  A7lite native URDF frame")
    print("Hand frame:   official XRoboToolkit transformed frame")
    print("Control:      full 6DoF relative hand pose")
    print("No enable and no motor command.")
    print("=" * 86)

    try:
        arm = A7lite(
            side="right",
            interface_name=args.interface,
            world_frame="urdf",
        )

        actual_angles = list(arm.get_angles())
        actual_pose = arm.get_pose()
        print("[OK] Right A7lite connected read-only.")
        print("Current joints:", fmt_deg(actual_angles))
        print(
            "Current TCP:",
            [
                actual_pose.x,
                actual_pose.y,
                actual_pose.z,
            ],
        )
        print(
            "Press right Grip, then move and rotate "
            "the controller slowly.",
            flush=True,
        )

        while True:
            try:
                data, source = udp_socket.recvfrom(65535)
            except socket.timeout:
                if (
                    time.monotonic()
                    - last_receive_time
                    > 0.5
                ):
                    grip_was_active = False
                    hand_position_anchor = None
                    hand_rotation_anchor = None
                    print(
                        "[WATCHDOG] Waiting for hand packets...",
                        flush=True,
                    )
                    last_receive_time = time.monotonic()
                continue

            if source != expected_source:
                continue

            packet = json.loads(
                data.decode("utf-8")
            )
            if int(packet.get("version", 0)) != 5:
                continue

            sequence = int(packet["sequence"])
            if (
                latest_sequence is not None
                and sequence <= latest_sequence
            ):
                continue

            latest_sequence = sequence
            last_receive_time = time.monotonic()

            packet_age_ms = (
                time.monotonic_ns()
                - int(packet["host_monotonic_ns"])
            ) / 1e6

            (
                hand_position,
                hand_rotation,
                quaternion_norm,
            ) = hand_pose_components(
                packet["right_controller_pose_robot"]
            )

            grip_active = (
                float(packet.get("right_grip", 0.0))
                > 0.9
            )

            if grip_active and not grip_was_active:
                hand_position_anchor = (
                    hand_position.copy()
                )
                hand_rotation_anchor = hand_rotation

                joint_anchor = list(arm.get_angles())
                robot_pose_anchor = (
                    arm.forward_kinematics(joint_anchor)
                )
                (
                    robot_position_anchor,
                    robot_rotation_anchor,
                ) = pose_to_components(
                    robot_pose_anchor
                )
                last_good_solution = list(joint_anchor)

                print(
                    "\n[GRIP ON] Full 6DoF references captured.",
                    flush=True,
                )

            if not grip_active and grip_was_active:
                hand_position_anchor = None
                hand_rotation_anchor = None
                robot_position_anchor = None
                robot_rotation_anchor = None
                joint_anchor = None
                last_good_solution = None

                print(
                    "\n[GRIP OFF] Probe references cleared.",
                    flush=True,
                )

            grip_was_active = grip_active

            if (
                not grip_active
                or hand_position_anchor is None
                or hand_rotation_anchor is None
                or robot_position_anchor is None
                or robot_rotation_anchor is None
                or joint_anchor is None
                or last_good_solution is None
            ):
                continue

            hand_translation_delta = (
                hand_position
                - hand_position_anchor
            ) * args.scale

            hand_rotation_delta = (
                hand_rotation
                * hand_rotation_anchor.inv()
            )

            target_translation_delta = (
                clamp_translation(
                    hand_translation_delta,
                    max_axis,
                    max_radius,
                )
            )
            target_rotation_delta = clamp_rotation(
                hand_rotation_delta,
                max_rotation,
                args.rotation_gain,
            )

            target_position = (
                robot_position_anchor
                + target_translation_delta
            )
            target_rotation = (
                target_rotation_delta
                * robot_rotation_anchor
            )
            target_pose = components_to_pose(
                target_position,
                target_rotation,
            )

            actual_angles = list(arm.get_angles())
            solution, seed_number, error = solve_closest(
                arm,
                target_pose,
                [
                    last_good_solution,
                    actual_angles,
                    joint_anchor,
                ],
                last_good_solution,
            )

            now = time.monotonic()

            if solution is None:
                if now - last_report >= 0.5:
                    last_report = now
                    print(
                        f"[IK HOLD] age={packet_age_ms:.1f} ms "
                        f"{error}",
                        flush=True,
                    )
                continue

            last_good_solution = solution
            solved_pose = arm.forward_kinematics(
                solution
            )
            (
                solved_position,
                solved_rotation,
            ) = pose_to_components(solved_pose)

            position_error_mm = (
                float(
                    np.linalg.norm(
                        target_position
                        - solved_position
                    )
                )
                * 1000.0
            )
            rotation_error_deg = math.degrees(
                (
                    target_rotation
                    * solved_rotation.inv()
                ).magnitude()
            )

            if now - last_report >= (
                1.0 / args.report_hz
            ):
                last_report = now

                hand_rotation_deg = math.degrees(
                    hand_rotation_delta.magnitude()
                )

                print(
                    f"\n[HAND/IK] packet age="
                    f"{packet_age_ms:.1f} ms "
                    f"qnorm={quaternion_norm:.6f}\n"
                    f"  hand translation "
                    f"{fmt_cm(hand_translation_delta)}\n"
                    f"  hand rotation    "
                    f"{hand_rotation_deg:+.2f} deg\n"
                    f"  target TCP delta "
                    f"{fmt_cm(target_translation_delta)}\n"
                    f"  solved joints    "
                    f"{fmt_deg(solution)} "
                    f"(seed #{seed_number})\n"
                    f"  FK error         "
                    f"{position_error_mm:.2f} mm, "
                    f"{rotation_error_deg:.3f} deg",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nProbe stopped.")
        return 0

    finally:
        udp_socket.close()
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass
        print(
            "Connection closed. "
            "No hardware command was sent."
        )


if __name__ == "__main__":
    raise SystemExit(main())
