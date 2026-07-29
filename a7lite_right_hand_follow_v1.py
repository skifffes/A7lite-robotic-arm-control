#!/usr/bin/env python3
"""
Right A7lite full 6DoF hand-follow teleoperation v1.

Runs in the linker_a7 Conda environment.

Corrected data/control chain:
  official XRoboToolkit hand transform
  -> relative full 6DoF hand pose
  -> A7lite native URDF world frame
  -> public A7lite inverse_kinematics()
  -> complete IK joint target

No per-joint interpolation is used. Instead, the desired SE(3) pose is advanced
in small position/orientation steps, with adaptive backtracking when needed.

Run only after the read-only full-hand probe is stable.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial.transform import Slerp

from linkerbot.arm.a7_lite.a7_lite import A7lite
from linkerbot.arm.common import Pose


def norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values))


def fmt_cm(values: np.ndarray) -> str:
    return (
        "["
        + ", ".join(
            f"{100.0 * value:+.2f}"
            for value in values
        )
        + "] cm"
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

    quaternion_norm = float(
        np.linalg.norm(array[3:7])
    )
    if not (0.95 <= quaternion_norm <= 1.05):
        raise ValueError(
            "Invalid hand quaternion norm "
            f"{quaternion_norm}"
        )

    return (
        array[:3],
        Rotation.from_quat(array[3:7]),
    )


def clamp_translation(
    delta: np.ndarray,
    max_axis: float,
    max_radius: float,
) -> np.ndarray:
    bounded = np.clip(
        delta,
        -max_axis,
        max_axis,
    )
    radius = norm(bounded)
    if radius > max_radius and radius > 1e-12:
        bounded = bounded * (
            max_radius / radius
        )
    return bounded


def clamp_rotation(
    delta: Rotation,
    max_angle: float,
    gain: float,
) -> Rotation:
    rotvec = delta.as_rotvec() * gain
    angle = norm(rotvec)
    if angle > max_angle and angle > 1e-12:
        rotvec = rotvec * (
            max_angle / angle
        )
    return Rotation.from_rotvec(rotvec)


def step_position(
    current: np.ndarray,
    target: np.ndarray,
    max_step: float,
) -> np.ndarray:
    difference = target - current
    distance = norm(difference)
    if distance <= max_step or distance < 1e-12:
        return target.copy()
    return (
        current
        + difference * (max_step / distance)
    )


def interpolate_rotation(
    start: Rotation,
    end: Rotation,
    fraction: float,
) -> Rotation:
    if fraction >= 1.0:
        return end
    slerp = Slerp(
        [0.0, 1.0],
        Rotation.concatenate([start, end]),
    )
    return slerp([fraction])[0]


def step_rotation(
    current: Rotation,
    target: Rotation,
    max_step_angle: float,
) -> Rotation:
    difference = (
        target * current.inv()
    )
    angle = difference.magnitude()

    if angle <= max_step_angle or angle < 1e-12:
        return target

    fraction = max_step_angle / angle
    return (
        interpolate_rotation(
            Rotation.identity(),
            difference,
            fraction,
        )
        * current
    )


def solve_closest(
    arm: A7lite,
    target_pose: Pose,
    seeds: list[list[float]],
    reference: list[float],
):
    candidates = []
    errors = []

    for seed_number, seed in enumerate(
        seeds,
        start=1,
    ):
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
                    for a, b in zip(
                        solution,
                        reference,
                    )
                ),
                sum(
                    (a - b) ** 2
                    for a, b in zip(
                        solution,
                        reference,
                    )
                ),
            )
            candidates.append(
                (
                    score,
                    solution,
                    seed_number,
                )
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

    parser.add_argument("--max-axis-cm", type=float, default=3.0)
    parser.add_argument("--max-radius-cm", type=float, default=4.0)
    parser.add_argument("--max-rotation-deg", type=float, default=20.0)

    parser.add_argument("--control-hz", type=float, default=40.0)
    parser.add_argument("--tcp-speed-cm-s", type=float, default=3.0)
    parser.add_argument("--rotation-speed-deg-s", type=float, default=30.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=1.5)
    parser.add_argument("--joint-envelope-deg", type=float, default=35.0)
    parser.add_argument("--limit-margin-deg", type=float, default=3.0)

    parser.add_argument("--watchdog", type=float, default=0.30)
    parser.add_argument("--safe-hold-seconds", type=float, default=1.0)

    args = parser.parse_args()

    period = 1.0 / args.control_hz
    max_position_step = (
        args.tcp_speed_cm_s / 100.0
    ) * period
    max_rotation_step = math.radians(
        args.rotation_speed_deg_s
    ) * period
    max_joint_step = math.radians(
        args.max_joint_step_deg
    )

    max_axis = args.max_axis_cm / 100.0
    max_radius = args.max_radius_cm / 100.0
    max_rotation = math.radians(
        args.max_rotation_deg
    )
    joint_envelope = math.radians(
        args.joint_envelope_deg
    )
    limit_margin = math.radians(
        args.limit_margin_deg
    )

    max_hold_cycles = max(
        1,
        int(
            args.safe_hold_seconds
            * args.control_hz
        ),
    )

    expected_source = (
        "127.0.0.1",
        args.source_port,
    )

    stop_requested = False

    def request_stop(
        _signum: int,
        _frame: object,
    ) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(
        signal.SIGINT,
        request_stop,
    )
    signal.signal(
        signal.SIGTERM,
        request_stop,
    )

    udp_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )
    udp_socket.bind(
        ("127.0.0.1", args.listen_port)
    )
    udp_socket.settimeout(0.10)

    arm = None
    enabled = False
    old_velocities = None
    old_accelerations = None

    latest_packet = None
    last_sequence = None
    last_receive_time = time.monotonic()

    grip_was_active = False

    hand_position_anchor = None
    hand_rotation_anchor = None
    robot_position_anchor = None
    robot_rotation_anchor = None
    joint_anchor = None

    commanded_position_delta = np.zeros(3)
    commanded_rotation_delta = Rotation.identity()
    commanded_joints = None

    hold_cycles = 0
    tracking_bad_cycles = 0
    last_ready_report = 0.0
    last_live_report = 0.0

    print("=" * 88)
    print("RIGHT A7lite full 6DoF hand-follow v1")
    print("Robot frame: A7lite native URDF frame")
    print("Hand frame:  official XRoboToolkit transformed frame")
    print(
        f"Position envelope: ±{args.max_axis_cm:.1f} cm, "
        f"radius {args.max_radius_cm:.1f} cm"
    )
    print(
        f"Rotation envelope: ±{args.max_rotation_deg:.1f} deg"
    )
    print("No home and no calibration.")
    print("=" * 88)

    try:
        print(
            "Waiting for a valid packet "
            "with right Grip released...",
            flush=True,
        )

        while (
            latest_packet is None
            and not stop_requested
        ):
            try:
                data, source = (
                    udp_socket.recvfrom(65535)
                )
            except socket.timeout:
                continue

            if source != expected_source:
                continue

            packet = json.loads(
                data.decode("utf-8")
            )
            if int(packet.get("version", 0)) != 5:
                continue

            if (
                float(
                    packet.get(
                        "right_grip",
                        0.0,
                    )
                ) > 0.9
            ):
                print(
                    "[BLOCK] Release right Grip.",
                    flush=True,
                )
                continue

            latest_packet = packet
            last_sequence = int(
                packet["sequence"]
            )
            last_receive_time = time.monotonic()

        if stop_requested:
            return 130

        arm = A7lite(
            side="right",
            interface_name=args.interface,
            world_frame="urdf",
        )
        time.sleep(1.0)

        actual_angles = list(arm.get_angles())
        control_angles = list(
            arm.get_control_angles()
        )

        target_mismatch = max(
            abs(actual - control)
            for actual, control in zip(
                actual_angles,
                control_angles,
            )
        )

        print("[OK] Right A7lite connected.")
        print(
            "Target mismatch: "
            f"{math.degrees(target_mismatch):.3f} deg"
        )

        if target_mismatch > math.radians(0.5):
            print(
                "[BLOCK] Existing target mismatch "
                "exceeds 0.5 deg."
            )
            return 2

        joint_limits = list(
            arm.get_joint_limits()
        )

        for index, (
            angle,
            (lower, upper),
        ) in enumerate(
            zip(
                actual_angles,
                joint_limits,
            ),
            start=1,
        ):
            if not (
                lower + limit_margin
                <= angle
                <= upper - limit_margin
            ):
                print(
                    f"[BLOCK] J{index} is too close "
                    "to its mechanical limit."
                )
                return 2

        print(
            "\nClear the right-arm workspace and "
            "keep emergency stop/power reachable."
        )

        if input("Type OK to start: ").strip() != "OK":
            print("[CANCELLED]")
            return 3

        old_velocities = list(
            arm.get_control_velocities()
        )
        old_accelerations = list(
            arm.get_control_acceleration()
        )

        arm.set_velocities([0.18] * 7)
        arm.set_accelerations([1.0] * 7)
        arm.enable()
        enabled = True

        commanded_joints = list(
            arm.get_angles()
        )

        print(
            "[ENABLED] Press right Grip to anchor "
            "full hand following.",
            flush=True,
        )

        udp_socket.setblocking(False)
        next_tick = time.monotonic()

        while not stop_requested:
            now = time.monotonic()

            for _ in range(256):
                try:
                    data, source = (
                        udp_socket.recvfrom(65535)
                    )
                except BlockingIOError:
                    break

                if source != expected_source:
                    continue

                packet = json.loads(
                    data.decode("utf-8")
                )
                if int(packet.get("version", 0)) != 5:
                    continue

                sequence = int(
                    packet["sequence"]
                )
                if (
                    last_sequence is not None
                    and sequence <= last_sequence
                ):
                    continue

                latest_packet = packet
                last_sequence = sequence
                last_receive_time = (
                    time.monotonic()
                )

            now = time.monotonic()

            if (
                now - last_receive_time
                > args.watchdog
            ):
                raise TimeoutError(
                    "Hand packet stale for "
                    f"{now - last_receive_time:.3f} s"
                )

            if now < next_tick:
                time.sleep(
                    min(
                        0.001,
                        next_tick - now,
                    )
                )
                continue

            next_tick = now + period

            (
                hand_position,
                hand_rotation,
            ) = hand_pose_components(
                latest_packet[
                    "right_controller_pose_robot"
                ]
            )

            grip_active = (
                float(
                    latest_packet.get(
                        "right_grip",
                        0.0,
                    )
                ) > 0.9
            )

            if (
                grip_active
                and not grip_was_active
            ):
                hand_position_anchor = (
                    hand_position.copy()
                )
                hand_rotation_anchor = (
                    hand_rotation
                )

                joint_anchor = list(
                    arm.get_angles()
                )
                robot_pose_anchor = (
                    arm.forward_kinematics(
                        joint_anchor
                    )
                )
                (
                    robot_position_anchor,
                    robot_rotation_anchor,
                ) = pose_to_components(
                    robot_pose_anchor
                )

                commanded_position_delta = (
                    np.zeros(3)
                )
                commanded_rotation_delta = (
                    Rotation.identity()
                )
                commanded_joints = list(
                    joint_anchor
                )

                hold_cycles = 0
                tracking_bad_cycles = 0

                print(
                    "\n[GRIP ON] Full 6DoF "
                    "references captured.",
                    flush=True,
                )

            if (
                not grip_active
                and grip_was_active
            ):
                hold_angles = list(
                    arm.get_angles()
                )
                arm._set_angles(
                    hold_angles
                )
                commanded_joints = (
                    hold_angles
                )

                hand_position_anchor = None
                hand_rotation_anchor = None
                robot_position_anchor = None
                robot_rotation_anchor = None
                joint_anchor = None

                commanded_position_delta = (
                    np.zeros(3)
                )
                commanded_rotation_delta = (
                    Rotation.identity()
                )

                hold_cycles = 0
                tracking_bad_cycles = 0

                print(
                    "\n[GRIP OFF] Holding "
                    "release position.",
                    flush=True,
                )

            grip_was_active = grip_active

            if not grip_active:
                if (
                    now - last_ready_report
                    >= 1.0
                ):
                    last_ready_report = now
                    print(
                        f"[READY] seq={last_sequence}; "
                        "waiting for right Grip.",
                        flush=True,
                    )
                continue

            if (
                hand_position_anchor is None
                or hand_rotation_anchor is None
                or robot_position_anchor is None
                or robot_rotation_anchor is None
                or joint_anchor is None
                or commanded_joints is None
            ):
                continue

            desired_position_delta = (
                hand_position
                - hand_position_anchor
            ) * args.scale

            desired_position_delta = (
                clamp_translation(
                    desired_position_delta,
                    max_axis,
                    max_radius,
                )
            )

            desired_rotation_delta = (
                hand_rotation
                * hand_rotation_anchor.inv()
            )
            desired_rotation_delta = (
                clamp_rotation(
                    desired_rotation_delta,
                    max_rotation,
                    args.rotation_gain,
                )
            )

            requested_position_delta = (
                step_position(
                    commanded_position_delta,
                    desired_position_delta,
                    max_position_step,
                )
            )

            requested_rotation_delta = (
                step_rotation(
                    commanded_rotation_delta,
                    desired_rotation_delta,
                    max_rotation_step,
                )
            )

            actual_angles = list(
                arm.get_angles()
            )

            accepted = None
            rejection_reason = "no candidate"

            for fraction in (
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
            ):
                candidate_position_delta = (
                    commanded_position_delta
                    + fraction
                    * (
                        requested_position_delta
                        - commanded_position_delta
                    )
                )

                candidate_rotation_delta = (
                    interpolate_rotation(
                        commanded_rotation_delta,
                        requested_rotation_delta,
                        fraction,
                    )
                )

                target_position = (
                    robot_position_anchor
                    + candidate_position_delta
                )
                target_rotation = (
                    candidate_rotation_delta
                    * robot_rotation_anchor
                )
                target_pose = (
                    components_to_pose(
                        target_position,
                        target_rotation,
                    )
                )

                solution, seed_number, error = (
                    solve_closest(
                        arm,
                        target_pose,
                        [
                            commanded_joints,
                            actual_angles,
                            joint_anchor,
                        ],
                        commanded_joints,
                    )
                )

                if solution is None:
                    rejection_reason = (
                        f"IK failed: {error}"
                    )
                    continue

                joint_step = max(
                    abs(new - previous)
                    for new, previous in zip(
                        solution,
                        commanded_joints,
                    )
                )

                if joint_step > max_joint_step:
                    rejection_reason = (
                        "joint step "
                        f"{math.degrees(joint_step):.2f} deg"
                    )
                    continue

                valid = True
                for index, (
                    value,
                    anchor,
                    (lower, upper),
                ) in enumerate(
                    zip(
                        solution,
                        joint_anchor,
                        joint_limits,
                    ),
                    start=1,
                ):
                    if (
                        abs(value - anchor)
                        > joint_envelope
                    ):
                        rejection_reason = (
                            f"J{index} anchor envelope"
                        )
                        valid = False
                        break

                    if not (
                        lower + limit_margin
                        <= value
                        <= upper - limit_margin
                    ):
                        rejection_reason = (
                            f"J{index} limit margin"
                        )
                        valid = False
                        break

                if not valid:
                    continue

                solved_pose = (
                    arm.forward_kinematics(
                        solution
                    )
                )
                (
                    solved_position,
                    solved_rotation,
                ) = pose_to_components(
                    solved_pose
                )

                position_error = norm(
                    target_position
                    - solved_position
                )
                rotation_error = (
                    target_rotation
                    * solved_rotation.inv()
                ).magnitude()

                if position_error > 0.002:
                    rejection_reason = (
                        "FK position error "
                        f"{1000.0 * position_error:.2f} mm"
                    )
                    continue

                if rotation_error > math.radians(0.5):
                    rejection_reason = (
                        "FK rotation error "
                        f"{math.degrees(rotation_error):.2f} deg"
                    )
                    continue

                accepted = (
                    candidate_position_delta,
                    candidate_rotation_delta,
                    solution,
                    target_position,
                    target_rotation,
                    seed_number,
                    fraction,
                    joint_step,
                )
                break

            if accepted is None:
                hold_cycles += 1

                if (
                    hold_cycles == 1
                    or hold_cycles % 10 == 0
                ):
                    print(
                        f"[HOLD] {hold_cycles}/"
                        f"{max_hold_cycles}: "
                        f"{rejection_reason}",
                        flush=True,
                    )

                if hold_cycles >= max_hold_cycles:
                    raise RuntimeError(
                        "No safe full-pose IK step "
                        "for the hold interval"
                    )
                continue

            hold_cycles = 0

            (
                commanded_position_delta,
                commanded_rotation_delta,
                commanded_joints,
                target_position,
                target_rotation,
                seed_number,
                fraction,
                joint_step,
            ) = accepted

            arm._set_angles(
                commanded_joints
            )

            max_tracking_error = max(
                abs(command - actual)
                for command, actual in zip(
                    commanded_joints,
                    actual_angles,
                )
            )

            if (
                max_tracking_error
                > math.radians(5.0)
            ):
                tracking_bad_cycles += 1
            else:
                tracking_bad_cycles = 0

            if tracking_bad_cycles >= int(
                0.5 * args.control_hz
            ):
                raise RuntimeError(
                    "Joint tracking error exceeded "
                    "5 deg for 0.5 s"
                )

            actual_pose = (
                arm.forward_kinematics(
                    actual_angles
                )
            )
            (
                actual_position,
                actual_rotation,
            ) = pose_to_components(
                actual_pose
            )

            actual_position_delta = (
                actual_position
                - robot_position_anchor
            )
            position_tracking_error = (
                target_position
                - actual_position
            )
            rotation_tracking_error = (
                target_rotation
                * actual_rotation.inv()
            ).magnitude()

            if (
                now - last_live_report
                >= 0.25
            ):
                last_live_report = now

                print(
                    f"[FOLLOW] hand xyz "
                    f"{fmt_cm(desired_position_delta)} "
                    f"rot="
                    f"{math.degrees(desired_rotation_delta.magnitude()):.1f}°\n"
                    f"         target   "
                    f"{fmt_cm(commanded_position_delta)} "
                    f"rot="
                    f"{math.degrees(commanded_rotation_delta.magnitude()):.1f}°\n"
                    f"         actual   "
                    f"{fmt_cm(actual_position_delta)}\n"
                    f"         error    "
                    f"{fmt_cm(position_tracking_error)} "
                    f"rot="
                    f"{math.degrees(rotation_tracking_error):.2f}°\n"
                    f"         step x{fraction:g}, "
                    f"qstep="
                    f"{math.degrees(joint_step):.2f}°, "
                    f"seed #{seed_number}",
                    flush=True,
                )

        print("\n[STOP] Interrupt requested.")
        return 130

    except Exception as exc:
        print(
            f"\n[ABORT] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    finally:
        if arm is not None and enabled:
            try:
                hold_angles = list(
                    arm.get_angles()
                )
                arm._set_angles(
                    hold_angles
                )
                time.sleep(0.1)
            except Exception:
                pass

            if (
                old_velocities is not None
                and old_accelerations is not None
            ):
                try:
                    arm.set_velocities(
                        old_velocities
                    )
                    arm.set_accelerations(
                        old_accelerations
                    )
                except Exception:
                    pass

            try:
                print(
                    "[ACTION] Disabling right arm..."
                )
                arm.disable()
                time.sleep(0.2)
                print(
                    "[DISABLED] Right arm disabled."
                )
            except Exception as exc:
                print(
                    f"[WARNING] disable failed: {exc}",
                    file=sys.stderr,
                )

        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass

        udp_socket.close()
        print("Connections closed.")


if __name__ == "__main__":
    raise SystemExit(main())
