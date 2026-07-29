#!/usr/bin/env python3
"""
Right A7lite position-only hand-follow teleoperation v2.

Runs in the linker_a7 Conda environment.

Control chain:
  official XRoboToolkit right-controller transform
  -> relative hand translation only
  -> A7lite native URDF world frame
  -> numerical 3x7 TCP-position Jacobian
  -> damped least-squares differential joint step

Important safety behavior:
  * Pressing Grip captures references and holds the current seven joint angles.
    No inverse kinematics is called for the zero-displacement hold.
  * Wrist orientation is intentionally unconstrained.
  * Every command is checked against the per-cycle joint-step limit, the
    anchor envelope, and the SDK mechanical joint limits (including J3).
  * Unsafe or unsolved commands are held; a persistent failure disables the arm.
  * The default translation envelope is limited to 1 cm for initial testing.

Run only after the read-only hand probe is stable and the workspace is clear.
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

from linkerbot.arm.a7_lite.a7_lite import A7lite


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


def hand_position_from_pose(values: list[float]) -> np.ndarray:
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

    return array[:3].copy()


def pose_position(pose: Any) -> np.ndarray:
    position = np.array(
        [pose.x, pose.y, pose.z],
        dtype=float,
    )
    if not np.all(np.isfinite(position)):
        raise ValueError(
            "Forward kinematics returned a non-finite position"
        )
    return position


def forward_position(
    arm: A7lite,
    joint_angles: list[float] | np.ndarray,
) -> np.ndarray:
    pose = arm.forward_kinematics(
        [float(value) for value in joint_angles]
    )
    return pose_position(pose)


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


def numerical_position_jacobian(
    arm: A7lite,
    joint_angles: np.ndarray,
    base_position: np.ndarray,
    joint_limits: list[tuple[float, float]],
    limit_margin: float,
    epsilon: float,
) -> np.ndarray:
    """
    Numerically estimate d(TCP xyz) / d(joint angle).

    Central differences are used when both perturbations remain inside the
    protected joint range. Near a protected edge, a one-sided difference is
    used instead. No motor command is sent by this function.
    """
    joint_count = len(joint_angles)
    jacobian = np.zeros(
        (3, joint_count),
        dtype=float,
    )

    for index, (lower, upper) in enumerate(
        joint_limits
    ):
        plus_allowed = (
            joint_angles[index] + epsilon
            <= upper - limit_margin
        )
        minus_allowed = (
            joint_angles[index] - epsilon
            >= lower + limit_margin
        )

        if plus_allowed and minus_allowed:
            plus_angles = joint_angles.copy()
            minus_angles = joint_angles.copy()

            plus_angles[index] += epsilon
            minus_angles[index] -= epsilon

            plus_position = forward_position(
                arm,
                plus_angles,
            )
            minus_position = forward_position(
                arm,
                minus_angles,
            )

            jacobian[:, index] = (
                plus_position - minus_position
            ) / (2.0 * epsilon)

        elif plus_allowed:
            plus_angles = joint_angles.copy()
            plus_angles[index] += epsilon

            plus_position = forward_position(
                arm,
                plus_angles,
            )

            jacobian[:, index] = (
                plus_position - base_position
            ) / epsilon

        elif minus_allowed:
            minus_angles = joint_angles.copy()
            minus_angles[index] -= epsilon

            minus_position = forward_position(
                arm,
                minus_angles,
            )

            jacobian[:, index] = (
                base_position - minus_position
            ) / epsilon

        else:
            jacobian[:, index] = 0.0

    if not np.all(np.isfinite(jacobian)):
        raise ValueError(
            "Numerical position Jacobian is non-finite"
        )

    return jacobian


def damped_position_delta(
    jacobian: np.ndarray,
    position_error: np.ndarray,
    damping: float,
    current_joints: np.ndarray,
    joint_anchor: np.ndarray,
    posture_gain: float,
) -> np.ndarray:
    """
    Position-priority damped least-squares differential IK.

    A small null-space term gently prefers the Grip-on joint posture without
    constraining TCP orientation. The Cartesian position task remains primary.
    """
    identity_3 = np.eye(3)
    regularized = (
        jacobian @ jacobian.T
        + (damping ** 2) * identity_3
    )

    solved_error = np.linalg.solve(
        regularized,
        position_error,
    )
    joint_delta = (
        jacobian.T @ solved_error
    )

    if posture_gain > 0.0:
        damped_pseudoinverse = (
            jacobian.T
            @ np.linalg.solve(
                regularized,
                identity_3,
            )
        )
        null_projector = (
            np.eye(len(current_joints))
            - damped_pseudoinverse @ jacobian
        )
        posture_delta = (
            posture_gain
            * (joint_anchor - current_joints)
        )
        joint_delta = (
            joint_delta
            + null_projector @ posture_delta
        )

    if not np.all(np.isfinite(joint_delta)):
        raise ValueError(
            "Differential IK returned a non-finite joint step"
        )

    return joint_delta


def joint_candidate_rejection(
    candidate: np.ndarray,
    joint_anchor: np.ndarray,
    joint_limits: list[tuple[float, float]],
    joint_envelope: float,
    limit_margin: float,
) -> str | None:
    if (
        candidate.shape != joint_anchor.shape
        or not np.all(np.isfinite(candidate))
    ):
        return "invalid joint candidate"

    for index, (
        value,
        anchor,
        (lower, upper),
    ) in enumerate(
        zip(
            candidate,
            joint_anchor,
            joint_limits,
        ),
        start=1,
    ):
        if abs(value - anchor) > joint_envelope:
            return f"J{index} anchor envelope"

        if not (
            lower + limit_margin
            <= value
            <= upper - limit_margin
        ):
            return f"J{index} limit margin"

    return None


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interface",
        default="can2",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=48090,
    )
    parser.add_argument(
        "--source-port",
        type=int,
        default=48091,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )

    # Conservative first-test envelope: at most 1 cm.
    parser.add_argument(
        "--max-axis-cm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-radius-cm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--control-hz",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--tcp-speed-cm-s",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--position-deadband-mm",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--joint-envelope-deg",
        type=float,
        default=35.0,
    )
    parser.add_argument(
        "--limit-margin-deg",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--jacobian-step-deg",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--dls-damping",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--posture-gain",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--solver-tolerance-mm",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--watchdog",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--safe-hold-seconds",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    positive_arguments = {
        "control-hz": args.control_hz,
        "tcp-speed-cm-s": args.tcp_speed_cm_s,
        "max-axis-cm": args.max_axis_cm,
        "max-radius-cm": args.max_radius_cm,
        "max-joint-step-deg": args.max_joint_step_deg,
        "joint-envelope-deg": args.joint_envelope_deg,
        "limit-margin-deg": args.limit_margin_deg,
        "jacobian-step-deg": args.jacobian_step_deg,
        "dls-damping": args.dls_damping,
        "solver-tolerance-mm": args.solver_tolerance_mm,
        "watchdog": args.watchdog,
        "safe-hold-seconds": args.safe_hold_seconds,
    }
    for name, value in positive_arguments.items():
        if not math.isfinite(value) or value <= 0.0:
            parser.error(
                f"--{name} must be finite and positive"
            )

    if (
        not math.isfinite(args.scale)
        or args.scale <= 0.0
    ):
        parser.error(
            "--scale must be finite and positive"
        )

    if (
        not math.isfinite(args.position_deadband_mm)
        or args.position_deadband_mm < 0.0
    ):
        parser.error(
            "--position-deadband-mm must be finite "
            "and non-negative"
        )

    if (
        not math.isfinite(args.posture_gain)
        or args.posture_gain < 0.0
    ):
        parser.error(
            "--posture-gain must be finite "
            "and non-negative"
        )

    period = 1.0 / args.control_hz
    max_position_step = (
        args.tcp_speed_cm_s / 100.0
    ) * period

    max_joint_step = math.radians(
        args.max_joint_step_deg
    )
    joint_envelope = math.radians(
        args.joint_envelope_deg
    )
    limit_margin = math.radians(
        args.limit_margin_deg
    )
    jacobian_epsilon = math.radians(
        args.jacobian_step_deg
    )

    max_axis = args.max_axis_cm / 100.0
    max_radius = args.max_radius_cm / 100.0
    position_deadband = (
        args.position_deadband_mm / 1000.0
    )
    solver_tolerance = (
        args.solver_tolerance_mm / 1000.0
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
    robot_position_anchor = None
    joint_anchor = None

    commanded_position_delta = np.zeros(3)
    commanded_joints = None

    hold_cycles = 0
    tracking_bad_cycles = 0
    last_ready_report = 0.0
    last_live_report = 0.0

    print("=" * 88)
    print(
        "RIGHT A7lite position-only "
        "differential hand-follow v2"
    )
    print(
        "Robot frame: A7lite native URDF frame"
    )
    print(
        "Hand frame:  official XRoboToolkit "
        "transformed frame"
    )
    print(
        f"Position envelope: "
        f"±{args.max_axis_cm:.1f} cm, "
        f"radius {args.max_radius_cm:.1f} cm"
    )
    print(
        "Orientation: unconstrained "
        "(TCP position only)"
    )
    print(
        f"Joint step limit: "
        f"{args.max_joint_step_deg:.2f} deg/cycle"
    )
    print(
        "No home and no calibration."
    )
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

        actual_angles = list(
            arm.get_angles()
        )
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

        print(
            "[OK] Right A7lite connected."
        )
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

        if len(joint_limits) != len(actual_angles):
            print(
                "[BLOCK] Joint-limit count does not "
                "match joint count."
            )
            return 2

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
        print(
            "Initial test envelope is 1 cm. "
            "Do not increase joint-step limits."
        )

        if input(
            "Type OK to start: "
        ).strip() != "OK":
            print("[CANCELLED]")
            return 3

        old_velocities = list(
            arm.get_control_velocities()
        )
        old_accelerations = list(
            arm.get_control_acceleration()
        )

        arm.set_velocities(
            [0.18] * len(actual_angles)
        )
        arm.set_accelerations(
            [1.0] * len(actual_angles)
        )
        arm.enable()
        enabled = True

        commanded_joints = list(
            arm.get_angles()
        )

        print(
            "[ENABLED] Press right Grip to anchor "
            "position-only hand following.",
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
                if int(
                    packet.get("version", 0)
                ) != 5:
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

            hand_position = (
                hand_position_from_pose(
                    latest_packet[
                        "right_controller_pose_robot"
                    ]
                )
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

                joint_anchor = list(
                    arm.get_angles()
                )
                robot_position_anchor = (
                    forward_position(
                        arm,
                        joint_anchor,
                    )
                )

                commanded_position_delta = (
                    np.zeros(3)
                )
                commanded_joints = list(
                    joint_anchor
                )

                # Explicit zero-displacement hold:
                # no IK and no Jacobian are evaluated here.
                arm._set_angles(
                    commanded_joints
                )

                hold_cycles = 0
                tracking_bad_cycles = 0

                print(
                    "\n[GRIP ON] Position reference "
                    "captured; current joints held "
                    "without IK.",
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
                robot_position_anchor = None
                joint_anchor = None

                commanded_position_delta = (
                    np.zeros(3)
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
                or robot_position_anchor is None
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

            if (
                norm(desired_position_delta)
                <= position_deadband
            ):
                desired_position_delta = (
                    np.zeros(3)
                )

            requested_position_delta = (
                step_position(
                    commanded_position_delta,
                    desired_position_delta,
                    max_position_step,
                )
            )

            actual_angles = list(
                arm.get_angles()
            )

            command_array = np.asarray(
                commanded_joints,
                dtype=float,
            )
            anchor_array = np.asarray(
                joint_anchor,
                dtype=float,
            )

            current_command_position = (
                forward_position(
                    arm,
                    command_array,
                )
            )

            requested_cartesian_step = norm(
                requested_position_delta
                - commanded_position_delta
            )

            accepted = None
            rejection_reason = "no candidate"

            # At zero requested displacement, directly preserve the
            # current seven-joint command. This is the critical branch
            # that prevents a Grip-on IK solution jump.
            if requested_cartesian_step <= 1e-12:
                achieved_delta = (
                    current_command_position
                    - robot_position_anchor
                )
                accepted = (
                    achieved_delta,
                    command_array.copy(),
                    current_command_position,
                    current_command_position,
                    0.0,
                    0.0,
                    0.0,
                    math.nan,
                    1.0,
                    "hold/no IK",
                )
            else:
                jacobian = (
                    numerical_position_jacobian(
                        arm,
                        command_array,
                        current_command_position,
                        joint_limits,
                        limit_margin,
                        jacobian_epsilon,
                    )
                )

                singular_values = np.linalg.svd(
                    jacobian,
                    compute_uv=False,
                )
                sigma_min = float(
                    singular_values[-1]
                )
                sigma_max = float(
                    singular_values[0]
                )

                if sigma_max < 1e-8:
                    rejection_reason = (
                        "position Jacobian has "
                        "near-zero sensitivity"
                    )
                else:
                    for fraction in (
                        1.0,
                        0.5,
                        0.25,
                        0.125,
                        0.0625,
                        0.03125,
                    ):
                        candidate_requested_delta = (
                            commanded_position_delta
                            + fraction
                            * (
                                requested_position_delta
                                - commanded_position_delta
                            )
                        )
                        requested_target_position = (
                            robot_position_anchor
                            + candidate_requested_delta
                        )
                        position_error = (
                            requested_target_position
                            - current_command_position
                        )
                        error_before = norm(
                            position_error
                        )

                        try:
                            joint_delta = (
                                damped_position_delta(
                                    jacobian,
                                    position_error,
                                    args.dls_damping,
                                    command_array,
                                    anchor_array,
                                    args.posture_gain,
                                )
                            )
                        except np.linalg.LinAlgError as exc:
                            rejection_reason = (
                                "DLS solve failed: "
                                f"{exc}"
                            )
                            continue

                        raw_joint_step = float(
                            np.max(
                                np.abs(joint_delta)
                            )
                        )

                        joint_scale = 1.0
                        if (
                            raw_joint_step
                            > max_joint_step
                            and raw_joint_step > 1e-12
                        ):
                            joint_scale = (
                                max_joint_step
                                / raw_joint_step
                            )
                            joint_delta = (
                                joint_delta
                                * joint_scale
                            )

                        joint_step = float(
                            np.max(
                                np.abs(joint_delta)
                            )
                        )

                        if joint_step <= 1e-12:
                            rejection_reason = (
                                "differential joint "
                                "step is zero"
                            )
                            continue

                        candidate_joints = (
                            command_array
                            + joint_delta
                        )

                        joint_rejection = (
                            joint_candidate_rejection(
                                candidate_joints,
                                anchor_array,
                                joint_limits,
                                joint_envelope,
                                limit_margin,
                            )
                        )
                        if joint_rejection is not None:
                            rejection_reason = (
                                joint_rejection
                            )
                            continue

                        solved_position = (
                            forward_position(
                                arm,
                                candidate_joints,
                            )
                        )
                        solver_error = norm(
                            requested_target_position
                            - solved_position
                        )
                        achieved_cartesian_step = norm(
                            solved_position
                            - current_command_position
                        )

                        max_cartesian_command_step = max(
                            1.5 * max_position_step,
                            error_before
                            + solver_tolerance,
                        )
                        if (
                            achieved_cartesian_step
                            > max_cartesian_command_step
                        ):
                            rejection_reason = (
                                "Cartesian command step "
                                f"{1000.0 * achieved_cartesian_step:.2f} mm"
                            )
                            continue

                        # Accept either a close-enough solution or a
                        # meaningful decrease in Cartesian error.
                        close_enough = (
                            solver_error
                            <= solver_tolerance
                        )
                        improved = (
                            solver_error
                            <= 0.98 * error_before
                        )

                        if not (
                            close_enough
                            or improved
                        ):
                            rejection_reason = (
                                "DLS position error "
                                f"{1000.0 * solver_error:.2f} mm "
                                "did not improve"
                            )
                            continue

                        achieved_delta = (
                            solved_position
                            - robot_position_anchor
                        )
                        accepted = (
                            achieved_delta,
                            candidate_joints,
                            solved_position,
                            requested_target_position,
                            fraction,
                            joint_step,
                            solver_error,
                            sigma_min,
                            joint_scale,
                            "DLS position",
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
                        "No safe differential-position "
                        "step for the hold interval"
                    )
                continue

            hold_cycles = 0

            (
                commanded_position_delta,
                commanded_joint_array,
                target_position,
                requested_target_position,
                fraction,
                joint_step,
                solver_error,
                sigma_min,
                joint_scale,
                solver_mode,
            ) = accepted

            commanded_joints = [
                float(value)
                for value in commanded_joint_array
            ]

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

            actual_position = (
                forward_position(
                    arm,
                    actual_angles,
                )
            )
            actual_position_delta = (
                actual_position
                - robot_position_anchor
            )
            position_tracking_error = (
                target_position
                - actual_position
            )
            request_residual = (
                requested_target_position
                - target_position
            )

            if (
                now - last_live_report
                >= 0.25
            ):
                last_live_report = now

                if math.isnan(sigma_min):
                    solver_details = (
                        "hold/no IK"
                    )
                else:
                    solver_details = (
                        f"{solver_mode}, "
                        f"resid="
                        f"{1000.0 * solver_error:.2f} mm, "
                        f"sigma_min={sigma_min:.4f}"
                    )

                print(
                    f"[FOLLOW] hand xyz "
                    f"{fmt_cm(desired_position_delta)}\n"
                    f"         target   "
                    f"{fmt_cm(commanded_position_delta)}\n"
                    f"         actual   "
                    f"{fmt_cm(actual_position_delta)}\n"
                    f"         error    "
                    f"{fmt_cm(position_tracking_error)}\n"
                    f"         request  "
                    f"{fmt_cm(request_residual)}\n"
                    f"         step x{fraction:g}, "
                    f"qstep="
                    f"{math.degrees(joint_step):.2f}°, "
                    f"qscale={joint_scale:.3f}, "
                    f"{solver_details}",
                    flush=True,
                )

        print(
            "\n[STOP] Interrupt requested."
        )
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
