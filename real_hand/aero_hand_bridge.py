# aero_hand_bridge.py
"""
Abstraction layer between 7-actuator control values and the real TetherIA
Aero Hand, over ROS2. Callers only ever deal with 7 floats (radians) --
this module hides joint-space expansion, unit conversion, and ROS2 plumbing.

    with AeroHandBridge() as hand:
        hand.send_actuator_positions([0.0, 0.3, 0.35, 0.6, 0.6, 0.6, 0.6])
        feedback = hand.get_actuator_feedback()   # 7 floats, radians, or None
        joints   = hand.get_joint_feedback()      # 16 floats, radians, or None

Hardware-only and object-agnostic, so it lives in real_hand/ alongside the rest of
the ROS2/physical-hand code, separate from the simulation variants.
Variant scripts run from their own directory, so reach it with:

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "real_hand"))
    from aero_hand_bridge import AeroHandBridge

Assumes aero_hand_node (the hardware node) is already running and publishing
on /{side}/actuator_states, /{side}/joint_control. If not, run the command:
    
    ros2 run aero_hand_open aero_hand_node --ros-args -p right_port:=auto -p control_space:=joint

to begin publishing actuator states and accepting joint targets. The node will auto-detect the hand on USB and start streaming.

NOTE ON FEEDBACK: the hardware publishes ONLY raw motor angles (ActuatorStates.
actuations, 7 values, degrees). It does not publish joint positions -- the SDK's
get_joint_positions() is an unimplemented stub. Joint feedback here is therefore
*reconstructed* by inverting the tendon model, using the same 1.0/0.6/0.6
mcp/pip/dip coupling assumption that send_actuator_positions() commands with.
It is a model estimate, not a joint encoder reading: the hand has no joint-level
sensing, so tendon stretch and any object blocking a finger are invisible to it.
"""

import math
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from aero_hand_open_msgs.msg import ActuatorStates, JointControl


# ---------------------------------------------------------------------------
# Fixed hand geometry. Must stay identical to the coupling in EVERY variant's
# aero_hand_grasp_env.py (mug, pen, card, bottle) -- they all share this bridge,
# so a mismatch is a silent-wrong-behavior bug, not an error.
# ---------------------------------------------------------------------------

ACTUATOR_NAMES = [
    "thumb_abduction_actuator",
    "thumb_flex_actuator",
    "thumb_tendon_actuator",
    "index_finger_actuator",
    "middle_finger_actuator",
    "ring_finger_actuator",
    "pinky_finger_actuator",
]

# JointControl's own field order -- confirmed via `ros2 interface show aero_hand_open_msgs/msg/JointControl`
JOINT_NAMES = [
    "right_thumb_cmc_abd", "right_thumb_cmc_flex", "right_thumb_mcp", "right_thumb_ip",
    "right_index_mcp_flex", "right_index_pip", "right_index_dip",
    "right_middle_mcp_flex", "right_middle_pip", "right_middle_dip",
    "right_ring_mcp_flex", "right_ring_pip", "right_ring_dip",
    "right_pinky_mcp_flex", "right_pinky_pip", "right_pinky_dip",
]

ACTUATOR_JOINT_MAP = {
    "thumb_abduction_actuator": {"right_thumb_cmc_abd": 1.0},
    "thumb_flex_actuator":      {"right_thumb_cmc_flex": 1.0},
    "thumb_tendon_actuator":    {"right_thumb_mcp": 0.6, "right_thumb_ip": 0.6},
    "index_finger_actuator":    {"right_index_mcp_flex": 1.0, "right_index_pip": 0.6, "right_index_dip": 0.6},
    "middle_finger_actuator":   {"right_middle_mcp_flex": 1.0, "right_middle_pip": 0.6, "right_middle_dip": 0.6},
    "ring_finger_actuator":     {"right_ring_mcp_flex": 1.0, "right_ring_pip": 0.6, "right_ring_dip": 0.6},
    "pinky_finger_actuator":    {"right_pinky_mcp_flex": 1.0, "right_pinky_pip": 0.6, "right_pinky_dip": 0.6},
}

# Verified from aero_hand.usda's physics:lowerLimit / physics:upperLimit (degrees)
JOINT_LIMITS_DEG = {
    "right_index_mcp_flex": (0.0, 90.000206), "right_index_pip": (0.0, 90.000206), "right_index_dip": (0.0, 90.000206),
    "right_middle_mcp_flex": (0.0, 90.000206), "right_middle_pip": (0.0, 90.000206), "right_middle_dip": (0.0, 90.000206),
    "right_ring_mcp_flex": (0.0, 90.000206), "right_ring_pip": (0.0, 90.000206), "right_ring_dip": (0.0, 90.000206),
    "right_pinky_mcp_flex": (0.0, 90.000206), "right_pinky_pip": (0.0, 90.000206), "right_pinky_dip": (0.0, 90.000206),
    "right_thumb_cmc_abd": (0.0, 99.99833), "right_thumb_cmc_flex": (0.0, 54.76903),
    "right_thumb_mcp": (0.0, 90.000206), "right_thumb_ip": (0.0, 90.000206),
}


# ---------------------------------------------------------------------------
# Tendon model. Mirrors aero_open_sdk.joints_to_actuations (MOTOR_PULLEY_RADIUS,
# FingerCoeffs, ThumbFlexCoeffs, ThumbIPCoeffs) so this file stays dependency-free
# like the block above. Coefficients are mm/radian; pulley radius is mm.
#
# motor_actuation[a] = sum_j( TENDON_COEFFS[a][j] * joint[j] ) / MOTOR_PULLEY_RADIUS
#
# This is exactly what aero_hand_node applies to the joint targets we publish, so
# inverting it recovers our own actuator/joint values from ActuatorStates.
# ---------------------------------------------------------------------------

MOTOR_PULLEY_RADIUS = 9.000  # mm

TENDON_COEFFS = {
    # Abduction is a direct joint->motor mapping; the pulley radius cancels.
    "thumb_abduction_actuator": {"right_thumb_cmc_abd": MOTOR_PULLEY_RADIUS},
    "thumb_flex_actuator": {"right_thumb_cmc_abd": 2.5000, "right_thumb_cmc_flex": 12.4931},
    "thumb_tendon_actuator": {
        "right_thumb_cmc_abd": 2.5000,
        "right_thumb_cmc_flex": -2.5000,   # sign is intentional, per the SDK model
        "right_thumb_mcp": 9.4372,
        "right_thumb_ip": 12.5000,
    },
    "index_finger_actuator":  {"right_index_mcp_flex": 12.4912,  "right_index_pip": 7.3211,  "right_index_dip": 9.0000},
    "middle_finger_actuator": {"right_middle_mcp_flex": 12.4912, "right_middle_pip": 7.3211, "right_middle_dip": 9.0000},
    "ring_finger_actuator":   {"right_ring_mcp_flex": 12.4912,   "right_ring_pip": 7.3211,   "right_ring_dip": 9.0000},
    "pinky_finger_actuator":  {"right_pinky_mcp_flex": 12.4912,  "right_pinky_pip": 7.3211,  "right_pinky_dip": 9.0000},
}


def _build_coupling():
    """7x16: joint_targets[j] = sum_a( actuator_positions[a] * coupling[a][j] )"""
    coupling = [[0.0] * len(JOINT_NAMES) for _ in range(len(ACTUATOR_NAMES))]
    actuator_upper = []
    for a_idx, a_name in enumerate(ACTUATOR_NAMES):
        ratios = ACTUATOR_JOINT_MAP[a_name]
        bounds = []
        for j_name, ratio in ratios.items():
            j_idx = JOINT_NAMES.index(j_name)
            coupling[a_idx][j_idx] = ratio
            bounds.append(math.radians(JOINT_LIMITS_DEG[j_name][1]) / ratio)
        actuator_upper.append(min(bounds))
    return coupling, actuator_upper


def _build_motor_matrix(coupling):
    """
    7x7: motor_rad[a] = sum_b( actuator_positions[b] * motor[a][b] )

    Composition of our actuator->joint coupling with the SDK's joint->motor tendon
    model. Because the thumb tendons are cross-coupled (the thumb tendon motor is
    affected by abduction and flexion), this is lower-triangular rather than
    diagonal -- which is precisely why feedback cannot be a per-actuator scale.
    """
    n_act = len(ACTUATOR_NAMES)
    motor = [[0.0] * n_act for _ in range(n_act)]
    for a_idx, a_name in enumerate(ACTUATOR_NAMES):
        coeffs = TENDON_COEFFS[a_name]
        for b_idx in range(n_act):
            total = 0.0
            for j_name, c in coeffs.items():
                total += c * coupling[b_idx][JOINT_NAMES.index(j_name)]
            motor[a_idx][b_idx] = total / MOTOR_PULLEY_RADIUS

    # Guard the forward-substitution assumption below: if someone edits
    # ACTUATOR_JOINT_MAP into a shape that couples an actuator to a *later*
    # motor channel, fail loudly instead of returning quietly wrong angles.
    for i in range(n_act):
        if abs(motor[i][i]) < 1e-9:
            raise ValueError(f"Actuator '{ACTUATOR_NAMES[i]}' has no motor influence; cannot invert.")
        for j in range(i + 1, n_act):
            if abs(motor[i][j]) > 1e-9:
                raise ValueError(
                    "Motor matrix is not lower-triangular "
                    f"(row {ACTUATOR_NAMES[i]}, col {ACTUATOR_NAMES[j]}); "
                    "_motor_to_actuator needs a general solver."
                )
    return motor


class AeroHandBridge:
    def __init__(self, side: str = "right", node_name: str = "aero_hand_bridge"):
        self.side = side
        self._coupling, actuator_upper = _build_coupling()
        self._motor = _build_motor_matrix(self._coupling)
        self._actuator_lower = [0.0] * len(ACTUATOR_NAMES)
        self._actuator_upper = actuator_upper

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        self._node = Node(node_name)
        self._pub = self._node.create_publisher(JointControl, f"/{side}/joint_control", 10)
        self._node.create_subscription(
            ActuatorStates, f"/{side}/actuator_states", self._on_actuator_states, 10
        )

        self._latest_motor_rad = None
        self._latest_actuator_rad = None
        self._latest_joint_rad = None
        self._latest_actuator_speed = None
        self._latest_actuator_current = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Outgoing: 7 actuator positions -> real hand
    # ------------------------------------------------------------------
    def send_actuator_positions(self, actuator_positions):
        """actuator_positions: 7 floats, radians, in ACTUATOR_NAMES order. Clamped before sending."""
        if len(actuator_positions) != len(ACTUATOR_NAMES):
            raise ValueError(f"Expected {len(ACTUATOR_NAMES)} values, got {len(actuator_positions)}")

        clamped = [
            max(self._actuator_lower[i], min(self._actuator_upper[i], v))
            for i, v in enumerate(actuator_positions)
        ]
        joint_targets = self._actuator_to_joints(clamped)

        msg = JointControl()
        msg.header = Header()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.target_positions = joint_targets   # already in JointControl's own order
        self._pub.publish(msg)

    # ------------------------------------------------------------------
    # Incoming: real hand -> 7 actuator positions
    # ------------------------------------------------------------------
    def _motor_to_actuator(self, motor_rad):
        """
        Invert _build_motor_matrix by forward substitution (it is lower-triangular,
        asserted at construction). Returns 7 actuator positions in radians -- the
        same space send_actuator_positions() consumes, so a command round-trips.
        """
        n_act = len(ACTUATOR_NAMES)
        actuator = [0.0] * n_act
        for i in range(n_act):
            residual = motor_rad[i] - sum(self._motor[i][k] * actuator[k] for k in range(i))
            actuator[i] = residual / self._motor[i][i]
        return actuator

    def _actuator_to_joints(self, actuator_rad):
        """7 actuator positions (radians) -> 16 joint positions (radians), JOINT_NAMES order."""
        return [
            sum(actuator_rad[a] * self._coupling[a][j] for a in range(len(ACTUATOR_NAMES)))
            for j in range(len(JOINT_NAMES))
        ]

    def _on_actuator_states(self, msg: ActuatorStates):
        # ActuatorStates.actuations is 7 raw MOTOR angles in degrees (post-tendon),
        # in ACTUATOR_NAMES order. It is NOT the actuator-position space that
        # send_actuator_positions() takes -- converting degrees to radians alone
        # leaves you off by the tendon ratio (~2.5x on the fingers). Undo the
        # tendon model to get back to actuator, then to joint space.
        motor_rad = [math.radians(d) for d in msg.actuations]
        actuator_rad = self._motor_to_actuator(motor_rad)
        joint_rad = self._actuator_to_joints(actuator_rad)

        with self._lock:
            self._latest_motor_rad = motor_rad
            self._latest_actuator_rad = actuator_rad
            self._latest_joint_rad = joint_rad
            self._latest_actuator_speed = list(msg.actuator_speeds)
            self._latest_actuator_current = list(msg.actuator_currents)

    def get_actuator_feedback(self):
        """
        7 floats, radians, ACTUATOR_NAMES order -- directly comparable to the list
        you pass to send_actuator_positions(). None if no feedback received yet.
        """
        with self._lock:
            return list(self._latest_actuator_rad) if self._latest_actuator_rad is not None else None

    def get_joint_feedback(self):
        """
        16 floats, radians, JOINT_NAMES order -- the estimated joint positions of
        the real hand. None if no feedback received yet.

        Reconstructed from motor angles via the tendon model (see module docstring);
        the hand has no joint encoders.
        """
        with self._lock:
            return list(self._latest_joint_rad) if self._latest_joint_rad is not None else None

    def get_motor_feedback(self):
        """
        7 floats, radians -- the RAW motor angles as reported by the hardware,
        before the tendon model is undone. Use for diagnostics; use
        get_actuator_feedback() to compare against commands. None if no feedback yet.
        """
        with self._lock:
            return list(self._latest_motor_rad) if self._latest_motor_rad is not None else None

    def get_actuator_speed_feedback(self):
        """7 floats, RPM -- or None."""
        with self._lock:
            return list(self._latest_actuator_speed) if self._latest_actuator_speed is not None else None

    def get_actuator_current_feedback(self):
        """7 floats, mA -- a torque PROXY, not real Nm. Or None."""
        with self._lock:
            return list(self._latest_actuator_current) if self._latest_actuator_current is not None else None

    # ------------------------------------------------------------------
    # ROS2 plumbing
    # ------------------------------------------------------------------
    def spin_once(self, timeout_sec: float = 0.0):
        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def wait_for_feedback(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.time() + timeout_sec
        while self.get_actuator_feedback() is None:
            if time.time() > deadline:
                return False
            self.spin_once(timeout_sec=0.05)
        return True

    def shutdown(self):
        self._node.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()