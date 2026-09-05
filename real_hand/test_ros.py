"""
Choreographed demo for the TetherIA Aero Hand.

Poses are 7 actuator values in radians (ACTUATOR_NAMES order). Motion between
poses is interpolated and streamed at 100 Hz -- the hand's own sensing/firmware
loop rate -- so the servos get a continuous trajectory instead of a step command
they have to chase. Stepping straight to a target looks jerky and slams the
tendons; easing does not.

Hardware-only -- this drives the real hand and never touches the simulation, so it
lives in real_hand/ rather than inside any one scene variant (mug, pen, card,
bottle).

Run with aero_hand_node up:
    ros2 run aero_hand_open aero_hand_node --ros-args -p right_port:=auto
    python3 real_hand/test_ros.py
"""

import math
import time

from aero_hand_bridge import ACTUATOR_NAMES, JOINT_NAMES, AeroHandBridge

# Actuator indices, for readable poses.
T_ABD, T_FLEX, T_CURL, INDEX, MIDDLE, RING, PINKY = range(7)
FINGERS = (INDEX, MIDDLE, RING, PINKY)

RATE_HZ = 100.0
DT = 1.0 / RATE_HZ

# Safe travel. send_actuator_positions() clamps anyway, but staying inside the
# limits here keeps the easing curves honest -- a clamped target silently
# flattens the end of a glide.
CURL = 1.50     # finger fully flexed   (limit 1.571)
HALF = 0.80     # finger half flexed
FLEX = 0.90     # thumb CMC flexion     (limit 0.956)


def pose(thumb_abd=0.0, thumb_flex=0.0, thumb_curl=0.0,
         index=0.0, middle=0.0, ring=0.0, pinky=0.0):
    return [thumb_abd, thumb_flex, thumb_curl, index, middle, ring, pinky]


OPEN      = pose()
FIST      = pose(1.20, 0.85, 1.60, CURL, CURL, CURL, CURL)
POINT     = pose(0.60, 0.40, 1.60, 0.0,  CURL, CURL, CURL)
PEACE     = pose(0.80, 0.50, 1.60, 0.0,  0.0,  CURL, CURL)
THREE     = pose(0.90, 0.60, 1.70, 0.0,  0.0,  0.0,  CURL)
FOUR      = pose(1.00, 0.70, 1.80, 0.0,  0.0,  0.0,  0.0)
THUMBS_UP = pose(0.0,  0.0,  0.0,  CURL, CURL, CURL, CURL)
ROCK_ON   = pose(1.00, 0.80, 1.70, 0.0,  CURL, CURL, 0.0)

# Thumb opposition. The thumb has to travel further to reach each successive
# finger, so abduction/flexion/curl all ramp together. These are geometric
# estimates -- if a pinch misses contact on your unit, trim the triple here.
PINCH = {
    "index":  pose(0.50, 0.80, 1.10, 1.00, 0.0,  0.0,  0.0),
    "middle": pose(0.70, 0.85, 1.20, 0.0,  1.10, 0.0,  0.0),
    "ring":   pose(0.90, FLEX, 1.35, 0.0,  0.0,  1.25, 0.0),
    "pinky":  pose(1.10, FLEX, 1.50, 0.0,  0.0,  0.0,  1.40),
}


def smoothstep(t):
    """Ease in/out -- zero velocity at both ends, so no jerk on arrival."""
    return t * t * (3.0 - 2.0 * t)


class Choreographer:
    def __init__(self, hand):
        self.hand = hand
        self.current = list(OPEN)

    def _tick(self, positions):
        self.hand.send_actuator_positions(positions)
        self.hand.spin_once(timeout_sec=0.0)
        time.sleep(DT)

    def glide(self, target, duration=0.8, label=None):
        if label:
            print(f"  {label}")
        start = list(self.current)
        steps = max(1, int(duration * RATE_HZ))
        for i in range(1, steps + 1):
            a = smoothstep(i / steps)
            self._tick([s + (t - s) * a for s, t in zip(start, target)])
        self.current = list(target)

    def hold(self, seconds):
        for _ in range(int(seconds * RATE_HZ)):
            self._tick(self.current)

    def ripple(self, cycles=3, period=1.1, amplitude=CURL, label=None):
        """Mexican wave down the fingers -- a phase-shifted cosine per finger."""
        if label:
            print(f"  {label}")
        steps = int(cycles * period * RATE_HZ)
        for i in range(steps):
            t = i / RATE_HZ
            p = list(OPEN)
            for k, finger in enumerate(FINGERS):
                phase = 2.0 * math.pi * (t / period - 0.18 * k)
                p[finger] = amplitude * 0.5 * (1.0 - math.cos(phase))
            self._tick(p)
        self.current = list(OPEN)
        self.glide(OPEN, 0.3)

    def piano(self, taps=2, label=None):
        """Quick independent finger taps, index -> pinky."""
        if label:
            print(f"  {label}")
        for _ in range(taps):
            for finger in FINGERS:
                down = list(OPEN)
                down[finger] = CURL
                self.glide(down, 0.16)
                self.glide(OPEN, 0.16)


def main():
    with AeroHandBridge() as hand:
        if not hand.wait_for_feedback(timeout_sec=5.0):
            raise RuntimeError("No feedback from hand — is aero_hand_node running?")

        show = Choreographer(hand)
        print("Aero Hand demo\n")

        try:
            show.glide(OPEN, 1.0, "open palm")
            show.hold(0.4)

            show.ripple(cycles=3, label="finger wave")
            show.hold(0.3)

            print("  counting 1 → 5")
            for name, p in (("one", POINT), ("two", PEACE), ("three", THREE),
                            ("four", FOUR), ("five", OPEN)):
                show.glide(p, 0.45)
                show.hold(0.35)

            show.glide(ROCK_ON, 0.6, "rock on")
            show.hold(0.6)
            show.glide(THUMBS_UP, 0.7, "thumbs up")
            show.hold(0.8)
            show.glide(OPEN, 0.6)

            print("  thumb opposition")
            for finger in ("index", "middle", "ring", "pinky"):
                show.glide(PINCH[finger], 0.55)
                show.hold(0.3)
                show.glide(OPEN, 0.35)

            show.piano(taps=2, label="piano taps")

            show.glide(FIST, 1.2, "slow clench")
            show.hold(1.0)
            show.glide(OPEN, 1.2, "release")
            show.hold(0.5)

        except KeyboardInterrupt:
            print("\ninterrupted — returning to open palm")
            show.glide(OPEN, 0.8)

        # Settle, then report where the hand actually ended up.
        show.hold(0.5)
        actuators = hand.get_actuator_feedback()
        joints = hand.get_joint_feedback()

        print("\nfinal pose (commanded → measured, radians):")
        for name, cmd, meas in zip(ACTUATOR_NAMES, show.current, actuators):
            print(f"  {name:<26}{cmd:>7.4f} → {meas:>7.4f}  ({meas - cmd:+.4f})")

        print("\njoint positions on the real hand (radians):")
        for name, q in zip(JOINT_NAMES, joints):
            print(f"  {name:<26}{q:>8.4f}")


if __name__ == "__main__":
    main()
