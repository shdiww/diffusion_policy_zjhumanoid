"""
Streaming trajectory replay for zj_humanoid (== deployment pipeline rehearsal).

Reads a recorded episode and replays it through the same streaming interfaces
that eval/deployment will use:

  joint space:     publish upperlimb/Joints -> /zj_humanoid/upperlimb/servoj/{arm}
  cartesian space: publish upperlimb/DualPose -> /zj_humanoid/upperlimb/servol/right_arm
                   (current right-arm-only (N,6) dataset only; fixed 30Hz,
                   ONLY right_arm_pose assigned; official SDK servo params)
  hand:            call /zj_humanoid/hand/joint_switch/{arm} along the same timeline

Speed: joint velocity = per-frame joint delta / frame interval (30Hz recorded).
Cartesian right-arm replay uses a fixed 30Hz frame-by-frame timeline.

Safety flow:
  --unlock -> [--gohome: check vs home sequence, movej home if off,
               then one right-hand OPEN_POSE reset]
  -> align to episode first frame (movej, slow)
  -> stream (--hand on: per-frame hand replay) -> hold last frame
  -> --verify prints tracking error

This script SENDS MOTION COMMANDS to the real robot when you run it.
Keep the emergency stop within reach!

Usage (inside naviai_learn_gpu):
    source /home/naviai/diffusion_ws/scripts/zj_collect_env.sh
    cd /home/naviai/diffusion_ws/diffusion_policy
    python replay_real_zj.py -i <data_dir> -e 0 --arm right --gohome --verify
    python replay_real_zj.py -i <data_dir> -e 0 --space cartesian --arm right --hand off --gohome --unlock
"""
import time
import pathlib
import click
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot

import rospy
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from upperlimb.msg import Joints, DualPose
from geometry_msgs.msg import Pose, Point, Quaternion
from zj_humanoid.upperlimb.srv import MoveJ, Servo, ServoRequest
from zj_humanoid.hand.srv import HandJoint
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world_zj.zj_home import (
    HOME_JS, OPEN_POSE, is_home)

# robot_joint (19) column ranges: [L7, R7, Neck2, Waist2, Lift1]
ARM_COLS = {'left': (0, 7), 'right': (7, 14), 'dual': (0, 14)}
ARM_TYPES = {'left': 1, 'right': 2, 'dual': 3}
ARM_SUFFIX = {'left': 'left_arm', 'right': 'right_arm', 'dual': 'dual_arm'}
# hand joint columns in robot_hand (12): [L6, R6]
HAND_COLS = {'left': (0, 6), 'right': (6, 12), 'dual': (0, 12)}
HAND_SUFFIX = {'left': 'left', 'right': 'right', 'dual': 'dual'}

NS = '/zj_humanoid'
SRV_MOVEJ = NS + '/upperlimb/movej/{arm}'
SRV_UNLOCK = NS + '/upperlimb/unlock'
SRV_HAND = NS + '/hand/joint_switch/{arm}'
SRV_SET_SERVO_PARAMS = NS + '/upperlimb/set_servo_params'
SRV_CLEAR_SERVO_PARAMS = NS + '/upperlimb/clear_servo_params'
TOPIC_SERVOJ = NS + '/upperlimb/servoj/{arm}'
TOPIC_SERVOL = NS + '/upperlimb/servol/{arm}'

# official SDK streaming servo params (fixed, not user-tunable)
SERVO_TIME = 0.02
SERVO_GAIN = 800


def current_joints():
    msg = rospy.wait_for_message(
        '/zj_humanoid/upperlimb/joint_states', JointState, timeout=10)
    return np.asarray(msg.position, dtype=np.float64)


# global latest-joints cache for --verify (avoids per-frame wait_for_message)
_LATEST_JOINTS = {'arr': None}


def _joint_sub_cb(msg):
    _LATEST_JOINTS['arr'] = np.asarray(msg.position, dtype=np.float64)


def _start_joint_cache():
    if _LATEST_JOINTS.get('sub') is None:
        _LATEST_JOINTS['sub'] = rospy.Subscriber(
            '/zj_humanoid/upperlimb/joint_states', JointState,
            _joint_sub_cb, queue_size=10)


def current_joints_latest():
    _start_joint_cache()
    return _LATEST_JOINTS['arr']


def rpy_to_pose_msg(pose6):
    q = ScipyRot.from_euler('xyz', pose6[3:6]).as_quat()
    return Pose(position=Point(x=pose6[0], y=pose6[1], z=pose6[2]),
                orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))


def unlock():
    rospy.wait_for_service(SRV_UNLOCK, timeout=10)
    resp = rospy.ServiceProxy(SRV_UNLOCK, Trigger)()
    if not resp.success:
        raise RuntimeError(f"unlock ({SRV_UNLOCK}) failed: {resp.message}")
    print(f"Safety lock released ({SRV_UNLOCK}).")


def set_servo_params():
    """Enable streaming servo params on the robot side (official SDK flow).

    Uses the fixed official values time=0.02, gain=800 (not user-tunable)
    and targets the right arm (arm_type=2). Must be called after the robot
    is aligned and before any servol stream. Raises RuntimeError if the
    service rejects the request, so streaming never starts with parameters
    left unset.
    """
    rospy.wait_for_service(SRV_SET_SERVO_PARAMS, timeout=10)
    resp = rospy.ServiceProxy(SRV_SET_SERVO_PARAMS, Servo)(
        ServoRequest(time=SERVO_TIME, gain=SERVO_GAIN,
                     arm_type=ARM_TYPES['right']))
    if not resp.success:
        raise RuntimeError(
            f"set_servo_params (time={SERVO_TIME}, gain={SERVO_GAIN}, "
            f"arm_type={ARM_TYPES['right']}) failed: {resp.message}")
    print(f"set_servo_params ok (time={SERVO_TIME}, gain={SERVO_GAIN}, "
          f"arm_type={ARM_TYPES['right']}).")


def clear_servo_params():
    """Disable streaming servo params (official SDK finish flow).

    Targets the right arm (arm_type=2). Runs in the finally path of a
    Cartesian stream, so it must never hide the primary stream exception:
    any service failure here is logged as a warning and swallowed.
    """
    try:
        rospy.wait_for_service(SRV_CLEAR_SERVO_PARAMS, timeout=10)
        resp = rospy.ServiceProxy(SRV_CLEAR_SERVO_PARAMS, Servo)(
            ServoRequest(arm_type=ARM_TYPES['right']))
        if not resp.success:
            print(f"[warn] clear_servo_params (arm_type="
                  f"{ARM_TYPES['right']}) failed: {resp.message}")
        else:
            print("clear_servo_params ok.")
    except Exception as e:  # noqa: BLE001 - must not hide stream errors
        print(f"[warn] clear_servo_params error: {e}")


def movej(joints, arm_suffix, t, arm_type):
    srv = SRV_MOVEJ.format(arm=arm_suffix)
    rospy.wait_for_service(srv, timeout=10)
    call = rospy.ServiceProxy(srv, MoveJ)
    resp = call(joints=np.asarray(joints, dtype=np.float64).tolist(),
                v=0.0, acc=0.0, t=t, is_async=False, arm_type=arm_type)
    if not resp.success:
        raise RuntimeError(f"movej({srv}) failed: {resp.message}")
    print(f"movej({srv}) ok ({t}s).")


def ensure_home(thresh=0.1):
    cur = current_joints()
    ok, err = is_home(cur, thresh)
    if ok:
        print(f"Already near home (max err {err:.4f} rad).")
        return
    print(f"Not at home (max err {err:.4f} rad > {thresh}). Moving home...")
    movej(HOME_JS, 'whole_body', t=8.0, arm_type=31)
    time.sleep(1.0)
    cur2 = current_joints()
    _, err2 = is_home(cur2, thresh)
    if err2 > thresh:
        raise RuntimeError(
            f"still {err2:.4f} rad from home after movej "
            f"(threshold {thresh}); aborting.")


def hand_to_pose(pose6, arm_hand):
    srv = SRV_HAND.format(arm=arm_hand)
    rospy.wait_for_service(srv, timeout=10)
    call = rospy.ServiceProxy(srv, HandJoint)
    resp = call(np.asarray(pose6, dtype=np.float32).tolist())
    if not resp.success:
        print(f"[warn] hand {srv} failed: {resp.message}")


def load_episode(data_dir, episode_idx):
    rb = ReplayBuffer.create_from_path(
        str(pathlib.Path(data_dir).joinpath(
            'replay_buffer.zarr').absolute()), mode='r')
    if episode_idx >= rb.n_episodes:
        raise RuntimeError(f"episode {episode_idx} not found, "
                           f"have {rb.n_episodes}")
    return rb.get_episode(episode_idx)


def stream_joint(ep, rel_t, arm, rate, verify, hand_call=None):
    # robot_joint: new right-arm-only data is 7-dim; legacy is 19-dim [L7 R7 N W L]
    q_all = ep['robot_joint']
    if q_all.shape[1] == 7:
        cols = (0, 7)  # right-arm-only format
    else:
        cols = ARM_COLS[arm]
    q = q_all[:, cols[0]:cols[1]]
    n = len(q)
    ts = rel_t / rate
    pub = rospy.Publisher(
        TOPIC_SERVOJ.format(arm=ARM_SUFFIX[arm]), Joints, queue_size=10)
    time.sleep(0.5)  # let publisher connect
    t0 = time.monotonic()
    actual = []
    hand_cols = HAND_COLS[arm]
    hand_arr = ep.get('robot_hand')
    if hand_arr is not None and hand_arr.shape[1] == 6:
        hand_cols = (0, 6)  # right-hand-only format
    for i in range(n):
        precise_wait(t0 + ts[i])
        pub.publish(Joints(joint=q[i].tolist()))
        if hand_call is not None:
            try:
                hand_call(np.asarray(
                    ep['robot_hand'][i, hand_cols[0]:hand_cols[1]],
                    dtype=np.float32).tolist())
            except rospy.ServiceException as e:
                print(f"[warn] hand step {i} failed: {e}")
        if verify:
            a = current_joints_latest()
            if a is not None:
                actual.append(a[cols[0]:cols[1]].copy())
                if i % 30 == 0 or i == n - 1:
                    print(f"[verify] frame {i}: err="
                          f"{np.abs(a[cols[0]:cols[1]] - q[i]).max():.4f}")
    # hold last frame briefly
    hold = Joints(joint=q[-1].tolist())
    for _ in range(10):
        precise_wait(t0 + ts[-1] + (_ + 1) * (1 / 30))
        pub.publish(hold)
    pub.unregister()
    if verify:
        return np.array(actual), q
    return None, None


def stream_cartesian_right(ep, hand_call=None):
    """Stream current right-arm-only (N,6) EEF poses at fixed 30Hz.

    Official SDK flow: one DualPose() per recorded frame with ONLY
    right_arm_pose assigned (no left pose reads/fields, no interpolation,
    no resampling). Timeline is fixed 1/30s per frame regardless of the
    recorded timestamps. Holds the last message for 10 frames at 30Hz.
    """
    eef = ep['robot_eef_pose']
    n = len(eef)
    pub = rospy.Publisher(
        TOPIC_SERVOL.format(arm='right_arm'), DualPose, queue_size=1)
    time.sleep(0.5)  # let publisher connect
    t0 = time.monotonic()
    hand_arr = ep.get('robot_hand')
    hand_cols = (0, 6)  # current right-hand-only format
    if hand_arr is not None and hand_arr.shape[1] == 12:
        hand_cols = (6, 12)  # legacy: right-hand columns [6:12]
    last = None
    try:
        for i in range(n):
            precise_wait(t0 + i / 30)
            last = DualPose()
            last.right_arm_pose = rpy_to_pose_msg(eef[i])
            pub.publish(last)
            if hand_call is not None:
                try:
                    hand_call(np.asarray(
                        hand_arr[i, hand_cols[0]:hand_cols[1]],
                        dtype=np.float32).tolist())
                except rospy.ServiceException as e:
                    print(f"[warn] hand step {i} failed: {e}")
        # hold last right message 10 frames at 30Hz; first hold at n/30,
        # exactly one tick after the last streamed frame i=n-1 at (n-1)/30
        for k in range(10):
            precise_wait(t0 + (n + k) / 30)
            pub.publish(last)
    finally:
        pub.unregister()


def report_verify(actual, target):
    """actual: (M, K) joint readings (may be fewer/lagged), target: (N, K)"""
    if actual is None:
        return
    a = np.array(actual)
    n = min(len(a), len(target))
    if n == 0:
        print("(no actual samples collected)")
        return
    err = np.abs(a[:n] - np.asarray(target)[:n])
    print(f"\n-- Verify (first {n}/{len(target)} frames) --")
    print(f"  per-joint max err: {np.round(err.max(axis=0), 4)} rad")
    print(f"  overall max/mean: {err.max():.4f} / {err.mean():.4f} rad")


def validate_replay_inputs(ep, space, arm, hand):
    """Pre-motion validation of episode data vs replay options.

    Raises click.UsageError before any robot motion is set up.
    """
    ts = ep['timestamp']
    if ts.ndim != 1 or ts.shape[0] == 0:
        raise click.UsageError(
            f"timestamp must be a non-empty 1-D array, got shape {ts.shape}.")
    if not np.all(np.isfinite(ts)):
        raise click.UsageError("timestamp contains non-finite values.")
    if np.any(np.diff(ts) <= 0):
        raise click.UsageError("timestamp must be strictly increasing.")
    n = ts.shape[0]
    if 'robot_joint' not in ep:
        raise click.UsageError("dataset has no 'robot_joint' key.")
    q = ep['robot_joint']
    if q.ndim != 2 or q.shape[0] != n or q.shape[1] not in (7, 19):
        raise click.UsageError(
            f"robot_joint must be (N, 7) or (N, 19), got {q.shape}.")
    if not np.all(np.isfinite(q)):
        raise click.UsageError("robot_joint contains non-finite values.")
    if q.shape[1] == 7 and arm != 'right':
        raise click.UsageError(
            "robot_joint (N, 7) is the right-arm-only format; "
            f"--arm must be 'right', got '{arm}'.")
    if hand == 'on':
        if 'robot_hand' not in ep:
            raise click.UsageError(
                "--hand on requires 'robot_hand' in dataset.")
        h = ep['robot_hand']
        if h.ndim != 2 or h.shape[0] != n or h.shape[1] not in (6, 12):
            raise click.UsageError(
                f"robot_hand must be (N, 6) or (N, 12), got {h.shape}.")
        if not np.all(np.isfinite(h)):
            raise click.UsageError("robot_hand contains non-finite values.")
        if h.shape[1] == 6 and arm != 'right':
            raise click.UsageError(
                "robot_hand (N, 6) is the right-hand-only format; "
                f"--arm must be 'right', got '{arm}'.")
    if space == 'cartesian':
        if 'robot_eef_pose' not in ep:
            raise click.UsageError(
                "cartesian replay requires 'robot_eef_pose' in dataset.")
        eef = ep['robot_eef_pose']
        if eef.ndim != 2 or eef.shape[0] != n:
            raise click.UsageError(
                f"robot_eef_pose must be (N, 6), got {eef.shape}.")
        if eef.shape[1] != 6:
            raise click.UsageError(
                "cartesian replay supports only the current right-arm-only "
                f"(N, 6) robot_eef_pose, got {eef.shape}; legacy (N, 12) "
                "left/dual cartesian is not supported.")
        if arm != 'right':
            raise click.UsageError(
                "cartesian replay supports only --arm right with the "
                f"current (N, 6) dataset, got --arm '{arm}'.")
        if not np.all(np.isfinite(eef)):
            raise click.UsageError("robot_eef_pose contains non-finite values.")


@click.command()
@click.option('--input', '-i', required=True, help='Dataset directory.')
@click.option('--episode', '-e', default=0, type=int, help='Episode index.')
@click.option('--space', '-sp', default='joint',
              type=click.Choice(['joint', 'cartesian']),
              help='Streaming space: joint (servoj) or cartesian (servol EEF).')
@click.option('--arm', default='left',
              type=click.Choice(['left', 'right', 'dual']),
              help='Which arm(s) to replay.')
@click.option('--hand', default='on', type=click.Choice(['on', 'off']),
              help='Replay hand joints too (same timeline).')
@click.option('--gohome', is_flag=True, default=False,
              help='Ensure robot is at home before replay (movej if off).')
@click.option('--home-thresh', default=0.1, type=float,
              help='Home detection tolerance per joint (rad).')
@click.option('--unlock', 'unlock_flag', is_flag=True, default=False,
              help='Release safety lock before replay.')
@click.option('--verify', is_flag=True, default=False,
              help='Record actual joints during replay and report error.')
@click.option('--skip-confirm', is_flag=True, default=False,
              help='Do not ask for confirmation before moving.')
def main(input, episode, space, arm, hand, gohome, home_thresh,
         unlock_flag, verify, skip_confirm):
    rospy.init_node('replay_real_zj', anonymous=True)

    ep = load_episode(input, episode)
    validate_replay_inputs(ep, space, arm, hand)
    rel_t = ep['timestamp'] - ep['timestamp'][0]
    n = len(rel_t)
    if space == 'cartesian':
        total_t = n / 30  # fixed 30Hz frame-by-frame EEF stream
    else:
        total_t = rel_t[-1]
    print(f"Episode {episode}: {n} frames, recorded {rel_t[-1]:.2f}s, "
          f"replay timeline {total_t:.2f}s")
    print(f"space: {space} | arm: {arm} | hand: {hand} | verify: {verify}")

    if space == 'cartesian':
        stream_topic = TOPIC_SERVOL.format(arm=ARM_SUFFIX[arm])
    else:
        stream_topic = TOPIC_SERVOJ.format(arm=ARM_SUFFIX[arm])
    prep = []
    if unlock_flag:
        prep.append(f"unlock ({SRV_UNLOCK})")
    if gohome:
        prep.append(f"upperlimb home if off (thresh {home_thresh} rad), "
                    f"then one right-hand OPEN_POSE "
                    f"({SRV_HAND.format(arm='right')})")
    if hand == 'on':
        prep.append(f"per-frame hand replay via "
                    f"{SRV_HAND.format(arm=HAND_SUFFIX[arm])}")
    prep.append("5 s movej alignment to episode first frame")
    summary = (f"Move REAL robot with {space} stream ({n} frames, "
               f"{total_t:.1f}s) via {stream_topic}")
    if space == 'cartesian':
        summary += (f"; fixed 30Hz frame-by-frame EEF, ONLY right_arm_pose; "
                    f"set_servo_params(time={SERVO_TIME}, gain={SERVO_GAIN}) "
                    f"before stream, clear_servo_params after")
    summary += f". Preparatory motions: {', '.join(prep)}; then stream."
    if not skip_confirm:
        if not click.confirm(summary + "?", default=False):
            print("Aborted.")
            return

    if unlock_flag:
        unlock()

    if gohome:
        ensure_home(home_thresh)

    # post-home hand reset: exactly one right-hand OPEN_POSE after a
    # successful --gohome, BEFORE first-frame movej alignment. Independent
    # of --hand on/off and never sent without --gohome. Failure prints a
    # warning (existing hand_to_pose behavior) but continues.
    if gohome:
        hand_to_pose(OPEN_POSE, 'right')
        print("Post-home hand reset: right-hand OPEN_POSE sent.")

    # optional per-frame hand replay proxy (--hand on only)
    hand_call = None
    if hand == 'on' and 'robot_hand' in ep:
        srv = SRV_HAND.format(arm=HAND_SUFFIX[arm])
        rospy.wait_for_service(srv, timeout=10)
        hand_call = rospy.ServiceProxy(srv, HandJoint)
        print(f"Hand: per-frame replay via {srv}")

    # align to episode first frame (slow movej)
    first = ep['robot_joint'][0]
    if first.shape[0] == 7:
        cols = (0, 7)  # right-arm-only format
    else:
        cols = ARM_COLS[arm]
    print("Aligning to episode first frame...")
    movej(first[cols[0]:cols[1]], ARM_SUFFIX[arm], t=5.0,
          arm_type=ARM_TYPES[arm])
    time.sleep(1.0)

    if space == 'joint':
        actual, target = stream_joint(ep, rel_t, arm, 1.0, verify, hand_call)
        if verify:
            report_verify(actual, target)
    else:
        if verify:
            print("(note: --verify only observes joint tracking; it cannot "
                  "assess Cartesian pose accuracy)")
        # official SDK flow: set servo params after alignment, before any
        # stream; clear them in finally whether the stream completes or not.
        # If set fails, streaming never starts.
        set_servo_params()
        try:
            stream_cartesian_right(ep, hand_call)
        finally:
            clear_servo_params()

    print("Done. Robot holds the last frame.")


if __name__ == '__main__':
    main()