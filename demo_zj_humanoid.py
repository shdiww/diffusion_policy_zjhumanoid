"""
ZJ Humanoid data collection (read-only).

Robot motion is driven by an external VR teleop ROS service.
This script only reads robot state / cameras / VR buttons, aligns
everything to a common clock and records to disk.

Usage (inside naviai_learn_gpu):
    source /home/naviai/diffusion_ws/scripts/zj_collect_env.sh
    python diffusion_policy/demo_zj_humanoid.py -o /datasets/zj_episodes
    # record only the chest camera
    python diffusion_policy/demo_zj_humanoid.py -o /datasets/zj_episodes --camera up

Cameras (default: all three in order up, head, right_wrist):
    --camera up           chest camera only
    --camera up --camera head   chest + head, in that order

Episode control:
    VR:  right controller primary button   -> start episode
         left  controller primary button   -> end episode (save)
         right controller secondary button -> discard current episode (no save)
    (fallback keyboard when --keyboard or no xr data):
         c = start, s = end, d = discard current (no save), q = quit
"""
import time
import sys
import pathlib
import click
import numpy as np

import rospy
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world_zj.zj_ros_env import (
    ZJRosEnv, select_camera_topics, DEFAULT_CAMERA_TOPICS)

ACTION_DIM = 12  # [right_eef_pose(6), right_hand(6)]


def make_tty_raw():
    import tty
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old


def restore_tty(old):
    import termios
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def kb_hit():
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


@click.command()
@click.option('--output', '-o', default='/home/naviai/diffusion_ws/data/zj_episodes',
              help='Directory to save demonstration dataset (on host).')
@click.option('--frequency', '-f', default=30.0, type=float,
              help='Record frequency in Hz (obs + video).')
@click.option('--max_duration', '-md', default=300.0, type=float,
              help='Max duration per episode in seconds.')
@click.option('--video_crf', default=18, type=int,
              help='Video recording quality (lower is better).')
@click.option('--no-vis', is_flag=True, default=False,
              help='Disable opencv visualization window.')
@click.option('--camera', 'camera_names', multiple=True,
              type=click.Choice(list(DEFAULT_CAMERA_TOPICS)),
              help='Camera(s) to record; repeatable, e.g. --camera up --camera head. '
                   'Default: all three in order (up, head, right_wrist).')
def main(output, frequency, max_duration, video_crf, no_vis, camera_names):
    dt = 1 / frequency

    try:
        camera_topics = select_camera_topics(camera_names)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint='--camera')

    print("Selected cameras (video index -> camera):")
    for i, name in enumerate(camera_topics):
        print(f"  {i}.mp4 <- {name} ({camera_topics[name]})")

    rospy.init_node('demo_zj_humanoid')

    use_vis = (not no_vis)
    if use_vis:
        import cv2
        cv2.setNumThreads(1)
    else:
        cv2 = None

    old_term = make_tty_raw()
    try:
        with ZJRosEnv(
                output_dir=output,
                frequency=frequency,
                camera_topics=camera_topics,
                video_crf=video_crf) as env:
            time.sleep(1.0)

            # warm up buffers
            t0 = time.time()
            obs = env.sample(t0)
            while 'robot_joint' not in obs and time.time() - t0 < 10:
                time.sleep(0.1)
                obs = env.sample(time.time())
            if 'robot_joint' not in obs:
                raise RuntimeError(
                    "No robot data! Check ROS master and robot node.")
            print(f"Ready! Recording at {frequency}Hz -> {env.output_dir}")
            print("VR: right-primary=start, left-primary=end/save, "
                  "right-secondary=discard | "
                  "keyboard: c=start s=end d=discard q=quit")

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            episode_t_start = None

            prev_xr = None

            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end  # sample aligned to this tick

                # ---- keyboard input (raw tty) ----
                key = kb_hit()
                if key == 'q':
                    stop = True
                elif key == 'c' and not is_recording:
                    env.start_episode()
                    episode_t_start = time.time()
                    is_recording = True
                    print("Recording!")
                elif key == 's' and is_recording:
                    env.end_episode()
                    is_recording = False
                    print("Stopped.")
                elif key == 'd' and is_recording:
                    env.discard_episode()
                    is_recording = False
                    print("Episode discarded.")

                # ---- VR button edge detection ----
                # button vector: [right_primary, left_primary, right_secondary]
                #   right_primary   rising edge -> start episode
                #   left_primary    rising edge -> end episode (save)
                #   right_secondary rising edge -> discard current episode
                #     (takes precedence over left_primary when both rise in one tick)
                xr = env._buf_xr.get_last()
                if xr is not None:
                    if prev_xr is None:
                        # first XR sample: baseline only, no start/end/discard
                        prev_xr = xr.copy()
                    else:
                        was_recording = is_recording
                        if not was_recording and xr[0] > prev_xr[0]:
                            env.start_episode()
                            episode_t_start = time.time()
                            is_recording = True
                            print("Recording! (VR)")
                        elif was_recording and xr[2] > prev_xr[2]:
                            env.discard_episode()
                            is_recording = False
                            print("Episode discarded. (VR)")
                        elif was_recording and xr[1] > prev_xr[1]:
                            env.end_episode()
                            is_recording = False
                            print("Stopped. (VR)")
                        prev_xr = xr.copy()

                # ---- sample aligned obs ----
                obs = env.sample(time.time())

                # ---- record ----
                if is_recording:
                    if 'robot_eef_pose' in obs and 'robot_hand' in obs:
                        action = np.concatenate(
                            [obs['robot_eef_pose'][:6],
                             obs['robot_hand'][:6]])
                        env.record_sample(obs, action)
                    if time.time() - episode_t_start > max_duration:
                        env.end_episode()
                        is_recording = False
                        print("Terminated by max duration.")

                # ---- visualize ----
                if use_vis and 'camera_up' in obs:
                    vis_img = obs['camera_up'].copy()
                    episode_id = env.n_episodes
                    text = f"Episode: {episode_id}"
                    if is_recording:
                        text += " REC"
                    cv2.putText(vis_img, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('up', vis_img)
                    cv2.pollKey()

                precise_wait(t_cycle_end)
                iter_idx += 1
    finally:
        restore_tty(old_term)

    print("Exited.")


if __name__ == '__main__':
    main()
