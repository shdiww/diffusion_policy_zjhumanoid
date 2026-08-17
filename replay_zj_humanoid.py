"""
Replay recorded zj_humanoid episodes (zarr + h264 videos).

Shows up/wrist cameras + end-effector 6D trajectory, one frame per
zarr row (videos were recorded at the same 30Hz grid).

Usage:
    source /home/naviai/diffusion_ws/scripts/zj_collect_env.sh
    python diffusion_policy/replay_zj_humanoid.py -i /datasets/zj_episodes

Keys:
    space = pause/resume,  left/right = prev/next episode,
    q = quit
"""
import pathlib
import time
import click
import numpy as np

import cv2
from diffusion_policy.common.replay_buffer import ReplayBuffer

DEFAULT_CAMERA_KEYS = ['up', 'right_wrist']


def draw_traj_canvas(poses: np.ndarray, canvas_w=720, canvas_h=320):
    """poses: (N,6) right-arm only (new format) or (N,12) [L6 R6] legacy.
    Draws xyz of the right arm over time; left arm too when legacy 12-dim."""
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)
    if len(poses) < 2:
        return canvas
    t = np.arange(len(poses))
    colors = [(255, 80, 80), (80, 255, 80), (80, 160, 255)]  # x y z
    cols = poses.shape[1] if poses.ndim > 1 else 0
    if cols >= 12:
        arms, labels = 2, ['Lx', 'Ly', 'Lz', 'Rx', 'Ry', 'Rz']
    elif cols >= 6:
        arms, labels = 1, ['Rx', 'Ry', 'Rz']
    else:
        return canvas
    for arm in range(arms):
        base = arm * 6
        for dim in range(3):
            y = poses[:, base + dim]
            lo, hi = y.min(), y.max()
            span = (hi - lo) or 1.0
            xs = 10 + t / max(len(t) - 1, 1) * (canvas_w - 20)
            ys = canvas_h - 20 - (y - lo) / span * (canvas_h - 40)
            color = colors[dim] if arm == 0 else (
                (colors[dim][0] // 2, colors[dim][1] // 2,
                 colors[dim][2] // 2))
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, color, 1)
            cv2.putText(canvas, labels[dim + arm * 3],
                        (10 + dim * 120 + arm * 360, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return canvas


class ZJReplayLoader:
    def __init__(self, input_dir):
        input_dir = pathlib.Path(input_dir)
        self.dir = input_dir
        self.replay_buffer = ReplayBuffer.create_from_path(
            str(input_dir.joinpath('replay_buffer.zarr').absolute()), mode='r')
        self.video_dir = input_dir.joinpath('videos')
        self.n_episodes = self.replay_buffer.n_episodes

    def get_episode_data(self, idx):
        ep = self.replay_buffer.get_episode(idx)
        return ep

    def open_video(self, idx, cam_idx=0):
        path = self.video_dir.joinpath(str(idx), f'{cam_idx}.mp4')
        if not path.exists():
            return None
        return cv2.VideoCapture(str(path))


@click.command()
@click.option('--input', '-i', required=True, help='Dataset directory '
              '(containing replay_buffer.zarr and videos/)')
@click.option('--episode', '-e', default=0, type=int,
              help='Start at episode index.')
@click.option('--save-video', '-sv', default=None,
              help='Optional: write composited replay to a .mp4 file.')
@click.option('--fps', '-f', default=30.0, type=float,
              help='Playback speed (fps).')
@click.option('--once', is_flag=True, default=False,
              help='Play all episodes once then exit.')
def main(input, episode, save_video, fps, once):
    loader = ZJReplayLoader(input)
    if loader.n_episodes == 0:
        raise RuntimeError("no episodes in dataset")
    print(f"Dataset: {loader.n_episodes} episodes, keys="
          f"{list(loader.get_episode_data(0).keys())}")

    ep_idx = episode
    pause = False
    save_writer = None
    if save_video is not None:
        save_writer = cv2.VideoWriter(
            save_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 1040))
    ep0 = loader.get_episode_data(0)
    has_hand = 'robot_hand' in ep0
    if not has_hand:
        print("(note: dataset has no robot_hand, legacy format)")

    try:
        while True:
            ep = loader.get_episode_data(ep_idx)
            n = len(ep['timestamp'])
            t_start = ep['timestamp'][0]
            print(f"--- Episode {ep_idx}: {n} steps, "
                  f"duration {ep['timestamp'][-1] - t_start:.2f}s ---")

            cap0 = loader.open_video(ep_idx, 0)
            cap1 = loader.open_video(ep_idx, 1)
            print(f"    videos: up={cap0 is not None}, wrist={cap1 is not None}")

            frame_idx = 0
            t0 = time.monotonic()
            while frame_idx < n:
                # frame pacing
                target_t = t0 + frame_idx / fps
                delay = target_t - time.monotonic()
                if delay > 0 and not pause:
                    time.sleep(delay)

                # read video frames (if available)
                img0 = None
                if cap0 is not None:
                    ok, img0 = cap0.read()
                    if not ok:
                        break
                img1 = None
                if cap1 is not None:
                    ok, img1 = cap1.read()
                    if not ok:
                        break

                # compose view
                if img0 is not None:
                    disp = img0.copy()
                    if img1 is not None:
                        h1 = disp.shape[0] // 2
                        img1r = cv2.resize(img1, (disp.shape[1], h1))
                        disp[0:h1, :] = img1r
                        cv2.rectangle(disp, (0, 0),
                                      (disp.shape[1] - 1, h1 - 1),
                                      (0, 255, 0), 1)
                else:
                    disp = np.zeros((720, 1280, 3), np.uint8)

                # trajectory canvas
                poses = ep['robot_eef_pose']
                traj = draw_traj_canvas(poses[:frame_idx + 1],
                                        canvas_w=disp.shape[1])
                disp = cv2.resize(disp, (1280, 720))
                view = np.vstack([disp, traj])

                # overlay info
                ts = ep['timestamp'][frame_idx]
                pose = poses[frame_idx]
                cv2.putText(view, f"Ep {ep_idx} t={ts - t_start:5.2f}s "
                            f"({frame_idx}/{n})", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if pose.shape[0] >= 12:  # legacy dual-arm data
                    cv2.putText(view, "L:" + np.round(pose[:6], 3).__repr__(),
                                (10, 735), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    cv2.putText(view, "R:" + np.round(pose[6:], 3).__repr__(),
                                (10, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                else:  # right-arm-only data
                    cv2.putText(view, "R:" + np.round(pose[:6], 3).__repr__(),
                                (10, 735), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                if has_hand:
                    hand = ep['robot_hand'][frame_idx]
                    if hand.shape[0] >= 12:  # legacy dual-hand data
                        cv2.putText(view,
                                    "HL:" + np.round(hand[:6], 3).__repr__(),
                                    (10, 805), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (100, 200, 255), 1)
                        cv2.putText(view,
                                    "HR:" + np.round(hand[6:], 3).__repr__(),
                                    (10, 840), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (100, 200, 255), 1)
                    else:  # right-hand-only data
                        cv2.putText(view,
                                    "HR:" + np.round(hand[:6], 3).__repr__(),
                                    (10, 805), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (100, 200, 255), 1)

                cv2.imshow('zj_humanoid replay', view)
                if save_writer is not None:
                    save_writer.write(view)

                key = cv2.pollKey()
                if key == ord('q'):
                    if save_writer is not None:
                        save_writer.release()
                    return
                elif key == ord(' '):
                    pause = not pause
                    print('paused' if pause else 'resumed')
                elif key == 81:  # left arrow
                    break
                elif key == 83:  # right arrow
                    break

                frame_idx += 1

            if cap0 is not None:
                cap0.release()
            if cap1 is not None:
                cap1.release()

            key = cv2.waitKey(1)
            if key == 81 or key == ord('a'):
                ep_idx = (ep_idx - 1) % loader.n_episodes
            elif key == 83 or key == ord('d'):
                ep_idx = (ep_idx + 1) % loader.n_episodes
            elif once:
                if ep_idx + 1 >= loader.n_episodes:
                    if save_writer is not None:
                        save_writer.release()
                    return
                ep_idx += 1
            else:
                # loop current episode unless arrow pressed within 2s
                k2 = cv2.waitKey(2000) & 0xFF
                if k2 == ord('q'):
                    if save_writer is not None:
                        save_writer.release()
                    return
                elif k2 == 81:
                    ep_idx = (ep_idx - 1) % loader.n_episodes
                elif k2 == 83:
                    ep_idx = (ep_idx + 1) % loader.n_episodes
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
