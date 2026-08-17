"""
ZJ Humanoid ROS data collection environment.

Subscribes to robot state / camera / VR topics, aligns all streams to a
common 30Hz clock (using ROS header stamps, nearest-sample-before policy),
records low-dim data to a zarr ReplayBuffer and images to h264 videos.

Design mirrors diffusion_policy/real_world/real_env.py but is read-only:
no motion command is ever sent; the robot is driven by an external
VR teleop service.
"""
from typing import Dict, List, Optional
import threading
import subprocess
import pathlib
import shutil
import time
from collections import deque
import bisect

import numpy as np
import imageio_ffmpeg

import rospy
from scipy.spatial.transform import Rotation as ScipyRot
from sensor_msgs.msg import JointState, Image
from upperlimb.msg import Pose as UPPose
from xr_msgs.msg import Custom as XRCustom
from diffusion_policy.common.replay_buffer import ReplayBuffer

# Supported RGB cameras, in default stable order.
DEFAULT_CAMERA_TOPICS = {
    'up': '/zj_humanoid/sensor/realsense_up/color/image_raw',
    'head': '/zj_humanoid/sensor/realsense_head/color/image_raw',
    'right_wrist': '/zj_humanoid/sensor/right_wrist/image_raw',
}


def select_camera_topics(camera_names=()):
    """Return an ordered {name: topic} mapping for the requested cameras.

    Order follows the explicit `camera_names` sequence; an empty sequence
    selects all supported cameras in DEFAULT_CAMERA_TOPICS order. Unknown
    or duplicated names raise ValueError.
    """
    if len(camera_names) == 0:
        return dict(DEFAULT_CAMERA_TOPICS)
    seen = set()
    selected = {}
    for name in camera_names:
        if name not in DEFAULT_CAMERA_TOPICS:
            raise ValueError(
                f"unsupported camera {name!r}; supported cameras: "
                + ", ".join(DEFAULT_CAMERA_TOPICS))
        if name in seen:
            raise ValueError(f"duplicate camera {name!r}")
        seen.add(name)
        selected[name] = DEFAULT_CAMERA_TOPICS[name]
    return selected

# joint order in /zj_humanoid/upperlimb/joint_states (19 joints)
JOINT_NAMES = ['Shoulder_Y_L', 'Shoulder_X_L', 'Shoulder_Z_L', 'Elbow_L',
               'Wrist_Z_L', 'Wrist_Y_L', 'Wrist_X_L',
               'Shoulder_Y_R', 'Shoulder_X_R', 'Shoulder_Z_R', 'Elbow_R',
               'Wrist_Z_R', 'Wrist_Y_R', 'Wrist_X_R',
               'Neck_Z', 'Neck_Y', 'Waist_Z', 'Waist_Y', 'Lifting_Z']


class TimestampBuffer:
    """Thread-safe buffer of (timestamp, data) with nearest-before lookup."""

    def __init__(self, maxlen=512):
        self._timestamps = deque(maxlen=maxlen)
        self._data = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def put(self, t, data):
        with self._lock:
            self._timestamps.append(float(t))
            self._data.append(data)

    def get_last_before(self, t):
        with self._lock:
            if len(self._timestamps) == 0:
                return None
            idx = bisect.bisect_right(self._timestamps, t) - 1
            if idx < 0:
                return None
            return self._data[idx]

    def get_last(self):
        with self._lock:
            if len(self._data) == 0:
                return None
            return self._data[-1]


class FFmpegVideoWriter:
    """H264 video writer via ffmpeg subprocess (rawvideo pipe)."""

    def __init__(self, fps=30, crf=18, preset='veryfast'):
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.proc = None
        self.resolution = None
        self.pix_fmt = None

    def start(self, path: str, resolution, pix_fmt='rgb24'):
        assert self.proc is None
        self.resolution = resolution
        self.pix_fmt = pix_fmt
        w, h = resolution
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [exe, '-y',
               '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-s', f'{w}x{h}', '-pix_fmt', pix_fmt,
               '-r', str(self.fps), '-i', '-',
               '-c:v', 'libx264', '-preset', self.preset,
               '-crf', str(self.crf), '-pix_fmt', 'yuv420p',
               str(path)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

    def write(self, img: np.ndarray):
        assert self.proc is not None and not self.proc.stdin.closed
        assert img.shape[1] == self.resolution[0] and img.shape[0] == self.resolution[1], \
            f"expected {self.resolution}, got {img.shape[1]}x{img.shape[0]}"
        self.proc.stdin.write(img.tobytes())

    def stop(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait()
        self.proc = None


class ZJRosEnv:
    """Read-only ROS data collection environment for ZJ Humanoid.

    camera_topics: ordered {name: topic} mapping of RGB cameras to record.
        None selects all supported cameras in DEFAULT_CAMERA_TOPICS order.
    """

    def __init__(self,
                 output_dir,
                 frequency=30,
                 camera_topics: Optional[Dict[str, str]] = None,
                 video_crf=18,
                 max_obs_buffer_size=256,
                 verbose=True):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir(), f"parent dir of {output_dir} not found"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.frequency = frequency
        self.dt = 1 / frequency
        self.video_crf = video_crf
        self.verbose = verbose
        if camera_topics is None:
            # copy to avoid sharing the mutable module-level default
            camera_topics = dict(DEFAULT_CAMERA_TOPICS)

        # ---- buffers ----
        self._buf_joint = TimestampBuffer(max_obs_buffer_size)
        self._buf_tcp_l = TimestampBuffer(max_obs_buffer_size)
        self._buf_tcp_r = TimestampBuffer(max_obs_buffer_size)
        self._buf_hand = TimestampBuffer(max_obs_buffer_size)
        self._buf_cam = {k: TimestampBuffer(max_obs_buffer_size)
                         for k in camera_topics}
        self._buf_xr = TimestampBuffer(max_obs_buffer_size * 2)
        self.camera_topics = dict(camera_topics)

        # ---- subscribers ----
        self._subs = []
        self._subs.append(rospy.Subscriber(
            '/zj_humanoid/upperlimb/joint_states', JointState,
            self._cb_joint, queue_size=100))
        self._subs.append(rospy.Subscriber(
            '/zj_humanoid/upperlimb/tcp_pose/left_arm', UPPose,
            self._cb_tcp_l, queue_size=100))
        self._subs.append(rospy.Subscriber(
            '/zj_humanoid/upperlimb/tcp_pose/right_arm', UPPose,
            self._cb_tcp_r, queue_size=100))
        self._subs.append(rospy.Subscriber(
            '/zj_humanoid/hand/joint_states', JointState,
            self._cb_hand, queue_size=100))
        for key, topic in camera_topics.items():
            self._subs.append(rospy.Subscriber(
                topic, Image,
                lambda msg, k=key: self._cb_cam(k, msg), queue_size=10))
        self._subs.append(rospy.Subscriber(
            '/xr_pose', XRCustom, self._cb_xr, queue_size=100))

        # ---- replay buffer ----
        self._replay_buffer = None

        # ---- episode recording state ----
        self._episode_id = 0
        self._video_writers: Dict[str, FFmpegVideoWriter] = {}
        self._ep_buf = None
        self._start_time = None
        self._spin_thread = None

    # ========= callbacks =========
    def _cb_joint(self, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0:
            t = time.time()
        self._buf_joint.put(t, np.asarray(msg.position, dtype=np.float64))

    def _cb_tcp(self, buf, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0:
            t = time.time()
        # NOTE: the rpy_rad/rpy_deg fields of upperlimb/Pose carry garbage
        # from the robot publisher; compute rpy from the reliable quaternion.
        q = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z,
             msg.quaternion.w]
        rpy = ScipyRot.from_quat(q).as_euler('xyz')
        pose = np.array([msg.position.x, msg.position.y, msg.position.z,
                         rpy[0], rpy[1], rpy[2]],
                        dtype=np.float64)
        buf.put(t, pose)

    def _cb_tcp_l(self, msg):
        self._cb_tcp(self._buf_tcp_l, msg)

    def _cb_tcp_r(self, msg):
        self._cb_tcp(self._buf_tcp_r, msg)

    def _cb_hand(self, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0:
            t = time.time()
        # 12 joints: [L6, R6] in topic order
        self._buf_hand.put(t, np.asarray(msg.position, dtype=np.float64))

    def _cb_cam(self, key, msg):
        t = msg.header.stamp.to_sec()
        if t <= 0:
            t = time.time()
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        # msg.data buffer is reused by rospy -> must copy
        self._buf_cam[key].put(t, img.copy())

    def _cb_xr(self, msg):
        t = msg.timestamp_ns * 1e-9
        if t <= 0:
            t = time.time()
        # button vector: [right_primary, left_primary, right_secondary]
        self._buf_xr.put(t, np.array([
            int(msg.right_controller.primary_button),
            int(msg.left_controller.primary_button),
            int(msg.right_controller.secondary_button)], dtype=np.int8))

    # ========= start / stop =========
    def start(self, wait=True):
        def _spin():
            rospy.spin()
        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()
        if wait:
            self.wait_ready(timeout=10.0)

    def stop(self):
        self.end_episode()
        for sub in self._subs:
            sub.unregister()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def wait_ready(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._buf_joint.get_last() is not None:
                return True
            rospy.sleep(0.1)
        raise TimeoutError("robot state topics not publishing")

    @property
    def n_episodes(self):
        if self._replay_buffer is None:
            return 0
        return self._replay_buffer.n_episodes

    def _get_replay_buffer(self):
        if self._replay_buffer is None:
            zarr_path = str(self.output_dir.joinpath(
                'replay_buffer.zarr').absolute())
            self._replay_buffer = ReplayBuffer.create_from_path(
                zarr_path=zarr_path, mode='a')
        return self._replay_buffer

    # ========= sampling (spatio-temporal alignment) =========
    def sample(self, t: float) -> dict:
        """Aligned observation at wall-clock time t (nearest sample before t)."""
        obs = {'timestamp': t}
        joint = self._buf_joint.get_last_before(t)
        if joint is not None:
            # right-arm-only: 19-dim full joint -> right arm cols 7:14 (7 dims)
            obs['robot_joint'] = joint[7:14]
        tcp_r = self._buf_tcp_r.get_last_before(t)
        if tcp_r is not None:
            # right-arm-only: 6-dim right-arm TCP pose (no left concatenation)
            obs['robot_eef_pose'] = tcp_r
        hand = self._buf_hand.get_last_before(t)
        if hand is not None:
            # right-hand-only: 12-dim [L6 R6] -> right hand cols 6:12 (6 dims)
            obs['robot_hand'] = hand[6:12]
        for key in self._buf_cam:
            img = self._buf_cam[key].get_last_before(t)
            if img is not None:
                obs[f'camera_{key}'] = img
        return obs

    # ========= episode recording =========
    def start_episode(self, start_time=None, camera_wait_timeout=3.0):
        if self._ep_buf is not None:
            self.end_episode()
        if start_time is None:
            start_time = time.time()
        self._start_time = start_time
        self._ep_buf = {
            'timestamp': [], 'action': [], 'robot_joint': [],
            'robot_eef_pose': [], 'robot_hand': [], 'stage': [],
        }
        episode_id = self.n_episodes
        self._episode_id = episode_id
        video_dir = self.output_dir.joinpath('videos', str(episode_id))
        video_dir.mkdir(parents=True, exist_ok=True)
        for i, key in enumerate(self._buf_cam):
            res = self._wait_cam_res(key, timeout=camera_wait_timeout)
            if res is None:
                if self.verbose:
                    print(f"    [warn] camera {key}: no frames, video skipped")
                continue
            writer = FFmpegVideoWriter(fps=self.frequency, crf=self.video_crf)
            writer.start(str(video_dir.joinpath(f'{i}.mp4')), res)
            self._video_writers[key] = writer
        if self.verbose:
            print(f"Episode {episode_id} started! (video dir: {video_dir})")

    def _wait_cam_res(self, key, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            img = self._buf_cam[key].get_last()
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
            time.sleep(0.05)
        return None

    def record_sample(self, obs: dict, action: np.ndarray, stage=0):
        """Must be called inside an episode."""
        assert self._ep_buf is not None, "call start_episode() first"
        self._ep_buf['timestamp'].append(obs['timestamp'])
        self._ep_buf['action'].append(np.asarray(action, dtype=np.float64))
        self._ep_buf['robot_joint'].append(obs['robot_joint'])
        self._ep_buf['robot_eef_pose'].append(obs['robot_eef_pose'])
        self._ep_buf['robot_hand'].append(obs['robot_hand'])
        self._ep_buf['stage'].append(stage)
        for key, writer in self._video_writers.items():
            img = obs.get(f'camera_{key}')
            if img is not None:
                writer.write(img)

    def end_episode(self):
        if self._ep_buf is None:
            return
        for writer in self._video_writers.values():
            writer.stop()
        self._video_writers = {}

        ep = dict()
        for k, v in self._ep_buf.items():
            ep[k] = np.array(v)
        if len(ep['timestamp']) > 0:
            replay_buffer = self._get_replay_buffer()
            replay_buffer.add_episode(ep, compressors='disk')
            episode_id = self._episode_id
            if self.verbose:
                print(f"Episode {episode_id} saved: {len(ep['timestamp'])} steps, "
                      f"duration {ep['timestamp'][-1]-ep['timestamp'][0]:.2f}s")
        self._ep_buf = None
        self._start_time = None

    def discard_episode(self):
        """Abort the current episode: stop video writers, delete the episode's
        video dir, and drop all buffered samples WITHOUT writing to the
        ReplayBuffer. n_episodes is unchanged."""
        if self._ep_buf is None:
            return
        for writer in self._video_writers.values():
            writer.stop()
        self._video_writers = {}

        video_dir = self.output_dir.joinpath('videos', str(self._episode_id))
        if video_dir.is_dir():
            shutil.rmtree(video_dir)

        self._ep_buf = None
        self._start_time = None
        if self.verbose:
            print(f"Episode {self._episode_id} discarded")
