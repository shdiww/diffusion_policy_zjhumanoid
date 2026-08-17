"""
ZJ Humanoid home pose constants and utilities.

Home sequence joint order (user-defined):
  [Lifting_Z, Waist_Z, Waist_Y, L7(7), R7(7), Neck_Z, Neck_Y]   -- 19 dims

Reordered to /zj_humanoid/upperlimb/joint_states order:
  [L7(0-6), R7(7-13), Neck_Z(14), Neck_Y(15), Waist_Z(16), Waist_Y(17), Lifting_Z(18)]
"""
import numpy as np

zjhome = [
    # Lifting_Z, Waist_Z, Waist_Y
    0.13,
    0.0,
    0.4,

    # L7
    0.1649149235117875, 
    0.5537071587104947, 
    0.37711959981606924, 
    -0.6024121885825707, 
    0.33912960580892104, 
    0.10591557408752754, 
    -0.0386528047218433,

    # R7
    0.11728961241260549, 
    -0.9969077764935719, 
    -0.6082233993220143, 
    -1.227907912451201, 
    0.3008040534950851, 
    -0.006749076628446094, 
    0.06263009504884917,

    # Neck_Z, Neck_Y
    0.0,
    0.35,
]


HOME_SEQUENCE = np.array(zjhome, dtype=np.float64)

# open hand pose (6 dims per hand)OPEN_POSE = [-0.5, 1.3 , 0.0, 0.0, 0.0, 0.0]
OPEN_POSE = np.array([-0.5, 1.3 , 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
# both hands (12 dims): [L6, R6]
OPEN_POSE_DUAL = np.concatenate([OPEN_POSE, OPEN_POSE])

# joint order constants (indices in /zj_humanoid/upperlimb/joint_states)
JOINT_NAMES = ['Shoulder_Y_L', 'Shoulder_X_L', 'Shoulder_Z_L', 'Elbow_L',
               'Wrist_Z_L', 'Wrist_Y_L', 'Wrist_X_L',
               'Shoulder_Y_R', 'Shoulder_X_R', 'Shoulder_Z_R', 'Elbow_R',
               'Wrist_Z_R', 'Wrist_Y_R', 'Wrist_X_R',
               'Neck_Z', 'Neck_Y', 'Waist_Z', 'Waist_Y', 'Lifting_Z']


def to_joint_states_order(home=HOME_SEQUENCE):
    """Reorder the user home sequence into joint_states order."""
    home = np.asarray(home, dtype=np.float64)
    assert home.shape == (19,), f"home must be 19 dims, got {home.shape}"
    lift = home[0:1]
    waist = home[1:3]
    left = home[3:10]
    right = home[10:17]
    neck = home[17:19]
    return np.concatenate([left, right, neck, waist, lift])


HOME_JS = to_joint_states_order()


def home_error(joint_states):
    """Per-joint abs error and max error vs HOME_JS (joint_states order)."""
    js = np.asarray(joint_states, dtype=np.float64)
    assert js.shape == (19,), f"joint_states must be 19 dims, got {js.shape}"
    per = np.abs(js - HOME_JS)
    return float(np.max(per)), per


def is_home(joint_states, thresh=0.1):
    """Return (bool, max_error) whether robot is near home."""
    err, _ = home_error(joint_states)
    return err <= thresh, err
