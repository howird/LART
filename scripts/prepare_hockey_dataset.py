#!/usr/bin/env python
"""Build LART-style per-clip pose tensors from the skating_actions_mhr dataset.

Reads the per-game MHR pkls in /data_ssd_2/shared/skating_actions_mhr, keeps
only clips with MHR pose data and drops RAPID_DECELERATION (0% MHR coverage
for that class), maps the remaining 11 classes to the same label indices
used by the prior DeGCN work, and assigns train/val/test by reusing that
work's canonical per-game split. Writes data/hockey/{train,val,test}.pkl plus
data/hockey/norm_stats.pkl (train-split mean/std for pose_shape and
joints_3D, matching PHALP_action_dataset's normalization convention).

Usage: uv run python scripts/prepare_hockey_dataset.py
"""

import glob
import os
import pickle

import numpy as np

MHR_DIR = "/data_ssd_2/shared/skating_actions_mhr"
SPLIT_SOURCE = "/data/ecresearch/skating_actions_dataset/annotations/combined_annotations_with_motion_v2.pkl"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hockey")

# Fixed 11-class vocabulary, index-matched to the prior DeGCN work
# (degcn-skating-actions/analyze_results.py ACTION_LABELS), verified by
# cross-checking per-class counts against combined_annotations_with_motion_v2.pkl.
ACTION_LABELS = [
    "GLID_FORW",
    "ACCEL_FORW",
    "GLID_BACK",
    "ACCEL_BACK",
    "TRANS_FORW_TO_BACK",
    "TRANS_BACK_TO_FORW",
    "POST_WHISTLE_GLIDING",
    "FACEOFF_BODY_POSITION",
    "MAINTAIN_POSITION",
    "PRONE",
    "ON_A_KNEE",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(ACTION_LABELS)}
EXCLUDED_LABEL = "RAPID_DECELERATION"


def load_split_map():
    """frame_dir -> 'train'/'val'/'test', from the reused DeGCN split."""
    with open(SPLIT_SOURCE, "rb") as f:
        d = pickle.load(f)
    split_map = {}
    for split_name, frame_dirs in d["split"].items():
        for fd in frame_dirs:
            split_map[fd] = split_name
    return split_map


def build_pose_shape(frame_mhr):
    # Per MHR's own docs: full-body pose = 204 params (`mhr_model_params` --
    # the model's canonical pose vector, not the same as concatenating
    # global_rot+body_pose_params, which is a different/partial decomposition;
    # verified these diverge elementwise). Shape/identity (`shape_params`) is
    # dropped -- body shape doesn't help distinguish actions.
    return frame_mhr["mhr_model_params"].astype(np.float32)


# mhr70 raw index order (sam_3d_body/metadata/mhr70.py's original_keypoint_info)
# -- NOT standard COCO order (wrists/hips are swapped relative to COCO-17).
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_HIP, KP_RIGHT_HIP = 9, 10

EPS = 1e-8


def _normalize_rows(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def rotmats_to_6d(rotmats):
    """(J, 3, 3) rotation matrices -> (J*6,) flattened 6D continuous rotation
    features (first two columns of each matrix), matching the convention
    phalp's SMPLHead.rot6d_to_rotmat expects as input."""
    cols01 = rotmats[:, :, :2]  # (J, 3, 2): columns a1, a2
    return cols01.transpose(0, 2, 1).reshape(-1).astype(np.float32)  # (J*6,)


def build_joints_3d_clip(mhr_list):
    """Per-clip (T frames) joints_3D features: 127-joint 6D rotations, 762 dims/frame."""
    rotations = np.stack([rotmats_to_6d(m["pred_global_rots"]) for m in mhr_list], axis=0)  # (T, 762)
    return rotations.astype(np.float32)


# SAM-3D-Body (which produces pred_cam_t / pred_keypoints_3d) is single-image:
# per-frame, camera-frame, "no temporal smoothing, no global world trajectory"
# (per hockey_pipeline's own README). A raw frame-to-frame difference of an
# independently-per-frame-estimated quantity is mostly noise -- verified this
# empirically (spot-checked an ACCEL_FORW and a TRANS_FORW_TO_BACK clip: no
# clean trend/transition, turning angle swinging ~170 degrees repeatedly
# through the whole clip, not just at the real transition). So cam_t is
# smoothed with a Savitzky-Golay filter before differentiating. Window=7 is
# safely below the shortest clip length (15 frames after the dataset's own
# 15-frame minimum-length filter).
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2


def build_motion_3d_clip(mhr_list):
    """Per-clip (T frames) motion features from the player's smoothed trajectory:
    signed_speed, signed_accel, cos(turning_angle), sin(turning_angle) -- 4 dims/frame.
    NOT currently wired into the model (extra_feat.enable) -- stored for later use
    once validated further; see plan doc."""
    from scipy.signal import savgol_filter

    T = len(mhr_list)
    cam_t = np.stack([m["pred_cam_t"].astype(np.float32) for m in mhr_list], axis=0)  # (T, 3)
    kp3d = np.stack([m["pred_keypoints_3d"].astype(np.float32) for m in mhr_list], axis=0)  # (T, 70, 3)

    cam_t_smooth = savgol_filter(cam_t, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER, axis=0)

    # velocity via central differences (np.gradient == the described
    # forward/backward-at-edges, average-of-adjacent-diffs-in-the-middle scheme)
    vel = np.gradient(cam_t_smooth, axis=0)  # (T, 3)
    speed = np.linalg.norm(vel, axis=1)  # (T,)
    vel_dir = vel / np.maximum(speed[:, None], EPS)

    # body-forward direction per frame, from joint positions (global_rot did
    # not validate as a usable facing-direction reference -- see plan doc)
    l_sh, r_sh = kp3d[:, KP_LEFT_SHOULDER, :], kp3d[:, KP_RIGHT_SHOULDER, :]
    l_hip, r_hip = kp3d[:, KP_LEFT_HIP, :], kp3d[:, KP_RIGHT_HIP, :]
    up = (l_sh + r_sh) / 2 - (l_hip + r_hip) / 2  # (T, 3) torso spine direction
    across = r_sh - l_sh  # (T, 3) left->right shoulder axis
    forward = _normalize_rows(np.cross(up, across))  # (T, 3)

    signed_speed = np.sum(forward * vel, axis=1)  # (T,)
    signed_accel = np.gradient(signed_speed)  # (T,)

    # turning rate of travel direction: signed angle between consecutive
    # velocity directions, using each frame's own torso `up` as the axis
    # reference to determine sign (avoids assuming a fixed world/camera axis)
    speed_ok = speed > (np.median(speed) * 0.1 + EPS)
    turn_angle = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        if speed_ok[t] and speed_ok[t - 1]:
            cross_v = np.cross(vel_dir[t - 1], vel_dir[t])
            sin_comp = np.dot(cross_v, up[t])
            cos_comp = np.dot(vel_dir[t - 1], vel_dir[t])
            turn_angle[t] = np.arctan2(sin_comp, cos_comp)
    turning_cos = np.cos(turn_angle)
    turning_sin = np.sin(turn_angle)

    return np.stack([signed_speed, signed_accel, turning_cos, turning_sin], axis=1).astype(np.float32)  # (T, 4)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    split_map = load_split_map()

    clips = {"train": [], "val": [], "test": []}
    skipped_no_mhr = 0
    skipped_excluded_label = 0
    skipped_no_split = 0

    game_files = sorted(glob.glob(os.path.join(MHR_DIR, "*.pkl")))
    for game_path in game_files:
        game_name = os.path.basename(game_path)[:-4]
        print(f"Processing {game_name} ...")
        with open(game_path, "rb") as f:
            game_data = pickle.load(f)

        for (track, start_frame, end_frame), rec in game_data.items():
            if rec["action"] == EXCLUDED_LABEL:
                skipped_excluded_label += 1
                continue
            if not rec["mhr_available"]:
                skipped_no_mhr += 1
                continue

            frame_dir = f"{game_name}_{start_frame}_{end_frame}_{track}"
            split = split_map.get(frame_dir)
            if split is None:
                skipped_no_split += 1
                continue

            T = len(rec["frames"])
            pose_shape = np.stack([build_pose_shape(m) for m in rec["mhr"]], axis=0)
            joints_3d = build_joints_3d_clip(rec["mhr"])
            motion_3d = build_motion_3d_clip(rec["mhr"])

            clips[split].append(
                {
                    "game": game_name,
                    "track": track,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "T": T,
                    "pose_shape": pose_shape.astype(np.float32),  # (T, 204)
                    "joints_3d": joints_3d.astype(np.float32),  # (T, 762)
                    "motion_3d": motion_3d.astype(np.float32),  # (T, 4), not yet wired into the model
                    "label": LABEL_TO_IDX[rec["action"]],
                }
            )

    print()
    print(f"Skipped (excluded label): {skipped_excluded_label}")
    print(f"Skipped (no MHR):         {skipped_no_mhr}")
    print(f"Skipped (not in split):   {skipped_no_split}")
    for split in ("train", "val", "test"):
        print(f"{split}: {len(clips[split])} clips")

    # normalization stats from train split only
    train_pose = np.concatenate([c["pose_shape"] for c in clips["train"]], axis=0)
    train_joints = np.concatenate([c["joints_3d"] for c in clips["train"]], axis=0)
    train_motion = np.concatenate([c["motion_3d"] for c in clips["train"]], axis=0)
    norm_stats = {
        "pose_shape_mean": train_pose.mean(axis=0),
        "pose_shape_std": train_pose.std(axis=0),
        "joints_3d_mean": train_joints.mean(axis=0),
        "joints_3d_std": train_joints.std(axis=0),
        "motion_3d_mean": train_motion.mean(axis=0),
        "motion_3d_std": train_motion.std(axis=0),
    }

    for split in ("train", "val", "test"):
        out_path = os.path.join(OUT_DIR, f"{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(clips[split], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {out_path}")

    stats_path = os.path.join(OUT_DIR, "norm_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(norm_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
