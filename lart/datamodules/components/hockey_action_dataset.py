import os
import pickle

import numpy as np
from torch.utils.data import Dataset


class HockeyPoseDataset(Dataset):
    """Single-label-per-clip 3D-pose action dataset for skating_actions_mhr.

    Each clip is a fixed-length (<= cfg.frame_length), single-person track
    built by scripts/prepare_hockey_dataset.py. Unlike PHALP_action_dataset
    (long multi-person tracklets with per-frame AVA/Kinetics labels), every
    clip here has exactly one label, so __getitem__ returns (input_data, label)
    rather than the AVA-oriented (input_data, output_data, meta_data, name)
    tuple.
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.frame_length = cfg.frame_length
        self.max_people = cfg.max_people
        self.use_joints = "joints_3D" in cfg.extra_feat.enable

        data_dir = os.path.join(cfg.data_dir, "hockey")
        with open(os.path.join(data_dir, f"{split}.pkl"), "rb") as f:
            self.clips = pickle.load(f)
        with open(os.path.join(data_dir, "norm_stats.pkl"), "rb") as f:
            stats = pickle.load(f)

        # floor near-zero std (many MHR pose/shape dims are ~constant across
        # this dataset) so those dims don't get divided by ~1e-6 and blow up
        # into huge, almost-identical outlier features that swamp real signal.
        std_floor = 1e-2
        self.pose_shape_mean = stats["pose_shape_mean"].astype(np.float32)
        self.pose_shape_std = np.maximum(stats["pose_shape_std"].astype(np.float32), std_floor)
        self.joints_3d_mean = stats["joints_3d_mean"].astype(np.float32)
        self.joints_3d_std = np.maximum(stats["joints_3d_std"].astype(np.float32), std_floor)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        L = self.frame_length
        t = min(clip["T"], L)

        pose_shape = np.zeros((L, self.max_people, self.cfg.extra_feat.pose_shape.dim), dtype=np.float32)
        has_detection = np.zeros((L, self.max_people, 1), dtype=np.float32)
        mask_detection = np.zeros((L, self.max_people, 1), dtype=np.float32)

        pose_shape[:t, 0, :] = (clip["pose_shape"][:t] - self.pose_shape_mean[None]) / (
            self.pose_shape_std[None] + 1e-6
        )
        has_detection[:t, 0, :] = 1.0

        input_data = {
            "pose_shape": pose_shape,
            "has_detection": has_detection,
            "mask_detection": mask_detection,
        }

        if self.use_joints:
            joints_3d = np.zeros((L, self.max_people, self.cfg.extra_feat.joints_3D.dim), dtype=np.float32)
            joints_3d[:t, 0, :] = (clip["joints_3d"][:t] - self.joints_3d_mean[None]) / (
                self.joints_3d_std[None] + 1e-6
            )
            input_data["joints_3D"] = joints_3d

        return input_data, clip["label"]
