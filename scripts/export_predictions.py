#!/usr/bin/env python
"""Export per-clip predictions from a trained HockeyLART checkpoint, in the
dict-format pickle degcn-skating-actions' analyze_results.py (--results-format
dict) and sequence_modelling.py (--results) already consume unmodified.

Bypasses the Lightning validation_step/DataLoader path entirely: HockeyPoseDataset's
__getitem__ only returns (input_data, label) with no clip identity, so this script
loops over dataset.clips directly with a plain (non-shuffled) DataLoader and zips
outputs back to clip identity by index -- avoids touching the training-path
Dataset/DataLoader contract for an export-only concern.

Usage: uv run python scripts/export_predictions.py --ckpt <path/to.ckpt> \
    --split test --output /path/to/results.pkl
"""

import argparse
import os
import pickle

import hydra
import numpy as np
import pyrootutils
import torch
import torch.nn.functional as F

pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from lart.datamodules.components.hockey_action_dataset import HockeyPoseDataset  # noqa: E402
from lart.models.lart_hockey import HockeyLART_LitModule  # noqa: E402


def load_model(cfg, ckpt_path, device):
    model = HockeyLART_LitModule(cfg.model.cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # torch.compile guard, matches train.py's own fallback-load logic
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def collate_clips(batch):
    """Minimal manual collate: batch is a list of (input_data_dict, label)."""
    input_data, labels = zip(*batch)
    collated = {
        key: torch.from_numpy(np.stack([d[key] for d in input_data], axis=0))
        for key in input_data[0].keys()
    }
    return collated, torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(
        description="Export per-clip predictions (dict format) for degcn-compatible evaluation/HMM."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a Lightning .ckpt file.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, required=True, help="Output pkl path (dict format).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="Extra Hydra config overrides, e.g. configs.data_dir=/path",
    )
    args = parser.parse_args()

    with hydra.initialize(version_base="1.2", config_path="../configs"):
        # configs.storage_folder interpolates ${hydra:sweep.subdir}, which is only
        # resolvable inside an active @hydra.main job -- override it to a concrete
        # path so HockeyLART_LitModule.__init__'s os.makedirs(...) doesn't crash.
        cfg = hydra.compose(
            config_name="lart_hockey.yaml",
            overrides=["configs.storage_folder=/tmp/lart_export_scratch", *args.overrides],
        )

    device = torch.device(args.device)
    model = load_model(cfg, args.ckpt, device)

    dataset = HockeyPoseDataset(cfg.configs, split=args.split)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_clips,
    )

    results = {}
    idx = 0
    with torch.no_grad():
        for input_data, labels in loader:
            input_data = {k: v.to(device) for k, v in input_data.items()}
            logits, _ = model.forward(input_data, mask_type=cfg.configs.mask_type_test)  # "zero", matches validation_step
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            logits_np = logits.cpu().numpy()
            preds = probs.argmax(axis=-1)
            labels_np = labels.numpy()

            bs = probs.shape[0]
            for i in range(bs):
                clip = dataset.clips[idx + i]
                key = f"{clip['game']}_{clip['start_frame']}_{clip['end_frame']}_{clip['track']}"
                results[key] = {
                    "probs": probs[i].tolist(),
                    "pred": int(preds[i]),
                    "logits": logits_np[i].tolist(),
                    "gt_label": int(labels_np[i]),
                }
            idx += bs

    assert idx == len(dataset.clips), f"processed {idx} clips, expected {len(dataset.clips)} -- iteration order mismatch"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
