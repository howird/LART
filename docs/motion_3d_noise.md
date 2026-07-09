# `motion_3d`: computed but not used, and why

`scripts/prepare_hockey_dataset.py::build_motion_3d_clip` computes a 4-dim
per-frame feature (`signed_speed`, `signed_accel`, `cos(turning_angle)`,
`sin(turning_angle)`) from each clip's player trajectory, and stores it as a
`motion_3d` field in `data/hockey/{train,val,test}.pkl` /
`norm_stats.pkl`. It is intentionally **not** added to
`configs/lart_hockey.yaml`'s `extra_feat.enable`, so it is not currently
consumed by `HockeyPoseDataset` / `HockeyLART_LitModule`. This doc records why.

## What it's supposed to capture

- **Signed speed**: `dot(forward, vel)` — the player's velocity projected onto
  a body-forward direction (derived from shoulder/hip joint positions, see
  `build_motion_3d_clip`/`_normalize_rows`). Positive = moving the way you're
  facing (forward skating), negative = moving opposite (backward skating).
- **Signed acceleration**: frame-to-frame derivative of signed speed.
- **Turning rate**: signed angle between consecutive velocity directions,
  represented as `(cos, sin)` to avoid the wraparound discontinuity of a raw
  angle.

All of it is derived from `pred_cam_t`, the per-frame 3D camera-space
translation of the tracked player, via finite differences (`np.gradient`).

## The problem

Frame-to-frame differences of raw `pred_cam_t` are dominated by noise, not
real motion. Spot-checked an `ACCEL_FORW` clip (should show a consistent
trend) and a `TRANS_FORW_TO_BACK` clip (should show one clear transition):
neither did. `signed_speed` flips sign frame-to-frame with no visible trend,
and the turning angle swings by ~150-170° *repeatedly throughout entire
clips*, not just at real direction changes. Dataset-wide, 30.8% of frames had
`turning_cos < 0` (i.e. an apparent >90° instantaneous turn) before any
mitigation.

**First mitigation tried**: smoothing `pred_cam_t` with a Savitzky-Golay
filter (`window_length=7, polyorder=2`, safely below the shortest clip length
of 15 frames) before differentiating. This measurably helped (median
`turning_cos` 0.987 → 0.9987; fraction with `turning_cos < 0` dropped 30.8% →
10.4%) but did not fully fix it — `signed_speed` still doesn't show a clean
ACCEL/GLID trend on spot-checked clips.

## Root cause (traced through SAM-3D-Body's source)

`pred_cam_t` is not a directly-measured, temporally-consistent 3D position —
it's algebraically reconstructed per frame from a CLIFF-style camera
parameterization. From
`~/sam-3d-body/sam_3d_body/models/heads/camera_head.py::PerspectiveHead.perspective_projection`:

```python
bs = bbox_size * s * default_scale_factor    # s = network-predicted scale (per frame)
tz = 2 * focal_length / bs                    # depth
cx = 2 * (bbox_center[:, 0] - img_size[:, 0] / 2) / bs
cy = 2 * (bbox_center[:, 1] - img_size[:, 1] / 2) / bs
pred_cam_t = [tx + cx, ty + cy, tz]
```

Confirmed this is exactly the source of our `pred_cam_t` field via
`~/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py` (`"pred_cam_t":
out["pred_cam_t"][idx]`, where `out` comes from `camera_project()` in
`sam3d_body.py`, which calls this same `perspective_projection`).

Two candidate noise sources were considered:

1. **Bounding-box jitter** (the crop moving/resizing frame to frame due to an
   imperfect per-frame detector). **Ruled out** — checked our actual data:
   `bbox_size` is *exactly constant per clip* (verified across 350 sampled
   clips, zero variation — this pipeline uses a fixed-size crop per track),
   and `bbox_center` moves smoothly frame to frame (it's a tracked position,
   not raw per-frame detection).

2. **The network's own per-frame-independent scale estimate `s`.** Since
   `bbox_size` is fixed, `tz`'s entire variation must come from `s` — and
   SAM-3D-Body has **no temporal-consistency mechanism**: it's a single-image
   model (`sam3d_body.py::forward_pose_branch`/`forward_decoder` process one
   image at a time, no recurrent or cross-frame fusion module), matching
   `hockey_pipeline`'s own README statement that "SAM-3D-Body is single-image:
   meshes are per-frame, camera-frame (no temporal smoothing, no global world
   trajectory)." Any natural frame-to-frame variation in the network's scale
   estimate (motion blur, subtle articulation/self-occlusion changing
   apparent body extent, compression artifacts, etc.) is real, human-body-scale
   inference noise with no cross-frame averaging to suppress it — **and
   because `tz ∝ 1/s`, that noise is reciprocally amplified in the depth
   channel.**

**Verified empirically**: per-frame jitter relative to each clip's own
dynamic range (`mean(|diff|) / (max - min)`), averaged over 500 clips:

| axis | jitter / range |
|---|---|
| x | 0.056 |
| y | 0.063 |
| **z (depth)** | **0.146** — ~2.5x noisier than x or y |

## Why this is an unlucky combination

The broadcast camera is side-mounted relative to the rink, so skating
up/down the length of the rink — exactly what distinguishes `GLID_FORW` from
`GLID_BACK`, and what `signed_speed` is meant to capture — is primarily a
**depth (Z)** motion from the camera's point of view. That's precisely the
axis with the worst signal-to-noise ratio. A uniform smoothing filter across
all three axes can't fully recover a clean signal when the underlying noise
is concentrated on the one axis carrying most of the real signal.

## Current status

`motion_3d` is computed and stored (for potential future use) but is **not**
wired into `configs/lart_hockey.yaml` (`extra_feat.enable`) or consumed by
`HockeyPoseDataset`/`HockeyLART_LitModule`. This is a data-quality ceiling
from the upstream single-image pose estimator, not a bug in our feature
computation.

## Possible directions if revisited later

- Axis-specific treatment: much heavier smoothing/lower-order polynomial fit
  specifically on the Z channel (accept more temporal blur there since it's
  noisier), or a robust/outlier-rejecting fit instead of a fixed-window filter.
- Exclude Z from `signed_speed` (use only image-plane X/Y motion) — would
  reduce noise but also discard most of the genuine along-rink motion signal,
  likely a net loss for `FORW`/`BACK`-type labels specifically.
- Use a source with actual temporal consistency instead of per-frame
  `pred_cam_t` — e.g. `hockey_pipeline`'s trajectory stage (homography-based
  foot position on the rink template, `hockey_pipeline/hockey_pipeline/trajectory/`)
  which is 2D-only but not per-frame-independent in the same way, if it
  applies any cross-frame fitting/smoothing (not yet checked).
