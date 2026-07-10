"""First-order HMM / Viterbi temporal decoding over per-clip action predictions.

Ported (near-verbatim) from degcn-skating-actions/sequence_modelling.py, adapted
to be self-contained within this repo (no cross-repo imports) and to default to
this repo's own data paths.

Sequence annotations are the same file degcn-skating-actions' sequence_dataset_builder.py
produces:
  - pkl with keys "train", "val", "test"; each value is a list of sequence dicts.
  - Each sequence has "actions": [[annotation_id, start_frame, end_frame, label], ...].
  - A transition is one action right after the other: (actions[i][3], actions[i+1][3]) = (label_from, label_to).

Transitions are checked against ALLOWED_TRANSITIONS: 1 = allowed, 0 = implausible.

Verified: lart's own data/hockey/{train,val,test}.pkl clips are built from the exact
same underlying sequence data as DEFAULT_SEQUENCE_ANNOTATIONS_PATH below (matching
per-split action counts and identical first-clip game/tracklet/frame metadata), so
this external file can be referenced directly without rebuilding anything locally.
"""

import argparse
import csv
import json
import os
import pickle

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read-only external source of ground-truth action sequences (unmodified, not copied
# into this repo -- same precedent as scripts/prepare_hockey_dataset.py's SPLIT_SOURCE).
DEFAULT_SEQUENCE_ANNOTATIONS_PATH = "/data_hdd/maria/skating_actions_dataset/annotations/sequence_annotations.pkl"

# Derived artifact (cheap to regenerate): lives inside the repo alongside norm_stats.pkl.
DEFAULT_TRANSITION_PROBS_CSV = os.path.join(_REPO_ROOT, "data", "hockey", "transition_probs.csv")

# Weight for transition probability in Viterbi (0 = emissions only, 1 = full transition contribution).
TRANSITION_WEIGHT = 0.1

# Label order (index -> name). 11 classes -- duplicated from (must stay index-identical
# to) scripts/prepare_hockey_dataset.py's ACTION_LABELS / LABEL_TO_IDX.
ACTION_LABELS = [
    "GLID_FORW",      # 0  GF
    "ACCEL_FORW",     # 1  AF
    "GLID_BACK",      # 2  GB
    "ACCEL_BACK",     # 3  AB
    "TRANS_FORW_TO_BACK",   # 4  TFB
    "TRANS_BACK_TO_FORW",   # 5  TBF
    "POST_WHISTLE_GLIDING", # 6  PWG
    "FACEOFF_BODY_POSITION",# 7  FO
    "MAINTAIN_POSITION",    # 8  MP
    "PRONE",          # 9  P
    "ON_A_KNEE",      # 10 OKn
]

# ALLOWED_TRANSITIONS[to_label][from_label]: 1 = allowed, 0 = implausible.
# Rows = to (GF, AF, GB, ...), Cols = from (same order); transposed at runtime via
# _allowed_from_to() so that _ALLOWED_FROM_TO[from][to] is correct.
ALLOWED_TRANSITIONS = [
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],   # to GF (TBF->GF allowed; GF->TBF stays 0 in row to TBF)
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],   # to AF (TBF->AF allowed; AF->TBF stays 0 in row to TBF)
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # to GB (TFB->GB allowed; GB->TFB stays 0 in row to TFB)
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # to AB (TFB->AB allowed; AB->TFB stays 0 in row "to TFB")
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],   # to TFB (TFB->TFB allowed)
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # to TBF (TBF->TBF allowed)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # to PWG
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # to FO
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # to MP
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # to P
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # to OKn
]


def _allowed_from_to():
    """Return 11x11 matrix M so that M[from][to] = 1 if transition from->to is allowed."""
    m = ALLOWED_TRANSITIONS
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


_ALLOWED_FROM_TO = _allowed_from_to()
NUM_CLASSES = len(ACTION_LABELS)


# Smoothing for transition prob estimation: add alpha to each allowed count, 11*alpha to denominator
TRANSITION_SMOOTHING_ALPHA = 0.1


def estimate_transition_probs_from_train(sequence_data=None, allowed_matrix=None, path=None, alpha=None, use_constraints=True):
    """
    Use the train split to estimate P(to | from) for each transition.
    - When use_constraints=True: implausible transitions (0 in the allowed matrix) remain 0.
      Allowed transitions use smoothed counts.
    - When use_constraints=False: no hard constraints; all n*n transitions get smoothed counts,
      so the matrix is all non-zero.
    - Smoothing: (count(from, to) + alpha) / (total_from + n*alpha). Denominator uses n*alpha.
    - alpha: smoothing constant (default TRANSITION_SMOOTHING_ALPHA = 0.1).

    sequence_data: dict with 'train' (list of sequence dicts). If None, loaded from path.
    allowed_matrix: [from][to] 0/1; if None uses _ALLOWED_FROM_TO (ignored when use_constraints=False).
    Returns: 2D list prob_matrix[from][to] of floats (each row sums to 1).
    """
    if sequence_data is None:
        sequence_data = load_sequence_annotations(path)
    n = len(ACTION_LABELS)
    if use_constraints:
        allowed = allowed_matrix if allowed_matrix is not None else _ALLOWED_FROM_TO
    else:
        allowed = [[1] * n for _ in range(n)]
    if alpha is None:
        alpha = TRANSITION_SMOOTHING_ALPHA

    # Count transitions in train: count[(from, to)] = number of times (from -> to)
    count = {}
    for seq in sequence_data.get("train", []):
        for label_from, label_to in get_transitions_in_sequence(seq):
            key = (label_from, label_to)
            count[key] = count.get(key, 0) + 1

    # Build probability matrix [from][to] with smoothing: numerator += alpha for allowed, denominator += n*alpha
    # Then renormalize each row so it sums to 1 over allowed to's.
    # When use_constraints=False we apply a minimum floor so no cell is zero (avoids underflow/display as 0).
    prob_matrix = [[0.0] * n for _ in range(n)]
    denom_extra = n * alpha  # 11 * alpha
    # When no constraints, floor ensures no cell is 0 or rounds to 0 in display (e.g. .4f)
    min_prob_floor = 1e-4 if not use_constraints else 0.0

    for from_idx in range(n):
        total_from = sum(count.get((from_idx, to_idx), 0) for to_idx in range(n))
        denominator = total_from + denom_extra
        if denominator <= 0:
            denominator = denom_extra
        row_sum = 0.0
        for to_idx in range(n):
            if allowed[from_idx][to_idx] == 0:
                prob_matrix[from_idx][to_idx] = 0.0
            else:
                numerator = count.get((from_idx, to_idx), 0) + alpha
                prob_matrix[from_idx][to_idx] = numerator / denominator
                row_sum += prob_matrix[from_idx][to_idx]
        if row_sum > 0:
            for to_idx in range(n):
                if allowed[from_idx][to_idx] == 1:
                    prob_matrix[from_idx][to_idx] /= row_sum
        # When no constraints, enforce minimum floor so no cell is zero
        if min_prob_floor > 0:
            for to_idx in range(n):
                if prob_matrix[from_idx][to_idx] < min_prob_floor:
                    prob_matrix[from_idx][to_idx] = min_prob_floor
            row_sum_after = sum(prob_matrix[from_idx][j] for j in range(n))
            if row_sum_after > 0:
                for to_idx in range(n):
                    prob_matrix[from_idx][to_idx] /= row_sum_after

    return prob_matrix


def print_transition_probs(prob_matrix, label_names=None, no_constraints=False):
    """
    Print estimated transition probabilities P(to|from) for verification.
    Rows = from, columns = to.
    label_names: list of str; default ACTION_LABELS.
    no_constraints: if True, header says all transitions are non-zero (no hard constraints).
    """
    names = label_names if label_names is not None else ACTION_LABELS
    n = len(prob_matrix)
    col_w = max(len(nm) for nm in names) if names else 4
    num_w = 10  # column width for probabilities and header
    print("\nEstimated transition probabilities P(to | from) [train]:")
    if no_constraints:
        print("  (no hard constraints; all transitions non-zero; each row sums to 1)\n")
    else:
        print("  (implausible transitions remain 0; each row sums to 1 over allowed to's)\n")
    print(f"  {'from':<{col_w}} " + " ".join(f"{names[j][:num_w]:>{num_w}}" for j in range(n)))
    print("  " + "-" * (col_w + 1 + n * (num_w + 1)))
    for i in range(n):
        row_vals = [prob_matrix[i][j] for j in range(n)]
        row_str = " ".join(f"{v:>{num_w}.4f}" if v != 0 else " " * num_w for v in row_vals)
        print(f"  {names[i]:<{col_w}} {row_str}")
    print()


def save_transition_matrix_csv(prob_matrix, csv_path, label_names=None):
    """
    Save transition matrix to CSV. Rows = from, columns = to.
    First row is header (empty cell then column labels). First column is row label (from).
    label_names: list of str for row/col names; default ACTION_LABELS.
    """
    names = label_names if label_names is not None else ACTION_LABELS
    n = len(prob_matrix)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [names[j] for j in range(n)])
        for i in range(n):
            w.writerow([names[i]] + [prob_matrix[i][j] for j in range(n)])
    print(f"Saved transition probability matrix to {csv_path}")


def load_transition_matrix_csv(csv_path, label_names=None):
    """
    Load transition probability matrix from CSV produced by save_transition_matrix_csv.
    csv_path: path to the CSV file.
    label_names: optional list to validate row/col order; if None, column names from header are used.
    Returns: prob_matrix as 2D list [from][to] of float.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Transition matrix CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_labels = header[1:]
        n = len(col_labels)
        prob_matrix = []
        for row in reader:
            if len(row) != n + 1:
                continue
            prob_matrix.append([float(row[j + 1]) for j in range(n)])
    return prob_matrix


def estimate_and_save_transition_probs(sequence_data=None, path=None, csv_path=None, no_constraints=False):
    """
    Estimate transition probabilities from train split, then save the matrix to CSV.
    sequence_data: optional; if None, loaded from default path.
    path: optional path to sequence annotations pkl.
    csv_path: where to save CSV; default is DEFAULT_TRANSITION_PROBS_CSV
      (or with a '_no_constraints_' suffix when no_constraints=True).
    no_constraints: if True, do not apply hard constraints; matrix is all non-zero (smoothed only).
    Returns (sequence_data, prob_matrix).
    """
    if sequence_data is None:
        sequence_data = load_sequence_annotations(path)
    use_constraints = not no_constraints
    prob_matrix = estimate_transition_probs_from_train(
        sequence_data,
        path=path,
        use_constraints=use_constraints,
    )
    print_transition_probs(prob_matrix, no_constraints=no_constraints)
    if csv_path is None:
        csv_path = DEFAULT_TRANSITION_PROBS_CSV
    if no_constraints and "_no_constraints_" not in os.path.basename(csv_path):
        root, ext = os.path.splitext(csv_path)
        out_path = root + "_no_constraints_" + (ext if ext else ".csv")
    else:
        out_path = csv_path
    save_transition_matrix_csv(prob_matrix, out_path)
    return sequence_data, prob_matrix


def load_sequence_annotations(path=None):
    """Load the sequence annotations pkl. Returns dict with keys 'train', 'val', 'test'."""
    p = path if path is not None else DEFAULT_SEQUENCE_ANNOTATIONS_PATH
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Sequence annotations not found: {p}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data


def get_transitions_in_sequence(sequence):
    """Yield (label_from, label_to) for each consecutive pair of actions in the sequence."""
    actions = sequence.get("actions") or []
    for i in range(len(actions) - 1):
        # actions[i] = [annotation_id, start_frame, end_frame, label]
        label_from = actions[i][3]
        label_to = actions[i + 1][3]
        yield (label_from, label_to)


def is_transition_allowed(label_from, label_to, allowed_matrix=None):
    """Return True if transition (label_from -> label_to) is allowed, False if implausible."""
    mat = allowed_matrix if allowed_matrix is not None else _ALLOWED_FROM_TO
    if label_from < 0 or label_from >= len(mat):
        return False
    if label_to < 0 or label_to >= len(mat[0]):
        return False
    return mat[label_from][label_to] == 1


def find_implausible_transitions(sequence_data, implausible_set=None, allowed_matrix=None):
    """
    Check all sequences for transitions that are implausible.
    sequence_data: dict with keys 'train'/'val'/'test', each a list of sequence dicts.
    implausible_set: optional set of (label_from, label_to) to use instead of matrix.
    allowed_matrix: optional 2D list; if None use ALLOWED_TRANSITIONS. Ignored if implausible_set is provided.
    Returns list of dicts, one per violation, with keys:
      split_name, sequence_id, game_name, tracklet_id,
      label_from, label_to, label_from_name, label_to_name,
      annotation_id_from, annotation_id_to,
      start_frame_from, end_frame_from, start_frame_to, end_frame_to.
    """
    use_matrix = implausible_set is None
    violations = []
    for split_name in ("train", "val", "test"):
        for seq in sequence_data.get(split_name, []):
            seq_id = seq.get("sequence_id", "")
            actions = seq.get("actions") or []
            for i in range(len(actions) - 1):
                act_from = actions[i]
                act_to = actions[i + 1]
                ann_id_from, start_from, end_from, label_from = act_from[0], act_from[1], act_from[2], act_from[3]
                ann_id_to, start_to, end_to, label_to = act_to[0], act_to[1], act_to[2], act_to[3]
                if use_matrix:
                    violation = not is_transition_allowed(label_from, label_to, allowed_matrix)
                else:
                    violation = (label_from, label_to) in implausible_set
                if violation:
                    violations.append({
                        "split_name": split_name,
                        "sequence_id": seq_id,
                        "game_name": seq.get("game_name", ""),
                        "tracklet_id": seq.get("tracklet_id", None),
                        "label_from": label_from,
                        "label_to": label_to,
                        "label_from_name": ACTION_LABELS[label_from] if 0 <= label_from < len(ACTION_LABELS) else str(label_from),
                        "label_to_name": ACTION_LABELS[label_to] if 0 <= label_to < len(ACTION_LABELS) else str(label_to),
                        "annotation_id_from": ann_id_from,
                        "annotation_id_to": ann_id_to,
                        "start_frame_from": start_from,
                        "end_frame_from": end_from,
                        "start_frame_to": start_to,
                        "end_frame_to": end_to,
                    })
    return violations


def get_disallowed_transitions(allowed_matrix=None):
    """Return list of (label_from, label_to) that are not allowed. Uses _ALLOWED_FROM_TO by default."""
    mat = allowed_matrix if allowed_matrix is not None else _ALLOWED_FROM_TO
    return [(i, j) for i in range(len(mat)) for j in range(len(mat[0])) if mat[i][j] == 0]


def print_disallowed_transitions(allowed_matrix=None):
    """Print all disallowed (from -> to) transitions with label names for verification."""
    disallowed = get_disallowed_transitions(allowed_matrix)
    print("Disallowed transitions (from -> to):")
    for l_from, l_to in sorted(disallowed):
        name_from = ACTION_LABELS[l_from] if 0 <= l_from < len(ACTION_LABELS) else str(l_from)
        name_to = ACTION_LABELS[l_to] if 0 <= l_to < len(ACTION_LABELS) else str(l_to)
        print(f"  {name_from} ({l_from}) -> {name_to} ({l_to})")
    print(f"Total: {len(disallowed)} disallowed pairs\n")


def format_disallowed_violation(v):
    """Format a single violation dict as a multi-line string with all details."""
    return (
        f"  game_name: {v['game_name']}\n"
        f"  split: {v['split_name']}  sequence_id: {v['sequence_id']}\n"
        f"  tracklet_id: {v['tracklet_id']}\n"
        f"  transition: {v['label_from_name']} ({v['label_from']}) -> {v['label_to_name']} ({v['label_to']})\n"
        f"  annotation_ids: {v['annotation_id_from']} -> {v['annotation_id_to']}\n"
        f"  frames (from): start={v['start_frame_from']} end={v['end_frame_from']}\n"
        f"  frames (to):   start={v['start_frame_to']} end={v['end_frame_to']}"
    )


def export_disallowed_transitions_report(violations, json_path=None, csv_path=None):
    """
    Export violation list to JSON and/or CSV with full info: game_name, frames (start/end),
    tracklet_id, annotation_ids, split, sequence_id, label names.
    violations: list of dicts from find_implausible_transitions().
    """
    if not violations:
        return
    if json_path:
        with open(json_path, "w") as f:
            json.dump(violations, f, indent=2)
        print(f"Wrote {len(violations)} disallowed transitions to {json_path}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "game_name", "split_name", "sequence_id", "tracklet_id",
                "label_from", "label_to", "label_from_name", "label_to_name",
                "annotation_id_from", "annotation_id_to",
                "start_frame_from", "end_frame_from", "start_frame_to", "end_frame_to",
            ])
            w.writeheader()
            w.writerows(violations)
        print(f"Wrote {len(violations)} disallowed transitions to {csv_path}")


def verify_transitions(sequence_data=None, implausible_set=None, allowed_matrix=None, path=None):
    """
    Load sequence annotations (if not provided), run implausible-transition check using
    ALLOWED_TRANSITIONS matrix (or implausible_set / allowed_matrix if provided), and print results.
    Returns (sequence_data, list of violation dicts).
    """
    if sequence_data is None:
        sequence_data = load_sequence_annotations(path)
    violations = find_implausible_transitions(
        sequence_data, implausible_set=implausible_set, allowed_matrix=allowed_matrix
    )
    mat = allowed_matrix if allowed_matrix is not None else _ALLOWED_FROM_TO
    n_implausible = sum(1 for i in range(len(mat)) for j in range(len(mat[0])) if mat[i][j] == 0) if implausible_set is None else len(implausible_set)
    print(f"Transition matrix: {len(mat)}x{len(mat[0])}, {n_implausible} disallowed pairs")
    if violations:
        print(f"Found {len(violations)} implausible transition(s):")
        for v in violations:
            print(format_disallowed_violation(v))
            print()
    else:
        print("No implausible transitions found in sequences.")
    return sequence_data, violations


def viterbi_sequence(probs, A, eps=1e-8):
    """
    Given emission probs (T, 11) and transition matrix A (11, 11) [from, to], compute:
    1) path: MAP label sequence via Viterbi; score = sum_t log(probs[t,y_t]) + TRANSITION_WEIGHT * sum_{t>0} log(A[y_{t-1},y_t]).
    2) adjusted_probs: (T, 11) where adjusted_probs[t] = softmax(dp[t,:]), dp[t,j] = best log-score ending in j at t.
    Work in log-space; clamp probs and A with eps before log.
    """
    T, K = probs.shape
    A = np.asarray(A, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    A = np.clip(A, eps, 1.0)
    A = A / A.sum(axis=1, keepdims=True)
    log_p = np.log(probs)
    log_A = np.log(A) * TRANSITION_WEIGHT
    dp = np.zeros((T, K), dtype=np.float64)
    backptr = np.zeros((T, K), dtype=np.int32)
    dp[0] = log_p[0]
    for t in range(1, T):
        # dp[t, j] = max_k ( dp[t-1, k] + TRANSITION_WEIGHT * log_A[k, j] ) + log_p[t, j]
        trans = dp[t - 1, :, None] + log_A.T
        backptr[t] = np.argmax(trans, axis=0)
        dp[t] = np.max(trans, axis=0) + log_p[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    # adjusted_probs[t] = softmax(dp[t, :])
    dp_shift = dp - dp.max(axis=1, keepdims=True)
    exp_dp = np.exp(dp_shift)
    adjusted_probs = exp_dp / exp_dp.sum(axis=1, keepdims=True)
    return path, adjusted_probs


def _get_scores_array(value, n):
    """Extract length-n float array from result value. Returns None if not found."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == n:
        return np.array(value, dtype=np.float64)
    if hasattr(value, "shape") and value.shape == (n,):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, dict):
        for k in ("scores", "probs", "logits", "pred_scores", "pred_score"):
            if k in value and hasattr(value[k], "__len__") and len(value[k]) == n:
                return np.asarray(value[k], dtype=np.float64)
        if "pred_scores" in value and hasattr(value["pred_scores"], "shape"):
            arr = np.asarray(value["pred_scores"], dtype=np.float64)
            if arr.size == n:
                return arr.ravel()
    return None


def _scores_to_probs(scores, eps=1e-8):
    """Convert n-dim scores to probability distribution (softmax), then clamp with eps and renormalize."""
    scores = np.asarray(scores, dtype=np.float64)
    x = scores - scores.max()
    exp_x = np.exp(x)
    probs = exp_x / exp_x.sum()
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum()
    return probs


def _set_value_probs_and_pred(container, probs_vec, pred_label):
    """Set 11-dim probs and pred label into result structure (dict or plain). Never returns None."""
    probs_list = probs_vec.tolist() if hasattr(probs_vec, "tolist") else list(probs_vec)
    if isinstance(container, dict):
        out = dict(container)
        for k in ("scores", "probs", "logits"):
            if k in out:
                out[k] = probs_list
        out["probs"] = probs_list
        out["pred"] = int(pred_label)
        if "label" in out:
            out["label"] = int(pred_label)
        if "pred_label" in out:
            out["pred_label"] = int(pred_label)
        return out
    return {"probs": probs_list, "pred": int(pred_label)}


def adjust_predictions_with_transitions(
    results_pkl_path,
    transition_matrix_csv_path=None,
    sequence_annotations_path=None,
):
    """
    Load results, transition matrix, and sequence annotations. Match result keys
    (format: 'game_name_start_frame_end_frame_tracklet_id') to actions in sequences.
    For actions that are the 2nd or later in a sequence, update prediction using
    previous action's prediction and the transition probability matrix. Save
    adjusted results to a new pkl in the same directory as the source file
    (suffix _adjusted_w{TRANSITION_WEIGHT}.pkl).

    results_pkl_path: path to the pkl file containing predicted results (dict
        keyed by 'game_start_end_tracklet' strings).
    transition_matrix_csv_path: path to the transition probability matrix CSV.
        If None, uses DEFAULT_TRANSITION_PROBS_CSV.
    sequence_annotations_path: path to sequence_annotations.pkl. If None, uses
        DEFAULT_SEQUENCE_ANNOTATIONS_PATH.

    Returns: (adjusted_results, transition_probs).
    """
    if transition_matrix_csv_path is None:
        transition_matrix_csv_path = DEFAULT_TRANSITION_PROBS_CSV
    if not os.path.isfile(results_pkl_path):
        raise FileNotFoundError(f"Results pkl not found: {results_pkl_path}")

    with open(results_pkl_path, "rb") as f:
        results = pickle.load(f)

    transition_probs = load_transition_matrix_csv(transition_matrix_csv_path)
    sequence_data = load_sequence_annotations(sequence_annotations_path)

    def result_key(game_name, start_frame, end_frame, tracklet_id):
        """Build results dict key from (game_name, start_frame, end_frame, tracklet_id)."""
        return f"{game_name}_{start_frame}_{end_frame}_{tracklet_id}"

    n = len(ACTION_LABELS)
    A = np.array(transition_probs, dtype=np.float64)
    eps = 1e-8

    if not isinstance(results, dict):
        raise TypeError("Results pkl must be a dict keyed by clip key (e.g. frame_dir or game_start_end_tracklet).")
    adjusted_results = dict(results)

    # Build per-sequence ordered lists of result keys (same format as results keys).
    sequences_keys = {}
    for split_name in ("train", "val", "test"):
        for seq_idx, seq in enumerate(sequence_data.get(split_name, [])):
            keys_in_order = []
            for act in seq.get("actions") or []:
                _, start_frame, end_frame, _ = act[0], act[1], act[2], act[3]
                k = result_key(seq["game_name"], start_frame, end_frame, seq["tracklet_id"])
                keys_in_order.append(k)
            id_ = (split_name, seq_idx)
            sequences_keys[id_] = keys_in_order

    processed_in_sequence = set()
    for (split_name, seq_idx), keys_in_order in sequences_keys.items():
        if not keys_in_order:
            continue
        probs_list = []
        valid = True
        for k in keys_in_order:
            v = results.get(k)
            arr = _get_scores_array(v, n)
            if arr is None:
                valid = False
                break
            probs_list.append(_scores_to_probs(arr, eps))
        if not valid or not probs_list:
            continue
        probs = np.stack(probs_list, axis=0)
        path, adjusted_probs = viterbi_sequence(probs, A, eps)
        for i, k in enumerate(keys_in_order):
            processed_in_sequence.add(k)
            orig = results.get(k)
            adjusted_results[k] = _set_value_probs_and_pred(orig if orig is not None else {}, adjusted_probs[i], path[i])

    for key, value in results.items():
        if key in processed_in_sequence:
            continue
        arr = _get_scores_array(value, n)
        if arr is not None:
            p = _scores_to_probs(arr, eps)
            pred = int(np.argmax(p))
            adjusted_results[key] = _set_value_probs_and_pred(value if value is not None else {}, p, pred)

    results_dir = os.path.dirname(os.path.abspath(results_pkl_path))
    base_name = os.path.splitext(os.path.basename(results_pkl_path))[0]
    out_path = os.path.join(results_dir, base_name + f"_adjusted_w{TRANSITION_WEIGHT}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(adjusted_results, f)
    print(f"Saved adjusted results to {out_path}")

    return adjusted_results, transition_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence modelling: transition probs and prediction adjustment.")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to results pkl file; if set, load results and transition matrix and run adjust_predictions_with_transitions.",
    )
    parser.add_argument(
        "--transition-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to transition probability matrix CSV (default: data/hockey/transition_probs.csv). Used when --results is set.",
    )
    parser.add_argument(
        "--sequence-annotations",
        type=str,
        default=DEFAULT_SEQUENCE_ANNOTATIONS_PATH,
        metavar="PATH",
        help="Path to sequence_annotations.pkl (train/val/test sequence structure).",
    )
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="When estimating transition probs only: use estimated values only (no hard constraints); matrix is all non-zero. Saved to a file with '_no_constraints_' in the name.",
    )
    args = parser.parse_args()

    if args.results is not None:
        adjusted_results, transition_probs = adjust_predictions_with_transitions(
            results_pkl_path=args.results,
            transition_matrix_csv_path=args.transition_csv,
            sequence_annotations_path=args.sequence_annotations,
        )
        print("Adjusted predictions saved (see message above for output path).")
    else:
        estimate_and_save_transition_probs(
            path=args.sequence_annotations,
            csv_path=DEFAULT_TRANSITION_PROBS_CSV,
            no_constraints=args.no_constraints,
        )
