import math
import os
import shutil

import click
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from librosa.sequence import _viterbi
from numba import bool, float64, njit, uint16, int32

from model import BIOPhonemeTagger
from utils import (
    canonical_to_lang,
    decode_bio_tags,
    forced_align_bio,
    load_langs,
    load_phoneme_list,
    load_phoneme_merge_map,
    load_phones_txt,
    merge_adjacent_segments,
    save_lab,
    forced_predict_slot,
)


def resolve_save_dir(save_dir, config_path):
    """Find the model folder (contains phonemes.txt) even if the config's path is stale.

    Tries: the path as written, relative to the config file, then the config's own folder.
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for cand in (save_dir, os.path.join(config_dir, save_dir), config_dir):
        if os.path.exists(os.path.join(cand, "phonemes.txt")):
            return cand
    return save_dir


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_wavs(path):
    if os.path.isfile(path) and path.lower().endswith(".wav"):
        return [path]
    if os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".wav")
        ]
    raise ValueError(f"--input must be a .wav file or a directory: {path}")


def find_matching_txt(wav_path):
    base, _ = os.path.splitext(wav_path)
    txt_path = base + ".txt"
    return txt_path if os.path.isfile(txt_path) else None


def bio_inputs(logits, id2label):
    labels = [id2label[i] for i in range(len(id2label))]
    scores = logits.detach().double().cpu().numpy()

    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("Expected logits shaped (frames, labels).")
    if not np.isfinite(scores).all():
        raise ValueError("Decoder received non-finite logits.")
    if any(
        tag != "O" and not (tag.startswith(("B-", "I-")) and len(tag) > 2)
        for tag in labels
    ):
        raise ValueError("Expected O and valid B-/I- labels.")

    keep = [i for i, tag in enumerate(labels) if tag != "O"]
    scores = scores[:, keep]
    labels = [labels[i] for i in keep]

    starts = np.array([tag.startswith("B-") for tag in labels], dtype=np.bool)
    if not starts.any():
        raise ValueError("Decoder requires at least one B- label.")

    allowed = np.array(
        [
            [
                nxt.startswith("B-")
                or (nxt.startswith("I-") and prev in (f"B-{nxt[2:]}", nxt))
                for nxt in labels
            ]
            for prev in labels
        ],
        dtype=np.bool,
    )

    return scores, labels, allowed, starts


def constrained_decode(logits, id2label):
    scores, labels, allowed, valid = bio_inputs(logits, id2label)
    preds = []
    for frame in scores:
        best = int(np.argmax(np.where(valid, frame, -np.inf)))
        preds.append(labels[best])
        valid = allowed[best]
    return preds


def viterbi_decode(logits, id2label, viterbi_bias=1):
    if not np.isfinite(viterbi_bias) or viterbi_bias < 1:
        raise ValueError("viterbi_bias must be finite and at least 1.")

    scores, labels, allowed, starts = bio_inputs(logits, id2label)
    if scores.shape[0] == 0:
        return []

    log_probs = scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)

    transitions = np.where(allowed, 0.0, -np.inf)
    for j, tag in enumerate(labels):
        if tag.startswith("I-"):
            transitions[allowed[:, j], j] = np.log(viterbi_bias)

    initial = np.where(starts, -np.log(starts.sum()), -np.inf)
    path, _ = _viterbi(log_probs, transitions, initial)

    return [labels[int(i)] for i in path]


@njit(
uint16[:](float64[:, :], uint16[:], int32[:, :], float64[:, :], bool[:], bool[:])
)
def _forced_align_viterbi(
    log_probs,
    target_seq,
    preds,
    pred_penalty,
    can_start,
    can_end,
):
    K = len(target_seq)
    _, T = log_probs.shape
    scores = np.full((K, T), -np.inf)
    pointers = np.zeros((K, T), dtype=np.uint16)

    for s in range(K):
        if can_start[s]:
            scores[s, 0] = log_probs[target_seq[s], 0]

    for t in range(1, T):
        for s in range(K):
            best_score = -np.inf
            best_prev = s
            for j in range(preds.shape[1]):
                prev = preds[s, j]
                if prev < 0:
                    break
                cand = scores[prev, t - 1] + pred_penalty[s, j]
                if cand > best_score:
                    best_score = cand
                    best_prev = prev

            scores[s, t] = best_score + log_probs[target_seq[s], t]
            pointers[s, t] = best_prev

    end = -1
    for s in range(K):
        if can_end[s] and (end < 0 or scores[s, T - 1] >= scores[end, T - 1]):
            end = s

    path = np.zeros(T, dtype=np.uint16)
    path[-1] = end
    for t in range(T - 2, -1, -1):
        path[t] = pointers[path[t + 1], t + 1]

    return path


def build_forced_align_graph(
    phones,
    slots,
    self_loop_penalty,
    forward_penalty,
    skip_penalty,
    forced_penalty,
):
    # (forced_predict_fa) are extra nodes sitting in the gaps between the real ones.
    labels, is_forced, real, gap_nodes = [], [], [], []
    for k in range(len(phones) + 1):
        gap_nodes.append([])
        for p in slots[k]:
            gap_nodes[k].append(len(labels))
            labels.append(p)
            is_forced.append(True)
        if k < len(phones):
            real.append(len(labels))
            labels.append(phones[k])
            is_forced.append(False)

    in_nodes = [[] for _ in labels]
    for k, nodes in enumerate(gap_nodes):
        for n in nodes:
            if k > 0:
                in_nodes[n].append(real[k - 1])
            in_nodes[n].extend(m for m in nodes if m != n)
        if k < len(phones):
            if k > 0:
                in_nodes[real[k]].append(real[k - 1])
            in_nodes[real[k]].extend(nodes)

    K = 2 * len(labels)
    preds = [[] for _ in range(K)]
    for n in range(len(labels)):
        b, i = 2 * n, 2 * n + 1
        preds[i] = [(i, self_loop_penalty), (b, forward_penalty)]
        extra = (
            forced_penalty + 2 * (self_loop_penalty - forward_penalty)
            if is_forced[n]
            else 0.0
        )
        for u in in_nodes[n]:
            preds[b].append((2 * u + 1, forward_penalty + extra))
            preds[b].append((2 * u, skip_penalty + extra))

    width = max(1, max(len(p) for p in preds))
    pred_idx = np.full((K, width), -1, dtype=np.int32)
    pred_pen = np.zeros((K, width), dtype=np.float64)
    for s, items in enumerate(preds):
        for j, (prev, pen) in enumerate(items):
            pred_idx[s, j] = prev
            pred_pen[s, j] = pen

    can_start = np.zeros(K, dtype=np.bool_)
    for n in gap_nodes[0] + [real[0]]:
        can_start[2 * n] = True
    can_end = np.zeros(K, dtype=np.bool_)
    for n in [real[-1]] + gap_nodes[-1]:
        can_end[2 * n : 2 * n + 2] = True

    target_seq = []
    for label in labels:
        target_seq.extend([f"B-{label}", f"I-{label}"])
    return target_seq, pred_idx, pred_pen, can_start, can_end


def forced_align_viterbi(
    logits,
    id2label,
    phones,
    self_loop_penalty=-4.6,
    forward_penalty=-0.6,
    skip_penalty=-2.3,
    visualize_probabilities=False,
    forced_pred_phones=None,
    forced_pred_penalty=0.0
):
    label2id = {label: id for id, label in id2label.items()}
    # turn to log probs
    log_probs = (
        torch.log_softmax(logits, dim=-1).detach().double().cpu().numpy().transpose()
    )

    # make target sequence
    slots = forced_predict_slot(phones, forced_pred_phones, label2id)
    target_seq, pred_idx, pred_pen, can_start, can_end = build_forced_align_graph(
        phones,
        slots,
        self_loop_penalty,
        forward_penalty,
        skip_penalty,
        forced_pred_penalty,
    )
    target_seq_idx = np.array([label2id[phn] for phn in target_seq], dtype=np.uint16)

    path = _forced_align_viterbi(
        log_probs,
        target_seq_idx,
        pred_idx,
        pred_pen,
        can_start,
        can_end,
    )

    if visualize_probabilities:
        import matplotlib.pyplot as plt

        clean_probs = np.exp(log_probs)[target_seq_idx, :].copy()
        min_prob = clean_probs.min()

        path_tokens = target_seq_idx[path]
        token_match = target_seq_idx[:, None] == path_tokens

        path_mask = np.ones_like(clean_probs, dtype=np.bool)
        path_mask[path, np.arange(len(path))] = False

        clean_probs[token_match & path_mask] = min_prob

        plt.imshow(clean_probs, aspect="auto", interpolation="none")

    return [target_seq[p] for p in path]


def apply_hard_silence(segments, audio, sr, threshold, min_duration, silence_phoneme):
    if len(audio) == 0:
        return segments

    frame_length = int(sr * 0.01)
    if frame_length < 1:
        frame_length = 1

    pad_len = (frame_length - (len(audio) % frame_length)) % frame_length
    padded_audio = np.pad(np.abs(audio), (0, pad_len), mode="constant")

    frames = padded_audio.reshape(-1, frame_length)
    frame_max = np.max(frames, axis=1)

    is_silent_frame = frame_max < threshold

    silence_intervals = []
    in_silence = False
    start_frame = 0

    for i, silent in enumerate(is_silent_frame):
        if silent and not in_silence:
            in_silence = True
            start_frame = i
        elif not silent and in_silence:
            in_silence = False
            duration = (i - start_frame) * 0.01
            if duration >= min_duration:
                silence_intervals.append((start_frame * 0.01, i * 0.01))

    if in_silence:
        duration = (len(is_silent_frame) - start_frame) * 0.01
        if duration >= min_duration:
            silence_intervals.append((start_frame * 0.01, len(is_silent_frame) * 0.01))

    if not silence_intervals:
        return segments

    temp_segments = segments.copy()

    for sil_start, sil_end in silence_intervals:
        next_temp_segments = []
        for s_start, s_end, s_label in temp_segments:
            if s_end <= sil_start or s_start >= sil_end:
                next_temp_segments.append((s_start, s_end, s_label))
                continue

            if s_start < sil_start:
                next_temp_segments.append((s_start, sil_start, s_label))
            if s_end > sil_end:
                next_temp_segments.append((sil_end, s_end, s_label))

        temp_segments = next_temp_segments

    for s, e in silence_intervals:
        temp_segments.append((s, e, silence_phoneme))

    temp_segments.sort(key=lambda x: x[0])
    return temp_segments


def continuous_segments(segments, duration):
    if duration <= 0:
        return []
    starts = []
    for s, _, ph in sorted(segments, key=lambda seg: seg[0]):
        s = float(s)
        s = max(0.0, s)
        if s >= duration or (starts and s <= starts[-1][0]):
            continue
        starts.append((s, ph))

    if not starts:
        return []
    starts[0] = (0.0, starts[0][1])
    return [
        (s, starts[i + 1][0] if i + 1 < len(starts) else duration, ph)
        for i, (s, ph) in enumerate(starts)
    ]


DEFAULT_SILENCE_ALIASES = ("pau", "sil")


def find_matching_lab(wav_path):
    base, _ = os.path.splitext(wav_path)
    lab_path = base + ".lab"
    return lab_path if os.path.isfile(lab_path) else None


def load_phones_lab(path):
    # Reads the phoneme column of an HTK-style .lab ("start end phoneme"), or a bare phoneme per line.
    phones = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                phones.append(parts[2])
            elif len(parts) == 1:
                phones.append(parts[0])
    return phones


def parse_aliases(alias_args, silence_phoneme):
    aliases = {name: silence_phoneme for name in DEFAULT_SILENCE_ALIASES}
    for item in alias_args:
        if "=" not in item:
            raise ValueError(f"--alias must look like old=new, got: {item}")
        old, new = item.split("=", 1)
        aliases[old.strip().casefold()] = new.strip()
    return aliases


def resolve_transcript(phones, label2id, aliases):
    # Maps every transcript phoneme to a label the model knows. Exact matches win,
    # so a model that really has "sil" keeps it. Otherwise pau/sil/etc. go through the alias table.
    model_phones, unknown = [], []
    for ph in phones:
        target = ph
        if f"B-{target}" not in label2id or f"I-{target}" not in label2id:
            target = aliases.get(ph.casefold(), ph)
        if f"B-{target}" in label2id and f"I-{target}" in label2id:
            model_phones.append(target)
        else:
            unknown.append(ph)
    if unknown:
        shown = ", ".join(sorted(set(unknown))[:15])
        raise ValueError(
            f"Transcript has phonemes the model doesn't know: {shown}. "
            "Use --alias old=new to map them."
        )
    return model_phones


def strict_align_path(
    log_probs,
    label2id,
    phones,
    self_loop_penalty=-4.6,
    forward_penalty=-0.6,
    skip_penalty=-2.3,
):
    # Same graph and Viterbi as forced_align_viterbi, but with no optional gap nodes:
    # every phoneme gets at least one frame, in order, and nothing else can appear.
    if log_probs.shape[1] < len(phones):
        raise ValueError(
            f"Transcript has {len(phones)} phonemes but the audio only has "
            f"{log_probs.shape[1]} frames."
        )
    no_slots = [[] for _ in range(len(phones) + 1)]
    target_seq, pred_idx, pred_pen, can_start, can_end = build_forced_align_graph(
        phones, no_slots, self_loop_penalty, forward_penalty, skip_penalty, 0.0
    )
    target_seq_idx = np.array([label2id[t] for t in target_seq], dtype=np.uint16)
    path = _forced_align_viterbi(
        log_probs, target_seq_idx, pred_idx, pred_pen, can_start, can_end
    )
    return target_seq, path


def finalize_segments(starts, labels, duration, min_gap=0.001):
    # One segment per transcript phoneme, strictly increasing, contiguous, inside the audio.
    n = len(starts)
    fixed = []
    for k, s in enumerate(starts):
        if k == 0:
            fixed.append(0.0)
            continue
        s = max(float(s), fixed[-1] + min_gap)
        s = min(s, duration - min_gap * (n - k))
        fixed.append(s)
    return [
        (fixed[k], fixed[k + 1] if k + 1 < n else duration, labels[k])
        for k in range(n)
    ]


def show_alignment(full_logits, segments, frame_duration, visualize_labels):
    import matplotlib.pyplot as plt

    plt.imshow(
        torch.softmax(full_logits, dim=-1).numpy().transpose(),
        aspect="auto",
        interpolation="none",
    )
    if visualize_labels:
        for s, e, ph in segments:
            s /= frame_duration
            e /= frame_duration
            plt.axvline(s, color="white", linestyle="--")
            plt.text(
                (s + e) / 2,
                0,
                ph,
                color="white",
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.show()


def strict_align_audio(
    full_logits,
    full_offsets,
    model,
    config,
    phones,
    aliases,
    duration,
    visualize_probabilities=False,
    visualize_labels=False,
):
    label2id = {label: i for i, label in model.id2label.items()}
    model_phones = resolve_transcript(phones, label2id, aliases)

    args = config.get("postprocess", {}).get("forced_alignment_args", {})
    log_probs = (
        torch.log_softmax(full_logits, dim=-1).detach().double().cpu().numpy().transpose()
    )
    target_seq, path = strict_align_path(
        log_probs,
        label2id,
        model_phones,
        self_loop_penalty=args.get("self_loop_penalty", -4.6),
        forward_penalty=args.get("forward_penalty", -0.6),
        skip_penalty=args.get("skip_penalty", -2.3),
    )

    frame_duration = config["data"]["frame_duration"]
    pred_tags = [target_seq[p] for p in path]

    # Sub-frame boundaries from the offset head, same decoder as normal inference.
    # It must hand back exactly the transcript, in order, or we use plain frame boundaries.
    decoded = decode_bio_tags(pred_tags, frame_duration, full_offsets)
    if [ph for _, _, ph in decoded] == model_phones:
        starts = [s for s, _, _ in decoded]
    else:
        print("  Warning: offset refinement unavailable, using frame-level boundaries.")
        nodes = np.asarray(path, dtype=np.int64) // 2
        first = np.searchsorted(nodes, np.arange(len(model_phones)), side="left")
        starts = (first * frame_duration).tolist()

    # Labels come from the transcript exactly as written (pau stays pau).
    segments = finalize_segments(starts, phones, duration)

    if visualize_probabilities:
        show_alignment(full_logits, segments, frame_duration, visualize_labels)

    return segments


def process_audio(
    model,
    audio,
    sr,
    config,
    device,
    lang_id=None,
    merge_map=None,
    lang_name=None,
    phones=None,
    no_use_offset=False,
    decoder="constrained",
    viterbi_bias=5,
    visualize_probabilities=False,
    visualize_labels=False,
    strict_fa=False,
    aliases=None,
):
    if visualize_probabilities:
        import matplotlib.pyplot as plt

    if len(audio) == 0:
        return []
    original_duration = len(audio) / sr

    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    total_len = len(audio)

    MAX_SEC = 28.0
    CHUNK_SIZE = int(MAX_SEC * sr)

    if lang_id is not None:
        lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
    else:
        lang_tensor = torch.zeros(1, dtype=torch.long).to(device)

    accumulated_logits = []
    accumulated_offsets = []

    for start in range(0, total_len, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_len)
        chunk = audio[start:end]

        if len(chunk) < 1600:
            if len(chunk) == 0:
                continue
            pad_res = 1600 - len(chunk)
            chunk = np.pad(chunk, (0, pad_res), mode="constant")

        expected_frames = math.ceil(
            (end - start) / sr / config["data"]["frame_duration"]
        )
        input_values = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([expected_frames], device=device)

        with torch.no_grad():
            logits, offsets = model(input_values, lang_tensor, lengths=lengths)

        accumulated_logits.append(logits.squeeze(0).cpu())
        if offsets is not None:
            accumulated_offsets.append(offsets.squeeze(0).cpu())

    full_logits = torch.cat(accumulated_logits, dim=0)

    full_offsets = None
    if accumulated_offsets and not no_use_offset:
        full_offsets = torch.cat(accumulated_offsets, dim=0)

    if phones and strict_fa:
        return strict_align_audio(
            full_logits,
            full_offsets,
            model,
            config,
            phones,
            aliases or {},
            original_duration,
            visualize_probabilities=visualize_probabilities,
            visualize_labels=visualize_labels,
        )

    if phones:
        postprocess = config.get("postprocess", {})
        forced_align_args = postprocess.get("forced_alignment_args", {})
        forced_predict_fa = postprocess.get("forced_predict_fa") or []
        forced_predict_penalty = forced_align_args.get("forced_predict_penalty", 0.0)

        # cleaned up a bit too
        if decoder == "constrained":
            pred_tags = forced_align_bio(
                full_logits,
                model.id2label,
                phones,
                forced_pred_phones=forced_predict_fa,
                forced_pred_penalty=forced_predict_penalty,
            )
        elif decoder == "viterbi":
            pred_tags = forced_align_viterbi(
                full_logits,
                model.id2label,
                phones,
                self_loop_penalty=forced_align_args.get("self_loop_penalty", -4.6),
                forward_penalty=forced_align_args.get("forward_penalty", -0.6),
                skip_penalty=forced_align_args.get("skip_penalty", -2.3),
                visualize_probabilities=visualize_probabilities,
                forced_pred_phones=forced_predict_fa,
                forced_pred_penalty=forced_predict_penalty,
            )
    else:
        if visualize_probabilities:
            plt.imshow(
                torch.softmax(full_logits, dim=-1).numpy().transpose(),
                aspect="auto",
                interpolation="none",
            )

        if decoder == "constrained":
            pred_tags = constrained_decode(full_logits, model.id2label)
        elif decoder == "viterbi":
            pred_tags = viterbi_decode(
                full_logits, model.id2label, viterbi_bias=viterbi_bias
            )

    segments = decode_bio_tags(
        pred_tags, config["data"]["frame_duration"], full_offsets
    )

    all_segments = []
    for s, e, ph in segments:
        if merge_map and lang_name:
            ph = canonical_to_lang(ph, lang_name, merge_map)
        all_segments.append((s, e, ph))

    continuous_seg = continuous_segments(all_segments, original_duration)

    if visualize_probabilities:
        if visualize_labels:
            for s, e, ph in continuous_seg:
                s /= config["data"]["frame_duration"]
                e /= config["data"]["frame_duration"]
                plt.axvline(s, color="white", linestyle="--")
                plt.text(
                    (s + e) / 2,
                    0,
                    ph,
                    color="white",
                    ha="center",
                    va="top",
                    fontsize=12,
                    fontweight="bold",
                )
        plt.tight_layout()
        plt.show()

    return continuous_seg


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    default="infer_test",
    help="Path to a .wav file or folder containing .wav files",
)
@click.option(
    "--checkpoint",
    "-ckpt",
    default="checkpoints_no_env/model.ckpt",
    help="Path to WFL .ckpt file",
)
@click.option(
    "--config",
    "-c",
    default="checkpoints_no_env/config.yaml",
    help="Path to config file",
)
@click.option(
    "--lang-id",
    "-l",
    type=int,
    default=None,
    help="Language ID (int) used during training. Example: `-l 0`",
)
@click.option(
    "--no_use_offset",
    is_flag=True,
    help="Disable offset head refinement (offsets ON by default).",
)
# long silence stuff
@click.option(
    "--silence-phoneme",
    default="SP",
    help="The phoneme label to use for hard-coded silence (default: SP)",
)
@click.option(
    "--silence-threshold",
    default=0.005,
    type=float,
    help="Amplitude threshold (0.0-1.0) to consider as silence",
)
@click.option(
    "--min-silence-duration",
    default=0.5,
    type=float,
    help="Minimum duration (seconds) required to trigger hard silence",
)
@click.option(
    "--decoder-type",
    "-d",
    default="viterbi",
    type=str,
    help="Decoder type for no transcription inference [constrained|viterbi] (default: viterbi)",
)
@click.option(
    "--viterbi-bias",
    default=5,
    type=float,
    help="Amount of bias (>=1) added to frames of the same phoneme for viterbi decoding. (default: 5)",
)
@click.option(
    "--visualize-probabilities",
    "-vprobs",
    is_flag=True,
    help="Visualize the probabilities predicted by the model",
)
@click.option(
    "--visualize-labels",
    "-vlabs",
    is_flag=True,
    help="Visualize the predicted labels when visualizing probabilities",
)
@click.option(
    "--strict-fa/--no-strict-fa",
    default=True,
    help="With a transcript, align exactly the phonemes given and insert nothing extra (default: on). "
    "--no-strict-fa restores the optional gap phonemes from forced_predict_fa in the config.",
)
@click.option(
    "--alias",
    "alias_args",
    multiple=True,
    help="Extra transcript alias old=new for phonemes the model doesn't know, e.g. --alias br=AP. "
    "pau and sil already map to the silence phoneme.",
)
@click.option(
    "--use-lab",
    is_flag=True,
    help="When a wav has no .txt, use the phonemes of an existing .lab next to it as the transcript "
    "(the original is kept as .lab.bak).",
)
def main(
    input_path,
    checkpoint,
    config,
    lang_id,
    no_use_offset,
    silence_phoneme,
    silence_threshold,
    min_silence_duration,
    decoder_type,
    viterbi_bias,
    visualize_probabilities,
    visualize_labels,
    strict_fa,
    alias_args,
    use_lab,
):
    cfg = load_config(config)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"Running on: {device}")

    save_dir = resolve_save_dir(cfg["output"]["save_dir"], config)
    phonemes_path = os.path.join(save_dir, "phonemes.txt")

    if not os.path.exists(phonemes_path):
        print(f"Error: {phonemes_path} not found.")
        return

    labels = load_phoneme_list(phonemes_path)
    merge_map = load_phoneme_merge_map(os.path.join(save_dir, "phoneme_merge_map.json"))
    try:
        aliases = parse_aliases(alias_args, silence_phoneme)
    except ValueError as e:
        print(f"Error: {e}")
        return

    lang_name = None
    if lang_id is not None:
        lang_path = os.path.join(save_dir, "langs.txt")
        if os.path.exists(lang_path):
            lang2id = load_langs(lang_path)
            id2lang = {v: k for k, v in lang2id.items()}
            lang_name = id2lang.get(lang_id)
            print(f"Language: {lang_name} (ID: {lang_id})")

    print("Loading model...")
    model = BIOPhonemeTagger(cfg, labels).to(device)
    model.eval()

    # weights_only=False because I dont like the the 'untrusted-models' warning
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = (
        checkpoint_data["state_dict"]
        if "state_dict" in checkpoint_data
        else checkpoint_data
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        return

    files = collect_wavs(input_path)
    print(f"Found {len(files)} files.")

    for wav_path in files:
        print(f"Processing: {wav_path}")

        # auto-detect .txt
        txt_path = find_matching_txt(wav_path)
        phones = None
        from_lab = False
        if txt_path:
            try:
                phones = load_phones_txt(txt_path)
                if phones:
                    print(
                        f"  Forced-align enabled (found: {os.path.basename(txt_path)})"
                    )
                else:
                    phones = None
            except Exception as e:
                print(f"  Warning: failed to read {txt_path}: {e}")
                phones = None

        elif use_lab:
            lab_path = find_matching_lab(wav_path)
            if lab_path:
                try:
                    phones = load_phones_lab(lab_path) or None
                    from_lab = phones is not None
                    if phones:
                        print(
                            f"  Forced-align enabled (found: {os.path.basename(lab_path)})"
                        )
                except Exception as e:
                    print(f"  Warning: failed to read {lab_path}: {e}")
                    phones = None

        try:
            audio, sr = sf.read(wav_path)
        except Exception as e:
            print(f"Error reading {wav_path}: {e}")
            continue

        if sr != cfg["data"]["sample_rate"]:
            audio_t = torch.tensor(audio, dtype=torch.float32)
            if audio_t.dim() > 1:
                audio_t = audio_t.mean(dim=1)
            audio = torchaudio.functional.resample(
                audio_t, sr, cfg["data"]["sample_rate"]
            ).numpy()
            sr = cfg["data"]["sample_rate"]

        try:
            segments = process_audio(
                model,
                audio,
                sr,
                cfg,
                device,
                lang_id=lang_id,
                merge_map=merge_map,
                lang_name=lang_name,
                phones=phones,
                no_use_offset=no_use_offset,
                decoder=decoder_type,
                viterbi_bias=viterbi_bias,
                visualize_probabilities=visualize_probabilities,
                visualize_labels=visualize_labels,
                strict_fa=strict_fa,
                aliases=aliases,
            )
        except ValueError as e:
            print(f"  Error: {e}")
            continue

        if (
            cfg.get("postprocess", {}).get("merge_segments", "right") != "none"
            and phones is None
        ):
            segments = merge_adjacent_segments(
                segments, cfg["postprocess"]["merge_segments"]
            )

        # Apply hard silence ONLY if we are NOT using forced alignment
        # Forced alignment already knows where silence is based on the text "SP" tag if its in the txt
        # adding heuristic silence on top of forced alignment usually breaks things so yea no
        if phones is None:
            segments = apply_hard_silence(
                segments,
                audio,
                sr,
                threshold=silence_threshold,
                min_duration=min_silence_duration,
                silence_phoneme=silence_phoneme,
            )

        strict_aligned = bool(phones) and strict_fa
        if not strict_aligned:
            segments = continuous_segments(segments, len(audio) / sr)
        out_path = os.path.splitext(wav_path)[0] + ".lab"
        if from_lab:
            bak_path = out_path + ".bak"
            if not os.path.exists(bak_path):
                shutil.copyfile(out_path, bak_path)
        save_lab(out_path, segments)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
