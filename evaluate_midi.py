#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_midi.py

GT(正解)のMIDIと推論MIDIを比較して、以下を評価します:
  - ノート単位 (onset のみ / onset+offset) の Precision / Recall / F1
  - ※ --onset-only を付けると offset/duration を完全に無視（onset一致のみ計算）
  - 速度(ベロシティ)の MAE / 相関 (一致ノートのみ; 任意)
  - フレーム単位 (任意 hop 秒) の Precision / Recall / F1
  - 一定遅延(レイテンシ)の自動最適化 (±100ms; 10ms刻み; onset F1 最大化)
  - sustain pedal(CC64) の扱い (有効化するとペダル下はノート長を延長)
  - 相対開始: 各MIDIの最初のノート開始を t=0 に揃える（--align-first-note）

依存: pretty_midi, numpy, mido(間接), (pandasは出力CSVに使う場合のみ)

使い方(単一ファイル):
  python evaluate_midi.py --gt ../audio_1/1.mid --pred ../audio_1/fused.mid \
    --onset-tol 0.05 --offset-tol 0.05 --offset-rel 0.2 \
    --pedal --search-latency --hop 0.02
  # onsetのみ
  python3 evaluate_midi.py --gt ../audio_1/1.mid --pred ../audio_1/1_om.mid --onset-only
  # 相対開始（両MIDIとも最初のノートをt=0に正規化）
  python3 evaluate_midi.py --gt ../audio_1/1.mid --pred ../audio_1/best.mid --align-first-note --onset-only --search-latency

使い方(ディレクトリ; ファイル名の stem で対応付け):
  python evaluate_midi.py --gt-dir ./GT --pred-dir ./PRED \
    --onset-tol 0.05 --offset-tol 0.05 --offset-rel 0.2 \
    --pedal --search-latency --hop 0.02 --csv summary.csv
  # onsetのみ（複数曲）
  python evaluate_midi.py --gt-dir ./GT --pred-dir ./PRED --onset-only --csv summary_onset.csv


"""

from __future__ import annotations
import argparse, math, json, sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pretty_midi
from pathlib import Path

# ----------------------------- 基本データ構造 -----------------------------

@dataclass
class Note:
    pitch: int
    onset: float
    offset: float
    velocity: int

    @property
    def duration(self) -> float:
        return max(0.0, self.offset - self.onset)


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    pairs: List[Tuple[Note, Note]]  # (gt, pred)


# ----------------------------- ユーティリティ -----------------------------

def _piano_programs() -> set:
    # GM: Acoustic Grand(0)〜Clavinet(7)を piano 扱い
    return set(range(0, 8))


def load_midi_notes(path: str, only_piano: bool = True, ignore_drums: bool = True, transpose: int = 0) -> List[Note]:
    pm = pretty_midi.PrettyMIDI(path)
    notes: List[Note] = []
    for inst in pm.instruments:
        if ignore_drums and inst.is_drum:
            continue
        if only_piano and inst.program not in _piano_programs():
            continue
        for n in inst.notes:
            pitch = int(n.pitch) + int(transpose)
            if pitch < 0 or pitch > 127:
                continue
            notes.append(Note(pitch, float(n.start), float(n.end), int(n.velocity)))
    notes.sort(key=lambda x: (x.onset, x.pitch, x.offset))
    return notes


def extract_pedal_intervals(pm: pretty_midi.PrettyMIDI, sustain_threshold: int = 64) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    all_cc = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number == 64:
                all_cc.append((cc.time, cc.value))
    if not all_cc:
        return intervals
    all_cc.sort()
    on = False
    start = None
    for t, v in all_cc:
        if not on and v >= sustain_threshold:
            on = True; start = t
        elif on and v < sustain_threshold:
            on = False; intervals.append((start, t)); start = None
    if on:
        end_time = max((n.end for inst in pm.instruments for n in inst.notes), default=0.0)
        intervals.append((start, end_time))
    if not intervals:
        return intervals
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def apply_sustain_extension(notes: List[Note], pedal_intervals: List[Tuple[float, float]]) -> List[Note]:
    if not pedal_intervals or not notes:
        return notes
    by_pitch: Dict[int, List[Note]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    for pitch in by_pitch:
        by_pitch[pitch].sort(key=lambda x: x.onset)

    def extend_time(t_end: float) -> float:
        for s, e in pedal_intervals:
            if s <= t_end <= e:
                return e
        return t_end

    new_notes: List[Note] = []
    for pitch, ns in by_pitch.items():
        for i, n in enumerate(ns):
            new_end = extend_time(n.offset)
            if i + 1 < len(ns):
                nxt = ns[i + 1]
                new_end = min(new_end, nxt.onset)
            if new_end < n.onset:
                new_end = n.offset
            new_notes.append(Note(n.pitch, n.onset, new_end, n.velocity))
    new_notes.sort(key=lambda x: (x.onset, x.pitch, x.offset))
    return new_notes


# ----------------------------- 評価コア -----------------------------

def match_notes(
    gt: List[Note],
    pr: List[Note],
    onset_tol: float = 0.05,
    require_offset: bool = False,
    offset_tol: float = 0.05,
    offset_rel: float = 0.2,
) -> MatchResult:
    tp = 0
    pairs: List[Tuple[Note, Note]] = []
    gt_by_pitch: Dict[int, List[Note]] = {}
    pr_by_pitch: Dict[int, List[Note]] = {}
    for n in gt:
        gt_by_pitch.setdefault(n.pitch, []).append(n)
    for n in pr:
        pr_by_pitch.setdefault(n.pitch, []).append(n)
    for p in gt_by_pitch:
        gt_by_pitch[p].sort(key=lambda x: x.onset)
    for p in pr_by_pitch:
        pr_by_pitch[p].sort(key=lambda x: x.onset)

    for pitch in set(list(gt_by_pitch.keys()) + list(pr_by_pitch.keys())):
        gts = gt_by_pitch.get(pitch, [])
        prs = pr_by_pitch.get(pitch, [])
        gi_used = set()
        for prn in prs:
            candidates = []
            for gi, gtn in enumerate(gts):
                if gi in gi_used:
                    continue
                d_on = abs(prn.onset - gtn.onset)
                if d_on <= onset_tol:
                    if require_offset:
                        d_dur = abs(prn.duration - gtn.duration)
                        if d_dur > max(offset_tol, offset_rel * gtn.duration):
                            continue
                    candidates.append((d_on, gi, gtn))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, gi, gtn = candidates[0]
                gi_used.add(gi)
                tp += 1
                pairs.append((gtn, prn))
    fp = len(pr) - tp
    fn = len(gt) - tp
    return MatchResult(tp=tp, fp=fp, fn=fn, pairs=pairs)


def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F


def frame_roll(notes: List[Note], hop: float, max_time: Optional[float] = None) -> np.ndarray:
    if not notes:
        T = 0 if max_time is None else int(np.ceil(max_time / hop))
        return np.zeros((T, 88), dtype=np.uint8)
    if max_time is None:
        max_time = max(n.offset for n in notes)
    T = int(np.ceil(max_time / hop))
    roll = np.zeros((T, 88), dtype=np.uint8)
    for n in notes:
        s = max(0, int(np.floor(n.onset / hop)))
        e = min(T, int(np.ceil(n.offset / hop)))
        if e > s and 0 <= n.pitch < 128:
            p = min(87, max(0, n.pitch - 21))
            roll[s:e, p] = 1
    return roll


def frame_metrics(gt: List[Note], pr: List[Note], hop: float) -> Tuple[float, float, float]:
    if hop <= 0:
        return 0.0, 0.0, 0.0
    max_t = 0.0
    if gt:
        max_t = max(max_t, max(n.offset for n in gt))
    if pr:
        max_t = max(max_t, max(n.offset for n in pr))
    if max_t <= 0:
        return 0.0, 0.0, 0.0
    Rg = frame_roll(gt, hop, max_t)
    Rp = frame_roll(pr, hop, max_t)
    tp = int(np.logical_and(Rg == 1, Rp == 1).sum())
    fp = int(np.logical_and(Rg == 0, Rp == 1).sum())
    fn = int(np.logical_and(Rg == 1, Rp == 0).sum())
    return prf1(tp, fp, fn)


def velocity_stats(pairs: List[Tuple[Note, Note]]) -> Dict[str, float]:
    if not pairs:
        return {"mae": 0.0, "pearson_r": 0.0}
    v_gt = np.array([g.velocity for g, _ in pairs], dtype=float)
    v_pr = np.array([p.velocity for _, p in pairs], dtype=float)
    mae = float(np.mean(np.abs(v_gt - v_pr)))
    if v_gt.std() == 0 or v_pr.std() == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(v_gt, v_pr)[0, 1])
    return {"mae": mae, "pearson_r": r}


def shift_notes(notes: List[Note], delta: float) -> List[Note]:
    return [Note(n.pitch, n.onset + delta, n.offset + delta, n.velocity) for n in notes]


# ----------------------------- 遅延探索 -----------------------------

def auto_latency_search(gt: List[Note], pr: List[Note], onset_tol: float) -> float:
    best_delta = 0.0
    best_f1 = -1.0
    for k in range(-10, 11):
        delta = 0.01 * k
        mr = match_notes(gt, shift_notes(pr, delta), onset_tol=onset_tol, require_offset=False)
        _, _, f1 = prf1(mr.tp, mr.fp, mr.fn)
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_delta = delta
    return best_delta


# ----------------------------- メイン処理 -----------------------------

def evaluate_pair(
    gt_path: str,
    pred_path: str,
    args: argparse.Namespace
) -> Dict[str, float]:
    pm_gt = pretty_midi.PrettyMIDI(gt_path)
    pm_pr = pretty_midi.PrettyMIDI(pred_path)

    gt_notes = load_midi_notes(gt_path, only_piano=not args.all_instruments, ignore_drums=True, transpose=args.transpose_gt)
    pr_notes = load_midi_notes(pred_path, only_piano=not args.all_instruments, ignore_drums=True, transpose=args.transpose_pred)

    # ペダル延長（先に適用）
    if args.pedal:
        ped_gt = extract_pedal_intervals(pm_gt)
        ped_pr = extract_pedal_intervals(pm_pr)
        gt_notes = apply_sustain_extension(gt_notes, ped_gt)
        pr_notes = apply_sustain_extension(pr_notes, ped_pr)

    # 相対開始: 各MIDIの最初のノート開始をt=0に揃える
    if args.align_first_note:
        if gt_notes:
            gt_shift = -gt_notes[0].onset
            gt_notes = shift_notes(gt_notes, gt_shift)
        if pr_notes:
            pr_shift = -pr_notes[0].onset
            pr_notes = shift_notes(pr_notes, pr_shift)

    # レイテンシ探索（predを丸ごとシフト）
    applied_delta = 0.0
    if args.search_latency:
        applied_delta = auto_latency_search(gt_notes, pr_notes, args.onset_tol)
        if abs(applied_delta) > 1e-9:
            pr_notes = shift_notes(pr_notes, applied_delta)

    # onset のみ
    mr_on = match_notes(
        gt_notes, pr_notes,
        onset_tol=args.onset_tol,
        require_offset=False
    )
    P_on, R_on, F_on = prf1(mr_on.tp, mr_on.fp, mr_on.fn)

    # onset-only の場合は offset と frame をスキップ
    if args.onset_only:
        return {
            "n_gt": len(gt_notes),
            "n_pred": len(pr_notes),
            "delta_applied": applied_delta,
            "P_on": P_on, "R_on": R_on, "F_on": F_on,
            "P_of": float("nan"), "R_of": float("nan"), "F_of": float("nan"),
            "P_fr": float("nan"), "R_fr": float("nan"), "F_fr": float("nan"),
            "vel_mae": 0.0,
            "vel_pearson_r": 0.0,
        }

    # onset+offset
    mr_of = match_notes(
        gt_notes, pr_notes,
        onset_tol=args.onset_tol,
        require_offset=True,
        offset_tol=args.offset_tol,
        offset_rel=args.offset_rel
    )
    P_of, R_of, F_of = prf1(mr_of.tp, mr_of.fp, mr_of.fn)

    # ベロシティ(一致ノートのみ)
    vstats = velocity_stats(mr_on.pairs)

    # フレーム (任意; hop<=0 の場合はスキップ)
    if args.hop > 0:
        P_fr, R_fr, F_fr = frame_metrics(gt_notes, pr_notes, hop=args.hop)
    else:
        P_fr = R_fr = F_fr = 0.0

    return {
        "n_gt": len(gt_notes),
        "n_pred": len(pr_notes),
        "delta_applied": applied_delta,
        "P_on": P_on, "R_on": R_on, "F_on": F_on,
        "P_of": P_of, "R_of": R_of, "F_of": F_of,
        "P_fr": P_fr, "R_fr": R_fr, "F_fr": F_fr,
        "vel_mae": vstats["mae"],
        "vel_pearson_r": vstats["pearson_r"],
    }


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gt", type=str, help="GT MIDI file")
    src.add_argument("--gt-dir", type=str, help="Directory of GT MIDIs")
    ap.add_argument("--pred", type=str, help="Pred MIDI file (single-file mode)")
    ap.add_argument("--pred-dir", type=str, help="Directory of Pred MIDIs (dir mode)")
    ap.add_argument("--suffix", type=str, default="", help="Pred側のファイル名に付く接尾辞(例: _pred)")
    ap.add_argument("--onset-tol", type=float, default=0.04, help="Onset 許容秒")
    ap.add_argument("--offset-tol", type=float, default=0.05, help="Offset 許容秒(絶対値)")
    ap.add_argument("--offset-rel", type=float, default=0.2, help="Offset 許容(相対・GT長に対して)")
    ap.add_argument("--pedal", action="store_true", help="CC64に基づいてノートを延長")
    ap.add_argument("--search-latency", action="store_true", help="一定遅延を自動探索して適用(±100ms)")
    ap.add_argument("--align-first-note", action="store_true", help="各MIDIの最初のノート開始をt=0に揃えて相対時間で評価")
    ap.add_argument("--hop", type=float, default=0.0, help="フレーム評価の hop 秒(0 以下で無効)")
    ap.add_argument("--onset-only", action="store_true", help="offset/duration を完全に無視（onset一致のみ評価）")
    ap.add_argument("--all-instruments", action="store_true", help="全インストゥメント対象(デフォルトはPiano系のみ)")
    ap.add_argument("--transpose-gt", dest="transpose_gt", type=int, default=0, help="GTに適用する半音単位の移調(+12で1オクターブ上)")
    ap.add_argument("--transpose-pred", dest="transpose_pred", type=int, default=0, help="Predに適用する半音単位の移調(-24で2オクターブ下)")
    ap.add_argument("--csv", type=str, default="", help="結果をCSV保存(ディレクトリモード時のみ)")

    args = ap.parse_args()

    if args.gt:
        if not args.pred:
            print("単一ファイルモードでは --pred が必要です", file=sys.stderr)
            sys.exit(2)
        res = evaluate_pair(args.gt, args.pred, args)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    # ディレクトリモード
    gt_dir = Path(args.gt_dir)
    pr_dir = Path(args.pred_dir) if args.pred_dir else gt_dir
    rows = []
    for gt_path in sorted(gt_dir.glob("*.mid")):
        stem = gt_path.stem
        pred_name = f"{stem}{args.suffix}.mid" if args.suffix else f"{stem}.mid"
        pr_path = pr_dir / pred_name
        if not pr_path.exists():
            print(f"[SKIP] pred not found for {gt_path.name} -> {pred_name}", file=sys.stderr)
            continue
        res = evaluate_pair(str(gt_path), str(pr_path), args)
        res["piece"] = stem
        rows.append(res)
        def _fmt(x):
            try:
                return f"{float(x):.3f}" if math.isfinite(float(x)) else "NaN"
            except Exception:
                return "NaN"
        print(f"[{stem}] F_on={_fmt(res['F_on'])}, F_of={_fmt(res['F_of'])}, "
              f"F_fr={_fmt(res['F_fr'])}, Δ={res['delta_applied']:+.3f}")

    if not rows:
        print("No pairs evaluated.", file=sys.stderr)
        sys.exit(1)

    # 集計(マクロ平均)
    import pandas as pd
    df = pd.DataFrame(rows)
    means = df.drop(columns=["piece"]).mean(numeric_only=True)
    print("\n=== Macro Averages ===")
    for k, v in means.items():
        if isinstance(v, (int, float)):
            print(f"{k:>14}: {v:.4f}" if isinstance(v, float) else f"{k:>14}: {v}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {args.csv}")

if __name__ == "__main__":
    main()
