#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI から 88鍵のフレームラベルを生成して .pkl に保存するツール。

出力形式は既定で dict[int -> (88,)] （各フレーム t の 0/1 ベクトル）。
--as-ndarray を付ければ (T,88) ndarray をそのまま保存。

ラベル種別は既定で "onset-only"（各音の開始フレームに 1）。
--mark-active を付けると発音区間中を 1 にする（フレームラベル）。

T（フレーム数）の決め方:
  1) --csv を渡した場合: CSVの frame 列から最大 frame_idx+1 を使用（動画フレーム列に揃う）
  2) --T を明示指定
  3) 上記が無ければ MIDI の終了時刻から T=ceil(end/hop)+1 を推定

使い方:
  python3 gen_midi_lables.py \
    --midi ../midi/9_re.mid \
    --out  ./labels/9_processed_renumbered.pkl \
    --hop  0.04 \
    --csv  ../preprocessed_csv/9_processed_renumbered.csv \
    --onset-only \
    --verbose

依存: pretty_midi, numpy, pandas（--csv 使用時）
"""

import os
import math
import argparse
import pickle
from typing import Dict, Tuple, Optional

import numpy as np
import pretty_midi as pm

try:
    import pandas as pd
except Exception:
    pd = None

PITCH_LO = 21  # A0
PITCH_HI = 108 # C8 (inclusive)
N_KEYS   = 88


def _infer_T_from_csv(csv_path: str) -> int:
    if pd is None:
        raise RuntimeError("pandas が見つかりません。--csv を使うには pandas が必要です")
    import re
    df = pd.read_csv(csv_path)
    if 'frame' not in df.columns:
        raise ValueError("CSV に 'frame' 列が必要です (例: Bumblebee.mp4_21_0)")
    # base_frame の末尾 _<frame_idx> を抜く
    base = df['frame'].astype(str).str.rsplit('_', n=1).str[0]
    frame_idx = base.str.rsplit('_', n=1).str[-1].astype(int)
    T = int(frame_idx.max()) + 1
    return T


def _infer_T_from_midi(mid: pm.PrettyMIDI, hop: float) -> int:
    t_end = 0.0
    for inst in mid.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            t_end = max(t_end, note.end)
    T = int(math.ceil(t_end / hop)) + 1
    return max(T, 1)


def _empty_labels(T: int) -> np.ndarray:
    return np.zeros((T, N_KEYS), dtype=np.float32)


def _pitch_to_index(p: int) -> Optional[int]:
    if p < PITCH_LO or p > PITCH_HI:
        return None
    return p - PITCH_LO


def _assign_onset(labels: np.ndarray, t: int, p: int):
    idx = _pitch_to_index(p)
    if idx is not None and 0 <= t < labels.shape[0]:
        labels[t, idx] = 1.0


def _assign_active(labels: np.ndarray, t0: int, t1: int, p: int):
    idx = _pitch_to_index(p)
    if idx is None:
        return
    t0 = max(t0, 0)
    t1 = min(t1, labels.shape[0])
    if t1 > t0:
        labels[t0:t1, idx] = 1.0


def build_labels_from_midi(midi_path: str, hop: float, T: Optional[int] = None,
                            align: str = 'floor', onset_only: bool = True,
                            epsilon: float = 1e-6, verbose: bool = False) -> np.ndarray:
    """MIDI から (T,88) を作る。align は onset のフレーム決定法。
    - 'floor': t = floor((t_note+eps)/hop)
    - 'round': t = round((t_note)/hop)
    - 'ceil' : t = ceil ((t_note-eps)/hop)
    """
    assert align in ('floor', 'round', 'ceil')
    mid = pm.PrettyMIDI(midi_path)

    if T is None:
        T = _infer_T_from_midi(mid, hop)
    labels = _empty_labels(T)

    # 先に onset-only を打つ
    for inst in mid.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            t_note = float(note.start)
            if align == 'floor':
                t = int(math.floor((t_note + epsilon) / hop))
            elif align == 'round':
                t = int(round(t_note / hop))
            else:
                t = int(math.ceil((t_note - epsilon) / hop))
            _assign_onset(labels, t, note.pitch)

    # 発音区間もマークする場合
    if not onset_only:
        for inst in mid.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                t0 = int(math.floor((float(note.start) + epsilon) / hop))
                t1 = int(math.ceil ((float(note.end)   - epsilon) / hop))
                _assign_active(labels, t0, t1, note.pitch)

    if verbose:
        nonzero = int(labels.sum())
        print(f"[stats] T={T}, hop={hop}, onset_only={onset_only}, nonzero={nonzero}")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--midi', required=True, help='入力 MIDI ファイル')
    ap.add_argument('--out', required=True, help='出力 .pkl パス（dict か ndarray を保存）')
    ap.add_argument('--hop', type=float, default=0.02, help='フレーム時間 (sec)')
    ap.add_argument('--T', type=int, default=None, help='フレーム数を明示指定（CSVが無ければ推奨）')
    ap.add_argument('--csv', default=None, help='動画CSV（frame 列から T を推定）')
    ap.add_argument('--align', choices=['floor','round','ceil'], default='floor', help='onset のフレーム決定法')
    ap.add_argument('--onset-only', action='store_true', help='オンセットのみを 1 とする（既定）')
    ap.add_argument('--mark-active', action='store_true', help='発音区間も 1 にする（onset-only を上書き併用）')
    ap.add_argument('--as-ndarray', action='store_true', help='True のとき (T,88) ndarray をそのまま保存')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    if args.csv is not None:
        T = _infer_T_from_csv(args.csv)
    else:
        T = args.T

    labels = build_labels_from_midi(
        midi_path=args.midi,
        hop=args.hop,
        T=T,
        align=args.align,
        onset_only=not args.mark_active if args.onset_only or not args.mark_active else False,
        verbose=args.verbose,
    )

    # 保存
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if args.as_ndarray:
        with open(args.out, 'wb') as f:
            pickle.dump(labels.astype(np.float32), f, protocol=4)
        if args.verbose:
            print(f"[save] ndarray (T,88) -> {args.out}  T={labels.shape[0]}")
    else:
        d: Dict[int, np.ndarray] = {int(t): labels[t].astype(np.float32) for t in range(labels.shape[0])}
        with open(args.out, 'wb') as f:
            pickle.dump(d, f, protocol=4)
        if args.verbose:
            print(f"[save] dict[int->(88,)] -> {args.out}  T={labels.shape[0]}")


if __name__ == '__main__':
    main()
