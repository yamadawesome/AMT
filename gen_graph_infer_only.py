#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_graph_infer_one_song.py

未知の 1 曲（MediaPipe 手骨格の CSV）から、推論用の PyG Data リスト
（1フレーム=1 Data, x=(42,2), edge_index=BASE_EDGES×2）を作成して .pt に保存します。

学習用 gen_graph_learning.py と互換の形に揃えています（y は付けません）。
- x: (42, 2)  … [手0の21ノード; 手1の21ノード] × [x,y]
- edge_index: to_undirected(BASE_EDGES for hand0 + BASE_EDGES+21 for hand1)
- 属性: t(intフレーム番号), song(str), sid(int; 単曲なので0), hand_mask(torch.LongTensor shape=(2,))

CSV 仕様（柔軟に自動検出）
- フレーム識別列: 'frame' もしくは 'name' / 'basename' / 'base'
  例: "Bumblebee.mp4_21_0" のように末尾が "_0"/"_1" で手のインデックスを表す形式に対応
- 座標列: x0..x20 / y0..y20  または  x_0..x_20 / y_0..y_20  を自動検出
- 値は前処理済み（0..1 正規化など）を想定。そのまま使用します。

使い方:
  python3 gen_graph_infer_only.py \
        --csv ../../MIDI_TEST_SET/miditest_videos/27/27.csv \
        --out ../../MIDI_TEST_SET/miditest_videos/27/27_graphs.pt \
        --frame-col 'file_name' \
        --allow-single-hand \
        --verbose

オプション:
  --song-name  … Data.song に入れる名前（省略時は CSV の stem）
  --frame-col  … フレーム列名が特殊な場合に明示
  --xy-style   … 列名のスタイルを固定したい場合: auto/x0/x_0（デフォ auto）
  --skip-nan   … NaN を含む手フレームをスキップ（デフォは NaN→0 埋め）

出力:
  torch.save(list[Data], OUT_PT)
  ついでに manifest(JSON) を --manifest に出力可
"""
from __future__ import annotations
import os, re, json, argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# -------------------------- エッジ定義（片手 21ノード） --------------------------
BASE_EDGES = torch.tensor(
    [[0, 1], [1, 2], [2, 3], [3, 4],
     [0, 5], [5, 6], [6, 7], [7, 8],
     [5, 9], [9,10], [10,11], [11,12],
     [9,13], [13,14], [14,15], [15,16],
     [13,17], [17,18], [18,19], [19,20]],
    dtype=torch.long
).t().contiguous()  # (2, 20)


def make_edges() -> torch.Tensor:
    """左右手の BASE_EDGES を結合して無向化（クロス無し）。"""
    e0 = BASE_EDGES.clone()            # (2,20)
    e1 = (BASE_EDGES + 21).clone()     # (2,20), 手1はノード番号+21
    e = torch.cat([e0, e1], dim=1)     # (2,40)
    return to_undirected(e).contiguous()


# -------------------------- CSV ユーティリティ --------------------------

def _detect_frame_col(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit and explicit in df.columns: return explicit
    for c in ["frame", "name", "basename", "base", "id"]:
        if c in df.columns: return c
    raise KeyError("frame列が見つかりません（--frame-col で指定してください）")


def _detect_xy_style(df: pd.DataFrame, style: str = "auto") -> Tuple[list, list]:
    cols = list(df.columns)

    def has_all(prefix: str, sep: str) -> bool:
        return all((f"{prefix}{sep}{i}" if sep else f"{prefix}{i}") in cols for i in range(21))

    def has_all_idxfirst() -> bool:
        return all(f"{i}_x" in cols and f"{i}_y" in cols for i in range(21))

    if style == "x0":
        if has_all('x', '') and has_all('y',''):
            return [f"x{i}" for i in range(21)], [f"y{i}" for i in range(21)]
        raise KeyError("x0..x20 / y0..y20 が見つかりません")

    if style == "x_0":
        if has_all('x','_') and has_all('y','_'):
            return [f"x_{i}" for i in range(21)], [f"y_{i}" for i in range(21)]
        raise KeyError("x_0..x_20 / y_0..y_20 が見つかりません")

    if style == "0_x":
        if has_all_idxfirst():
            return [f"{i}_x" for i in range(21)], [f"{i}_y" for i in range(21)]
        raise KeyError("0_x..20_x / 0_y..20_y が見つかりません")

    # auto 検出（優先度: 0_x → x0 → x_0）
    if has_all_idxfirst():
        return [f"{i}_x" for i in range(21)], [f"{i}_y" for i in range(21)]
    if has_all('x','') and has_all('y',''):
        return [f"x{i}" for i in range(21)], [f"y{i}" for i in range(21)]
    if has_all('x','_') and has_all('y','_'):
        return [f"x_{i}" for i in range(21)], [f"y_{i}" for i in range(21)]

    raise KeyError("x/y 列が見つかりません（x0..x20 / x_0..x_20 / 0_x..20_x のいずれかが必要）")

    if style == "x0":
        if has_all('x', '') and has_all('y',''): return [f"x{i}" for i in range(21)], [f"y{i}" for i in range(21)]
        raise KeyError("x0..x20 / y0..y20 が見つかりません")
    if style == "x_0":
        if has_all('x','_') and has_all('y','_'): return [f"x_{i}" for i in range(21)], [f"y_{i}" for i in range(21)]
        raise KeyError("x_0..x_20 / y_0..y_20 が見つかりません")

    # auto
    if has_all('x', '') and has_all('y',''):
        return [f"x{i}" for i in range(21)], [f"y{i}" for i in range(21)]
    if has_all('x','_') and has_all('y','_'):
        return [f"x_{i}" for i in range(21)], [f"y_{i}" for i in range(21)]
    raise KeyError("x/y 列（x0..x20 もしくは x_0..x_20）が見つかりません")


_frame_regex = re.compile(
    r"^(?P<prefix>.+?)_(?P<idx>\d+)(?:_(?P<hand>[01]))?$"
)

def _parse_frame_fields(s: str) -> Tuple[str, int, Optional[int]]:
    """
    例:
      'Bumblebee.mp4_21_0' → (prefix='Bumblebee.mp4', idx=21, hand=0)
      'Song_005_1'         → (prefix='Song', idx=5, hand=1)
      'f123'               → (prefix='f', idx=123, hand=None)
    """
    m = _frame_regex.match(str(s))
    if m:
        return m.group('prefix'), int(m.group('idx')), (int(m.group('hand')) if m.group('hand') is not None else None)
    # フォールバック: 末尾の数字をフレーム番号に解釈
    m2 = re.search(r"(.*?)(\d+)$", str(s))
    if m2:
        return m2.group(1), int(m2.group(2)), None
    return str(s), 0, None


# -------------------------- 1曲→list[Data] 生成 --------------------------

def build_graphs_for_song(csv_path: Path, song_name: Optional[str] = None, frame_col: Optional[str] = None,
                           xy_style: str = "auto", allow_single_hand: bool = False, skip_nan: bool = False,
                           verbose: bool = False) -> List[Data]:
    df = pd.read_csv(csv_path)
    fcol = _detect_frame_col(df, frame_col)
    xcols, ycols = _detect_xy_style(df, style=xy_style)

    # NaN への対処
    if skip_nan:
        df = df.dropna(subset=xcols + ycols)
    else:
        df[xcols + ycols] = df[xcols + ycols].fillna(0.0)

    # frame の分解
    recs = []
    for i, row in df.iterrows():
        base, idx, hand = _parse_frame_fields(row[fcol])
        x = row[xcols].to_numpy(dtype=float, copy=True)
        y = row[ycols].to_numpy(dtype=float, copy=True)
        pts = np.stack([x, y], axis=1)  # (21,2)
        recs.append({"base": base, "t": int(idx), "hand": (int(hand) if hand is not None else None), "pts": pts})

    # base×t ごとに手0/手1 を束ねる
    # 注意: hand==0/1 は MediaPipe の検出順で左右とは限らない
    buckets: dict[Tuple[str,int], dict[int, np.ndarray]] = {}
    for r in recs:
        key = (r["base"], r["t"])  # 1フレーム
        if key not in buckets:
            buckets[key] = {}
        h = r["hand"] if r["hand"] in (0,1) else (0 if 0 not in buckets[key] else 1 if 1 not in buckets[key] else 0)
        buckets[key][h] = r["pts"]

    # エッジ固定
    edges = make_edges()

    # Data を構築
    data_list: List[Data] = []
    song = song_name or csv_path.stem
    sid = 0  # 単曲なので 0 を付ける

    for (base, t), hands in sorted(buckets.items(), key=lambda kv: kv[0][1]):
        # 両手
        if 0 in hands and 1 in hands:
            nodes = np.vstack([hands[0], hands[1]])  # (42,2)
            hand_mask = np.array([1, 1], dtype=np.int64)
        elif allow_single_hand and (0 in hands or 1 in hands):
            if 0 in hands:
                nodes = np.vstack([hands[0], np.zeros((21,2), dtype=float)])
                hand_mask = np.array([1, 0], dtype=np.int64)
            else:
                nodes = np.vstack([np.zeros((21,2), dtype=float), hands[1]])
                hand_mask = np.array([0, 1], dtype=np.int64)
        else:
            if verbose:
                print(f"[skip] need both hands for t={t}, {base}got={list(hands.keys())}")
            continue

        x = torch.from_numpy(nodes.astype(np.float32))   # (42,2)
        d = Data(x=x, edge_index=edges.clone(), t=torch.tensor([int(t)], dtype=torch.long))
        d.hand_mask = torch.from_numpy(hand_mask)
        d.sid = torch.tensor([sid], dtype=torch.long)
        d.song = str(song)
        data_list.append(d)

    # t で昇順ソート（二重防御）
    data_list.sort(key=lambda d: int(d.t.item()))

    if verbose:
        n_in = len(set((b,t) for (b,t) in buckets.keys()))
        n_out = len(data_list)
        print(f"[build] song={song} frames(in/out)={n_in}/{n_out} single_ok={allow_single_hand}")

    return data_list


# -------------------------- CLI --------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="1曲CSV→推論用 list[Data] 生成")
    ap.add_argument("--csv", type=Path, required=True, help="MediaPipe手骨格のCSV（単曲）")
    ap.add_argument("--out", type=Path, required=True, help="出力 .pt（torch.save(list[Data])）")
    ap.add_argument("--song-name", type=str, default=None, help="Data.song に入れる名前（省略時はCSVのstem）")
    ap.add_argument("--frame-col", type=str, default=None, help="フレーム列名（自動検出に失敗する場合に指定）")
    ap.add_argument("--xy-style", type=str, choices=["auto","x0","x_0"], default="auto", help="x/y列名のスタイル")
    ap.add_argument("--allow-single-hand", action="store_true", help="片手しか無いフレームも残す（もう片手は0埋め）")
    ap.add_argument("--skip-nan", action="store_true", help="NaN を含む手をスキップ（デフォは0埋め）")
    ap.add_argument("--manifest", type=Path, default=None, help="生成サマリをJSONで保存")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    data_list = build_graphs_for_song(
        csv_path=args.csv,
        song_name=args.song_name,
        frame_col=args.frame_col,
        xy_style=args.xy_style,
        allow_single_hand=args.allow_single_hand,
        skip_nan=args.skip_nan,
        verbose=args.verbose,
    )

    if len(data_list) == 0:
        raise RuntimeError("出力が0件です。CSVの中身や --frame-col / --xy-style / --allow-single-hand を確認してください。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, args.out)

    if args.manifest is not None:
        manifest = {
            "csv": str(args.csv),
            "out": str(args.out),
            "n_frames": len(data_list),
            "song": (args.song_name or Path(args.csv).stem),
            "allow_single_hand": bool(args.allow_single_hand),
            "xy_style": args.xy_style,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[save] {args.out}  (n_frames={len(data_list)})")


if __name__ == "__main__":
    main()
