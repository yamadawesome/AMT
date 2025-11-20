#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ディレクトリ内の CSV（MediaPipe手骨格）をまとめて 88鍵ラベル（PKL）と結合し、
PyG Data（一フレーム=1 Data）のリストを **一つの .pt** に保存するツール（クロス無し・プラグイン無し）。

特長:
  - 片手だけのフレームも保持可能（ゼロ埋め + hand_mask）
  - x/y 列名は x0..x20 / y0..y20 または x_0..x_20 / y_0..y_20 を自動検出
  - ラベルPKLは dict[int->(88,)] でも ndarray(T,88) でもOK
  - エッジは BASE_EDGES×2（左右分）を to_undirected したもののみを使用（クロス無し）
  - Data には song 名（stem）と sid（連番）を保持

使い方:
  python3 gen_graph_learning.py \
      --csv-dir ./preprocessed_csv \
      --pkl-dir ./labels \
      --out ./dataset/all_graphs.pt \
      --allow-single-hand \
      --recursive \
      --pattern "*.csv" \
      --verbose

出力:
  torch.save(list[Data], out)
  ついでに manifest.tsv（CSVとPKLの対応・生成数など）も隣に出力
"""

import os
import re
import glob
import argparse
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# ============== 片手21ノードのエッジ（20本）。ノード0を手首として想定（あなたの改変版） ==============
BASE_EDGES = torch.tensor(
    [[0, 1], [1, 2], [2, 3], [3, 4],
     [0, 5], [5, 6], [6, 7], [7, 8],
     [5, 9], [9,10], [10,11], [11,12],
     [9,13], [13,14], [14,15], [15,16],
     [13,17], [17,18], [18,19], [19,20]],
    dtype=torch.long).t().contiguous()  # (2,20)

# ============== ユーティリティ ==============

def _detect_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    """x/y 列を 0..20 の順に返す。prefix は 'x' か 'y'。
    x0..x20 / x_0..x_20 の両方をサポート。
    """
    pat1 = re.compile(rf"^{prefix}_?([0-9]{{1,2}})$")   # x0, x_0
    pat2 = re.compile(rf"^([0-9]{{1,2}})_{prefix}$")    # 0_x
    pairs = []
    for c in df.columns:
        m1 = pat1.match(c)
        m2 = pat2.match(c) if not m1 else None
        if m1:
            idx = int(m1.group(1))
        elif m2:
            idx = int(m2.group(1))
        else:
            continue
        if 0 <= idx <= 20:
            pairs.append((idx, c))
    pairs.sort(key=lambda x: x[0])
    cols = [c for _, c in pairs]
    if len(cols) != 21:
        raise ValueError(f"{prefix} 列が21本見つかりません: {len(cols)} 本")
    return cols


def _row_to_nodes(row: pd.Series, xcols: List[str], ycols: List[str]) -> np.ndarray:
    x = row[xcols].to_numpy(dtype=np.float32)
    y = row[ycols].to_numpy(dtype=np.float32)
    nodes = np.stack([x, y], axis=1)  # (21,2)
    return nodes


def _load_labels(pkl_path: str) -> Dict[int, np.ndarray]:
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        lab = {}
        for k, v in obj.items():
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.shape[0] != 88:
                raise ValueError(f"label length != 88 for key={k}: {arr.shape}")
            lab[int(k)] = arr
        return lab
    elif isinstance(obj, np.ndarray):  # (T,88)
        if obj.ndim != 2 or obj.shape[1] != 88:
            raise ValueError(f"label array shape must be (T,88), got {obj.shape}")
        return {int(i): obj[i].astype(np.float32) for i in range(obj.shape[0])}
    else:
        raise TypeError(f"Unsupported PKL type: {type(obj)}")


def make_edges() -> torch.Tensor:
    """左右手の BASE_EDGES を結合し、無向にする（クロス無し）。"""
    e1 = BASE_EDGES.clone()
    e2 = (BASE_EDGES + 21).clone()
    edges = torch.cat([e1, e2], dim=1)  # (2,40)
    return to_undirected(edges).contiguous()


# ============== 1曲を list[Data] に変換 ==============

def build_graphs_for_csv(csv_path: str, pkl_path: str, edges: torch.Tensor, allow_single_hand: bool = False,
                          verbose: bool = False, song_name: Optional[str] = None, sid: Optional[int] = None) -> List[Data]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL not found: {pkl_path}")

    df = pd.read_csv(csv_path)
    labels = _load_labels(pkl_path)

    if 'frame' not in df.columns:
        raise ValueError("CSV に 'frame' 列が必要です (例: Bumblebee.mp4_21_0)")

    # frame キーを分解
    df['base_frame'] = df['frame'].str.rsplit('_', n=1).str[0]
    df['det_idx'] = df['frame'].str.rsplit('_', n=1).str[-1].astype(int)
    df['frame_idx'] = df['base_frame'].str.rsplit('_', n=1).str[-1].astype(int)

    xcols = _detect_cols(df, 'x')
    ycols = _detect_cols(df, 'y')

    if song_name is None:
        song_name = os.path.splitext(os.path.basename(csv_path))[0]
    data_list: List[Data] = []

    for base, g in df.groupby('base_frame'):
        g = g.sort_values('det_idx')
        t = int(g['frame_idx'].iloc[0])
        if t not in labels:
            if verbose:
                print(f"[skip] label missing for t={t} (song={song_name})")
            continue
        y = torch.from_numpy(labels[t].astype(np.float32))  # (88,)

        # 2行 (=両手) が理想
        if len(g) >= 2:
            rows = g.iloc[:2]  # 検出順 0,1
            nodes_l = _row_to_nodes(rows.iloc[0], xcols, ycols)
            nodes_r = _row_to_nodes(rows.iloc[1], xcols, ycols)
            nodes = np.concatenate([nodes_l, nodes_r], axis=0)
            hand_mask = np.ones((42,), dtype=np.float32)
        elif len(g) == 1 and allow_single_hand:
            nodes_l = _row_to_nodes(g.iloc[0], xcols, ycols)
            pad = np.zeros_like(nodes_l)
            nodes = np.concatenate([nodes_l, pad], axis=0)
            hand_mask = np.concatenate([np.ones((21,), dtype=np.float32), np.zeros((21,), dtype=np.float32)])
            if verbose:
                print(f"[pad] single hand at t={t} (song={song_name})")
        else:
            if verbose:
                print(f"[skip] need 2 rows for t={t}, got {len(g)} (song={song_name})")
            continue

        x = torch.from_numpy(nodes.astype(np.float32))       # (42,2)
        hm = torch.from_numpy(hand_mask)
        d = Data(x=x, edge_index=edges, y=y, t=torch.tensor([t], dtype=torch.long))
        d.hand_mask = hm
        d.sid = torch.tensor([sid if sid is not None else -1], dtype=torch.long)
        d.song = song_name
        data_list.append(d)

    data_list.sort(key=lambda d: int(d.t.item()))

    if verbose:
        print(f"[build] song={song_name}  frames(in/out)={df['base_frame'].nunique()}/{len(data_list)}  single_ok={allow_single_hand}")
    return data_list


# ============== ディレクトリを走査して一括生成 ==============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', required=True, help='MediaPipe CSV が入ったディレクトリ')
    ap.add_argument('--pkl-dir', default=None, help='ラベルPKLのディレクトリ（省略時は csv-dir と同じ）')
    ap.add_argument('--pattern', default='*.csv', help='CSV のグロブパターン（デフォルト: *.csv）')
    ap.add_argument('--recursive', action='store_true', help='再帰的に探索する')
    ap.add_argument('--out', required=True, help='出力 .pt（全曲まとめた list[Data]）')

    # フレームの扱い
    ap.add_argument('--allow-single-hand', action='store_true', help='片手のみのフレームも残す')

    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    csv_dir = args.csv_dir
    pkl_dir = args.pkl_dir or csv_dir

    # エッジ確定（固定: クロス無し）
    edges = make_edges()

    # ディレクトリ走査
    pattern = '**/' + args.pattern if args.recursive else args.pattern
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, pattern), recursive=args.recursive))
    csv_paths = [p for p in csv_paths if os.path.isfile(p)]
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"CSV が見つかりません: dir={csv_dir}, pattern={args.pattern}, recursive={args.recursive}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    manifest_path = os.path.splitext(args.out)[0] + '.manifest.tsv'
    man_rows: List[Tuple[str, str, int]] = []  # (csv, pkl, n_out)

    all_data: List[Data] = []
    sid = 0

    for csv_path in csv_paths:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        cand = os.path.join(pkl_dir, stem + '.pkl')  # 同 stem の .pkl を想定
        if not os.path.exists(cand):
            if args.verbose:
                print(f"[warn] PKL not found for stem={stem}: expect {cand}")
            continue

        one = build_graphs_for_csv(csv_path, cand, edges,
                                   allow_single_hand=args.allow_single_hand,
                                   verbose=args.verbose,
                                   song_name=stem,
                                   sid=sid)
        all_data.extend(one)
        man_rows.append((csv_path, cand, len(one)))
        sid += 1

    # 保存
    torch.save(all_data, args.out)

    # マニフェスト出力
    if len(man_rows) > 0:
        df = pd.DataFrame(man_rows, columns=['csv', 'pkl', 'n_out'])
        df.to_csv(manifest_path, sep='\t', index=False)

    print(f"✅ 完了: 合計 {len(all_data)} サンプルを {args.out} に保存しました")
    print(f"   manifest: {manifest_path}")


if __name__ == '__main__':
    main()
