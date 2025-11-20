#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_graphs_from_csv.py

MediaPipeの手CSV（*_processed_renumbered.csv）から、
ラベル無しの PyG Data リスト（推論用）を .pt に保存する。

- 各フレーム: 左右21点=計42ノード、無向40エッジ（左右は独立、手間の接続は無し）
- 片手しか無いフレームは既定ではスキップ（--allow-single-hand で保存可）
- Data には x(座標[42,2]), edge_index(2,E), song(str), frame(int), base_frame(str) を格納

Usage:
  python3 gen_graph_infer.py \
    --csv ../../MIDI_TEST_SET/miditest_videos/5/5.csv \
    --out .../../MIDI_TEST_SET/miditest_videos/5/5_graphs.pt \
    --allow-single-hand \
    --frame-col file_name \
    --verbose
"""

import os
import argparse
from typing import List

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# 片手21ノードのエッジ20本（MediaPipe Hands想定）
BASE_EDGES = torch.tensor(
    [[0, 1], [1, 2], [2, 3], [3, 4],
     [9, 5], [5, 6], [6, 7], [7, 8],
     [0, 9], [9,10], [10,11], [11,12],
     [9,13], [13,14], [14,15], [15,16],
     [13,17], [17,18], [18,19], [19,20]],
    dtype=torch.long).t().contiguous()   # shape: (2, 20)

def build_edge_index(connect_hands: bool = False) -> torch.Tensor:
    """左右それぞれ21ノードを同型に張り、必要なら左右を接続する（既定:非接続）。"""
    ei = torch.cat([BASE_EDGES, BASE_EDGES + 21], dim=1)  # (2, 40)
    if connect_hands:
        cross = torch.tensor([[0, 21]], dtype=torch.long).t().contiguous()  # 手首同士
        ei = torch.cat([ei, cross], dim=1)
    return to_undirected(ei)

def load_song_as_graphs_unlabeled(csv_path: str,
                                  allow_single_hand: bool = False,
                                  connect_hands: bool = False,
                                  verbose: bool = False) -> List[Data]:
    """単一CSV → list[Data]（フレーム昇順, ラベル無し）"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "frame" not in df.columns:
        raise ValueError("CSVに 'frame' 列が見つかりません。")

    # 例: "Bumblebee.mp4_21_0" → base_frame="Bumblebee.mp4_21", frame_idx=21
    df["base_frame"] = df["frame"].str.rsplit("_", n=1).str[0]
    df["frame_idx"] = df["base_frame"].str.rsplit("_", n=1).str[-1].astype(int)

    song = os.path.basename(csv_path).split("_")[0]
    edge_index = build_edge_index(connect_hands=connect_hands)

    # 必要列の存在チェック（座標: i_x, i_y）
    for i in range(21):
        for ax in ("x", "y"):
            col = f"{i}_{ax}"
            if col not in df.columns:
                raise ValueError(f"CSVに '{col}' 列が見つかりません。")

    # フレーム昇順で処理
    data_list: List[Data] = []
    for _, grp in df.sort_values("frame_idx").groupby("base_frame", sort=False):
        if len(grp) != 2:
            if not allow_single_hand:
                continue
            # 片手のみ：そのまま片手21点 + もう一方は0埋め
            hand = grp.iloc[0]
            left_coords = [[hand[f"{i}_x"], hand[f"{i}_y"]] for i in range(21)]
            right_coords = [[0.0, 0.0] for _ in range(21)]
            coords = left_coords + right_coords
            frame_idx = int(grp["frame_idx"].iloc[0])
            base_key = str(grp["base_frame"].iloc[0])
        else:
            # 親指先端(4_x)で左右判定（xが小さい方＝左）
            h0, h1 = grp.iloc[0], grp.iloc[1]
            left, right = (h0, h1) if h0["4_x"] < h1["4_x"] else (h1, h0)
            coords = [[left[f"{i}_x"],  left[f"{i}_y"]]  for i in range(21)] + \
                     [[right[f"{i}_x"], right[f"{i}_y"]] for i in range(21)]
            frame_idx = int(grp["frame_idx"].iloc[0])
            base_key = str(grp["base_frame"].iloc[0])

        x = torch.tensor(coords, dtype=torch.float32)  # (42, 2)

        data = Data(x=x, edge_index=edge_index, song=song, frame=frame_idx, base_frame=base_key)
        if verbose:
            print(f"{song}: frame={frame_idx:6d} nodes={x.size(0)} edges={edge_index.size(1)}")
        data_list.append(data)

    return data_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="1曲ぶんの *_processed_renumbered.csv")
    ap.add_argument("--out", required=True, help="出力 .pt (list[Data])")
    ap.add_argument("--allow-single-hand", action="store_true",
                    help="片手フレームも保存（もう一方は0埋め）")
    ap.add_argument("--connect-hands", action="store_true",
                    help="左右の手首同士を1本エッジで接続（任意）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_list = load_song_as_graphs_unlabeled(
        args.csv,
        allow_single_hand=args.allow_single_hand,
        connect_hands=args.connect_hands,
        verbose=args.verbose
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(data_list, args.out)

    print(f"✅ 完了: {len(data_list)} サンプルを {args.out} に保存しました")
    if len(data_list) == 0:
        print("⚠️ 有効フレームが0です。CSVの 'frame' 整合・座標列の有無・両手2行揃いを確認してください。")

if __name__ == "__main__":
    main()