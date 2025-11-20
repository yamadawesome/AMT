#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

- 入力: torch.save された list[Data]（1フレーム=1Data）
- 出力: (T_pred, 88) の確率（窓幅 T_seq と stride に応じて T_pred ≈ N-T_seq+1）

  nohup python3 infer_graph_probs.py > graph_27.log \
    --graph-pt /home/yamada/MIDI_TEST_SET/miditest_videos/27/27_graphs.pt \
    --state-dict ./checkpoints/hand_only_ep100.ckpt \
    --model-def ./model_def.py \
    --out-npy /home/yamada/MIDI_TEST_SET/miditest_videos/27/27.npy \
    --align center --stride 1 --device cuda &
"""
from __future__ import annotations
import os, types, json, argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Data, Batch

# ========================= 基本 I/O =========================

def _load_graph_frames(pt_path: Path, map_location: str = "cpu") -> List[Data]:
    obj = torch.load(pt_path, map_location=map_location)
    if isinstance(obj, list) and all(isinstance(x, Data) for x in obj):
        data_list = obj
    elif isinstance(obj, dict):
        for k in ["data_list", "frames", "items", "datas", "list"]:
            if k in obj and isinstance(obj[k], list) and all(isinstance(x, Data) for x in obj[k]):
                data_list = obj[k]
                break
        else:
            raise ValueError(f"未知の dict 形式: keys={list(obj.keys())}")
    else:
        raise TypeError(f"サポート外の形式: {type(obj)}")

    # t/idx/… で時間順に並べ替え
    def _t(d: Data, i: int) -> int:
        for key in ["frame_idx","frame","t","index","idx"]:
            if hasattr(d, key):
                v = getattr(d, key)
                if isinstance(v, torch.Tensor): v = int(v.item())
                return int(v)
        return i

    order = sorted(range(len(data_list)), key=lambda i: _t(data_list[i], i))
    data_list = [data_list[i] for i in order]
    return data_list


def _auto_to_probs(logits: torch.Tensor, channel: Optional[int]) -> torch.Tensor:
    x = logits
    if x.ndim == 3:
        if x.shape[1] == 88:
            if channel is None: channel = 0
            x = x[:, :, channel]
        elif x.shape[2] == 88:
            if channel is None: channel = 0
            x = x[:, channel, :]
        else:
            raise ValueError(f"出力形状不明: {tuple(x.shape)}")
    elif x.ndim == 2:
        if x.shape[1] != 88:
            raise ValueError(f"出力次元が88ではない: {tuple(x.shape)}")
    else:
        raise ValueError(f"出力テンソル次元が想定外: {x.ndim}")

    with torch.no_grad():
        if x.min().item() < -0.05 or x.max().item() > 1.05:
            x = torch.sigmoid(x)
    return x


# ========================= モデル読み込み =========================

def _load_model_from_full(ckpt_path: Path, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device)
    meta = None
    if hasattr(obj, "to") and hasattr(obj, "eval"):
        model = obj
    elif isinstance(obj, dict) and "model" in obj and hasattr(obj["model"], "eval"):
        model = obj["model"]
        meta = obj.get("meta", None)
    else:
        raise ValueError("--ckpt-full は torch.save(model) で保存した“フルモデル”が必要です。")
    model.to(device).eval()
    return model, meta


def _load_model_from_state(state_path: Path, model_def_path: Path, device: torch.device):
    # model_def.py を動的 import
    mod = types.ModuleType("model_def")
    mod.__file__ = str(model_def_path)
    src = Path(model_def_path).read_text(encoding="utf-8")
    exec(compile(src, str(model_def_path), "exec"), mod.__dict__)
    if not hasattr(mod, "build_model"):
        raise AttributeError("model_def.py に build_model(num_classes=88) が必要です。")

    model = mod.build_model(num_classes=88)
    sd = torch.load(state_path, map_location="cpu")
    meta = None
    if isinstance(sd, dict) and "state_dict" in sd:
        meta = sd.get("meta", None)
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, meta


# ========================= 時系列ウィンドウ推論 =========================

def infer_seq(
    model: torch.nn.Module,
    frames: List[Data],
    device: torch.device,
    channel: Optional[int],
    T_seq: int,
    stride: int,
    align: str = "last",
    show_progress: bool = True,
) -> Tuple[torch.Tensor, List[int]]:
    """list[Data] を窓幅 T_seq でスライドし、list[Batch] を forward に渡して (T_pred,88) を返す。"""
    N = len(frames)
    if N < T_seq:
        raise ValueError(f"フレーム数 {N} がシーケンス長 T={T_seq} より小さいです")

    # t を取り出し（保存済み）
    def _t(d: Data, i: int) -> int:
        for key in ["frame_idx","frame","t","index","idx"]:
            if hasattr(d, key):
                v = getattr(d, key)
                if isinstance(v, torch.Tensor): v = int(v.item())
                return int(v)
        return i
    t_all = [_t(d, i) for i, d in enumerate(frames)]

    anchor = (T_seq - 1) if align == "last" else (T_seq // 2)
    starts = list(range(0, N - T_seq + 1, stride))

    probs_list: List[torch.Tensor] = []
    targets: List[int] = []

    it = enumerate(starts)
    if show_progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(starts), desc="infer(seq)")
        except Exception:
            pass

    with torch.no_grad():
        for _, s in it:
            window = frames[s: s + T_seq]
            # 各時刻をバッチ化（ここでは1グラフ/時刻なのでバッチサイズ=1）
            seq_batches = [Batch.from_data_list([d]).to(device) for d in window]
            out = model(seq_batches)  # (B,88) ここでは B=1
            prob = _auto_to_probs(out, channel=channel)  # (1,88)
            probs_list.append(prob.squeeze(0).cpu())
            targets.append(int(t_all[s + anchor]))

    # 時系列順にソート（念のため）
    ord_idx = sorted(range(len(targets)), key=lambda i: targets[i])
    probs = torch.stack([probs_list[i] for i in ord_idx], dim=0)
    targets_sorted = [targets[i] for i in ord_idx]
    return probs, targets_sorted


# ========================= CLI =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unknown graph sequence → (T,88) probs inference (sequence-aware)")
    ap.add_argument("--graph-pt", type=Path, required=True, help="torch.save(list[Data])")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt-full", type=Path, help="torch.save(model) のフルモデル")
    g.add_argument("--state-dict", type=Path, help="state_dict チェックポイント")

    ap.add_argument("--model-def", type=Path, help="state_dict 用の model_def.py（build_model 必須）")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])  
    ap.add_argument("--channel", type=str, default="onset", help="onset/duration/offset または 0/1/2")

    ap.add_argument("--T", type=int, default=None, help="シーケンス長（未指定→ckpt.meta.window_T→9）")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--align", type=str, choices=["last","center"], default="center")

    ap.add_argument("--out-npy", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-meta", type=Path, default=None)
    return ap.parse_args()


def _parse_channel(arg: str) -> Optional[int]:
    if arg is None: return 0
    if arg.isdigit(): return int(arg)
    table = {"onset":0, "duration":1, "offset":2}
    return table.get(arg.lower(), 0)


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    ch = _parse_channel(args.channel)

    # 入力読み込み
    frames = _load_graph_frames(args.graph_pt, map_location=str(device))
    if len(frames) == 0:
        raise RuntimeError("フレーム0件。--graph-pt を確認してください。")

    # モデル読み込み & window_T
    meta = None
    if args.ckpt_full is not None:
        model, meta = _load_model_from_full(args.ckpt_full, device)
    else:
        if args.model_def is None:
            raise ValueError("--state-dict を使う場合は --model-def が必要です")
        model, meta = _load_model_from_state(args.state_dict, args.model_def, device)

    T_seq = args.T if args.T is not None else ((meta or {}).get("window_T", 9) if isinstance(meta, dict) else 9)

    # 推論
    probs, t_idx = infer_seq(
        model=model,
        frames=frames,
        device=device,
        channel=ch,
        T_seq=T_seq,
        stride=args.stride,
        align=args.align,
        show_progress=True,
    )

    # 保存
    import numpy as np, pandas as pd
    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, probs.numpy().astype("float32"))

    if args.out_csv is not None:
        df = pd.DataFrame(probs.numpy(), columns=[f"k{i:02d}" for i in range(88)])
        df.insert(0, "frame", t_idx)
        df.to_csv(args.out_csv, index=False)

    if args.out_meta is not None:
        meta_out = {
            "frame_index": t_idx,
            "shape": list(probs.shape),
            "channel": ch,
            "T_seq": T_seq,
            "stride": args.stride,
            "align": args.align,
        }
        args.out_meta.parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_meta).write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[save] NPY:  {args.out_npy} shape={tuple(probs.shape)}")
    if args.out_csv:  print(f"[save] CSV:  {args.out_csv}")
    if args.out_meta: print(f"[save] JSON: {args.out_meta}")


if __name__ == "__main__":
    main()
