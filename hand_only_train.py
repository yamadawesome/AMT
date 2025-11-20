# hand_only_train.py
"""
python3 hand_only_train.py \
  --pt ./dataset/all_graphs.pt \
  --epochs 100 \
  --batch-size 32 \
  --T 5 \
  --stride 1 \
  --save ./checkpoints/hand_only_ep100.ckpt
"""
import os, math, random, torch
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data, Batch
from hand_only_net import HandOnlyNet, fit_hand_only, make_pos_weight

class HandGraphWindowDataset(Dataset):

    def __init__(self, pt_path: str, T: int = 9, stride: int = 1):
        assert T % 2 == 1, "T は奇数"
        self.T = T
        self.h = T // 2

        raw = torch.load(pt_path, map_location="cpu")  # ← FutureWarning はこの用途なら無視してOK
        assert isinstance(raw, (list, tuple)) and len(raw) > 0, "pt の中身が空です"
        # 1) 形式チェック：List[Data] か？
        from torch_geometric.data import Data
        if not all(isinstance(d, Data) for d in raw):
            raise TypeError("この Dataset は List[Data] 形式の all_graphs.pt を期待します")

        # 2) sid（曲単位）にグループ化
        groups = {}
        for d in raw:
            sid = int(d.sid.item()) if hasattr(d, "sid") else 0
            groups.setdefault(sid, []).append(d)

        # 3) 各グループ内を t でソート
        for sid, arr in groups.items():
            arr.sort(key=lambda z: int(z.t.item()))

        # 4) ウィンドウ化して index を構築
        self.groups = groups  # {sid: [Data,...]}
        self.index = []       # List[(sid, center_idx)]
        for sid, arr in self.groups.items():
            N = len(arr)
            if N < T:  # 足りない曲はスキップ
                continue
            for c in range(self.h, N - self.h, stride):
                self.index.append((sid, c))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        sid, c = self.index[i]
        arr = self.groups[sid]
        sl = slice(c - self.h, c + self.h + 1)   # 長さT
        graph_seq = arr[sl]
        y_center = arr[c].y.float()              # (88,)
        return graph_seq, y_center

def collate_windows(batch):
    """
    batch: List[(graph_seq: List[Data], y: Tensor(88,))]
    → graph_seq_batched: List[Batch] (長さT), y: (B,88)
    """
    T = len(batch[0][0])
    # 時間ごとにまとめて Batch 化
    out_seq: List[Batch] = []
    for t in range(T):
        data_list_t = [sample[0][t] for sample in batch]
        out_seq.append(Batch.from_data_list(data_list_t))
    y = torch.stack([sample[1] for sample in batch], dim=0).float()
    return out_seq, y


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", default="./dataset/all_graphs.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=9)       # 窓幅（奇数）: 7/9/11 など
    parser.add_argument("--stride", type=int, default=1)  # サンプル間の間引き
    parser.add_argument("--save", default="./checkpoints/hand_only.ckpt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-dir", default="./logs/hand_only", help="CSV と TensorBoard を保存するディレクトリ")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = HandGraphWindowDataset(args.pt, T=args.T, stride=args.stride)
    # シンプルにランダム分割（シーケンス単位で分けたい場合は pt 作成側でまとめておく）
    n_total = len(ds)
    n_train = int(n_total * 0.8)
    n_val   = n_total - n_train
    g = torch.Generator().manual_seed(2025)
    ds_tr, ds_va = random_split(ds, [n_train, n_val], generator=g)

    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True,
                           collate_fn=collate_windows)
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True,
                           collate_fn=collate_windows)

    # モデル
    model = HandOnlyNet(num_keys=88, node_feat_dim=2, gcn_dim=128, proj_dim=256,
                        t_layers=2, t_kernel=3, dropout=0.1)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    model = fit_hand_only(
        model,
        train_loader=loader_tr,
        val_loader=loader_va,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        softf1_w=0.2,
        es_patience=None,     
        save_path=args.save,
        log_csv_path=os.path.join(log_dir, "metrics.csv"),
        tb_log_dir=os.path.join(log_dir, "tb")
    )
        

if __name__ == "__main__":
    main()