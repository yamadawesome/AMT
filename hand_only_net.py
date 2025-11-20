# hand_only_net.py
# 既存の late-fusion 学習コードから「音」分岐を完全に除いて、
# 手指骨格 (PyG グラフの時系列) だけで 88鍵オンセットを推定する最小実装。
# - モデル: HandOnlyNet(GCN + TemporalConv + Linear->88)
# - 入力: graph_seq (長さ T の PyG Batch のリスト), 各 Batch は B グラフを含む
# - 出力: logits (B,88)
# - 損失: BCEWithLogitsLoss (+ 任意で soft-F1)
#
# 使い方（例）:
#   from hand_only_net import HandOnlyNet, train_one_epoch, evaluate, soft_f1_loss, make_pos_weight
#   model = HandOnlyNet(num_keys=88, node_feat_dim=2, gcn_dim=128, proj_dim=256, t_layers=2, t_kernel=3)
#   pos_w = make_pos_weight(train_loader)  # (88,) テンソル
#   ... あとは train ループ内で logits = model(graph_seq); loss = BCE(logits, y) + softF1 など
#
# 注意: DataLoader はバッチごとに
#   - graph_seq: List[pyg.data.Batch] (len=T)
#   - y: Tensor (B, 88)
# を返すようにしておくこと。既存の Loader が (audio, graph_seq, y) を返す場合は audio を無視。

from __future__ import annotations
import math, os, csv
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class GCNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x


class DepthwiseTCN(nn.Module):
    """時間方向の軽量Conv。num_channelsごとに独立に畳み込む(groups=C)。
    入力: (B, C, T) -> 出力: (B, C, T)
    """
    def __init__(self, channels: int, kernel_size: int = 3, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size は奇数にしてください (中心抽出を想定)"
        pads = kernel_size // 2
        mods = []
        for _ in range(layers):
            mods += [
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pads, groups=channels, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HandOnlyNet(nn.Module):
    def __init__(
        self,
        num_keys: int = 88,
        node_feat_dim: int = 2,   # MediaPipe: (x,y) を使う前提
        gcn_dim: int = 128,
        proj_dim: int = 256,
        t_layers: int = 2,
        t_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gcn = GCNBlock(node_feat_dim, gcn_dim, dropout=dropout)
        self.proj = nn.Linear(gcn_dim, proj_dim)
        self.tcn = DepthwiseTCN(proj_dim, kernel_size=t_kernel, layers=t_layers, dropout=dropout)
        self.head = nn.Linear(proj_dim, num_keys)

    @torch.no_grad()
    def _check_seq(self, graph_seq: List[Batch]):
        assert isinstance(graph_seq, (list, tuple)) and len(graph_seq) >= 1
        for g in graph_seq:
            assert hasattr(g, 'x') and hasattr(g, 'edge_index') and hasattr(g, 'batch'), "各要素は PyG Batch である必要があります"

    def forward(self, graph_seq: List[Batch]) -> torch.Tensor:
        # graph_seq[t]: PyG Batch (B個の手グラフが入っている)
        self._check_seq(graph_seq)
        feats = []
        for g in graph_seq:
            h = self.gcn(g.x, g.edge_index)                # (sum_nodes, gcn_dim)
            h = global_mean_pool(h, g.batch)               # (B, gcn_dim)
            h = self.proj(h)                               # (B, proj_dim)
            feats.append(h)
        Z = torch.stack(feats, dim=1)                      # (B, T, proj_dim)
        Z = Z.transpose(1, 2)                              # (B, C=proj_dim, T)
        Z = self.tcn(Z)                                   # (B, C, T)
        center = Z[:, :, Z.shape[-1] // 2]                # (B, C)
        logits = self.head(center)                        # (B, 88)
        return logits


# ===== 損失・メトリクス =====
class BCEWithLogitsLossStable(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


def soft_f1_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """マクロ soft-F1 の 1 - F1 を返す（最小化用）"""
    probs = torch.sigmoid(logits)
    tp = (probs * targets).sum(dim=0)
    fp = (probs * (1 - targets)).sum(dim=0)
    fn = ((1 - probs) * targets).sum(dim=0)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)  # (88,)
    return 1.0 - f1.mean()


@torch.no_grad()
def make_pos_weight(loader) -> torch.Tensor:
    """(B,88) ラベルから pos_weight を推定。
    pos_weight[c] = (neg/pos)。極端値は [1, 20] にクリップ。
    """
    pos = None
    total = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            _, graph_seq, y = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            graph_seq, y = batch
        else:
            raise ValueError("batch は (audio, graph_seq, y) か (graph_seq, y) の想定です")
        y = y.float()
        if pos is None:
            pos = y.sum(dim=0)
        else:
            pos += y.sum(dim=0)
        total += y.shape[0]
    pos = pos.clamp(min=1.0)  # ゼロ割回避
    neg = total - pos
    pw = (neg / pos).clamp(1.0, 20.0)
    return pw


# ===== 学習/検証 ループ =====

def train_one_epoch(model: nn.Module, loader, optimizer, device, pos_weight: Optional[torch.Tensor] = None,
                    use_softf1: bool = True, softf1_w: float = 0.2, grad_clip: float = 5.0):
    model.train()
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = BCEWithLogitsLossStable(pos_weight=pos_weight)

    total_loss = 0.0
    n = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            _, graph_seq, y = batch
        else:
            graph_seq, y = batch
        y = y.to(device).float()
        graph_seq = [g.to(device) for g in graph_seq]

        optimizer.zero_grad(set_to_none=True)
        logits = model(graph_seq)
        loss = criterion(logits, y)
        if use_softf1:
            loss = loss + softf1_w * soft_f1_loss(logits, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach()) * y.size(0)
        n += y.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, pos_weight: Optional[torch.Tensor] = None, use_softf1: bool = True, softf1_w: float = 0.2):
    model.eval()
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = BCEWithLogitsLossStable(pos_weight=pos_weight)

    total_loss = 0.0
    n = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            _, graph_seq, y = batch
        else:
            graph_seq, y = batch
        y = y.to(device).float()
        graph_seq = [g.to(device) for g in graph_seq]

        logits = model(graph_seq)
        loss = criterion(logits, y)
        if use_softf1:
            loss = loss + softf1_w * soft_f1_loss(logits, y)

        total_loss += float(loss.detach()) * y.size(0)
        n += y.size(0)
    return total_loss / max(n, 1)



def fit_hand_only(model: nn.Module, train_loader, val_loader, device: torch.device,
                  epochs: int = 30, lr: float = 1e-4, weight_decay: float = 1e-4,
                  softf1_w: float = 0.2,
                  es_patience: int | None = 5,
                  min_delta: float = 1e-4,
                  save_path: str | None = None,
                  save_last: bool = True,
                  reduce_on_plateau: bool = True,
                  plateau_factor: float = 0.5,
                  plateau_patience: int = 5,
                  plateau_min_lr: float = 1e-6,
                  # ↓↓↓ 追加（ここから）
                  log_csv_path: str | None = None,
                  tb_log_dir: str | None = None):
    """
    log_csv_path: CSVに [epoch,train,val,lr,best_val,time] を1行/epochで追記
    tb_log_dir:   指定すると TensorBoard に train/val/lr を書く
    """
    model = model.to(device)
    pos_w = make_pos_weight(train_loader).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=plateau_factor, patience=plateau_patience, min_lr=plateau_min_lr
    ) if reduce_on_plateau else None)

    best_val = math.inf
    no_improve = 0

    # === ログ準備 ===
    csv_file = None
    csv_writer = None
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        new_file = not os.path.exists(log_csv_path)
        csv_file = open(log_csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if new_file:
            csv_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "best_val", "time"])

    tb = SummaryWriter(tb_log_dir) if tb_log_dir else None

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        for ep in range(1, epochs + 1):
            tr = train_one_epoch(model, train_loader, opt, device,
                                 pos_weight=pos_w, use_softf1=True, softf1_w=softf1_w)
            va = evaluate(model, val_loader, device,
                          pos_weight=pos_w, use_softf1=True, softf1_w=softf1_w)
            lr_now = float(opt.param_groups[0]["lr"])
            print(f"[ep {ep:02d}] train={tr:.4f}  val={va:.4f}  lr={lr_now:.2e}")

            # === ログ書き込み ===
            if csv_writer:
                csv_writer.writerow([ep, f"{tr:.6f}", f"{va:.6f}", f"{lr_now:.6e}",
                                     f"{min(best_val, va):.6f}", datetime.now().isoformat()])
                csv_file.flush()
            if tb:
                tb.add_scalar("Loss/train", tr, ep)
                tb.add_scalar("Loss/val", va, ep)
                tb.add_scalar("LR", lr_now, ep)

            # === ベスト更新 & 保存 ===
            improved = (va < best_val - min_delta)
            if improved:
                best_val = va
                no_improve = 0
                if save_path:
                    ckpt_obj = {
                        "state_dict": model.state_dict(),
                        "val_loss": va,
                        "epoch": ep,
                        "meta": {
                            "arch": "HandOnlyNet",
                            "num_keys": 88,
                            "node_feat_dim": 2,
                            "gcn_dim": 128,
                            "proj_dim": 256,
                            "t_layers": 2,
                            "t_kernel": 3,
                            "window_T": 9,
                        }
                    }
                    torch.save(ckpt_obj, save_path)
                    print(f"  -> saved ckpt to {save_path}")
                    infer_path = os.path.splitext(save_path)[0] + "_infer.pt"
                    torch.save({"state_dict": model.state_dict(), "meta": ckpt_obj["meta"]}, infer_path)
                    print(f"  -> saved infer weights to {infer_path}")
            else:
                no_improve += 1
                if (es_patience is not None) and (no_improve >= es_patience):
                    print("Early stop (patience reached).")
                    break

            if scheduler:
                scheduler.step(va)
    finally:
        if tb: tb.close()
        if csv_file: csv_file.close()

    if save_path and save_last:
        last_path = os.path.splitext(save_path)[0] + "_last.ckpt"
        torch.save({"state_dict": model.state_dict(), "val_loss": va, "epoch": ep}, last_path)
        print(f"  -> saved last to {last_path}")

    return model