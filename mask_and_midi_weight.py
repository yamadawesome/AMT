#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_and_midi_weight.py 

目的:
  - Omnizart確率と手モデル確率を時間/ピッチ整合→平滑→重み付け (従来機能)
  - さらに Onset について「救出＆抑圧」の非対称ルールを適用して
    * TPを rescue で維持
    * FPを suppress で削減
  - Omnizart API で MIDI 化

  python3 mask_and_midi_weight.py \
  --onset ../audio_10/onset_prob_88.npy \
  --duration ../audio_10/duration_prob_88.npy \
  --offset ../audio_10/offset_prob_88.npy \
  --t-unit 0.01 \
  --hand-probs ../audio_10/10.npy \
  --hand-hop 0.04 \
  --fusion-hop 0.04 \
  --onset-invert yes \
  --weight-floor 1 \
  --weight-gamma 1 \
  --a-logit 1.0 --b-logit 0.0 --c-logit 0.0 \
  --tau-hand 1 --tau-fp 0 \
  --onset-th 1.5 --dura-th 1.5 \
  --mode note --normalize \
  --out-piece ../audio_10/fused.npy \
  --out-midi  ../audio_10/fused.mid  

"""

from __future__ import annotations
import os, argparse
import numpy as np
from typing import Optional

try:
    from omnizart.music import inference as mz_inf
except Exception:
    mz_inf = None

EPS = 1e-6

# ------------------------- helpers -------------------------

def _load_2d(path: str) -> np.ndarray:
    x = np.load(path)
    if x.ndim != 2 or x.shape[1] != 88:
        raise ValueError(f"{path}: expected shape (T,88), got {x.shape}")
    return x.astype(np.float32)

def _load_optional_2d(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None: return None
    x = np.load(path)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim == 2:
        return x.astype(np.float32)
    raise ValueError(f"{path}: expected (T,) or (T,1) or (T,88)")

def clip01(x): return np.clip(x, EPS, 1-EPS)
def logit(p):  p=clip01(p); return np.log(p) - np.log(1-p)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def _pool_time(x: np.ndarray, k: int, mode: str) -> np.ndarray:
    assert k >= 1
    T, P = x.shape
    n = int(np.ceil(T / k))
    pad = n * k - T
    if pad > 0:
        x = np.pad(x, ((0, pad), (0, 0)), mode='edge')
    xb = x.reshape(n, k, P)
    if mode == 'max':  return xb.max(axis=1)
    if mode == 'mean': return xb.mean(axis=1)
    raise ValueError("mode must be 'max' or 'mean'")

def resample_time_linear(x: np.ndarray, hop_src: float, hop_tgt: float, T_ref: Optional[int] = None) -> np.ndarray:
    T_src, P = x.shape
    T_tgt = int(round(T_src * hop_src / hop_tgt)) if T_ref is None else int(T_ref)
    if T_tgt <= 1:
        return np.repeat(x[:1], T_tgt, axis=0)
    t_src = np.arange(T_src, dtype=np.float64) * hop_src
    t_tgt = np.arange(T_tgt, dtype=np.float64) * hop_tgt
    out = np.empty((T_tgt, P), dtype=np.float32)
    for p in range(P):
        out[:, p] = np.interp(t_tgt, t_src, x[:, p], left=x[0, p], right=x[-1, p])
    return out

def _moving_avg_1d(vec: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return vec
    pad = win // 2
    v = np.pad(vec, (pad, pad), mode='edge')
    ker = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(v, ker, mode='valid')

def smooth_time(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x
    T, P = x.shape
    out = np.empty_like(x)
    for p in range(P):
        out[:, p] = _moving_avg_1d(x[:, p], win)
    return out

def blur_pitch(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x
    T, P = x.shape
    out = np.empty_like(x)
    pad = win // 2
    ker = np.ones(win, dtype=np.float32) / float(win)
    for t in range(T):
        v = np.pad(x[t], (pad, pad), mode='edge')
        out[t] = np.convolve(v, ker, mode='valid')
    return out

def _zscore(u: np.ndarray) -> np.ndarray:
    mu, sd = float(u.mean()), float(u.std())
    if sd < 1e-8: return u*0.0
    return (u - mu) / sd

def compute_best_lag(a: np.ndarray, b: np.ndarray, lag_max: int) -> int:
    assert a.shape == b.shape
    T = a.shape[0]
    aa = _zscore(a.mean(axis=1))
    bb = _zscore(b.mean(axis=1))
    best, bestlag = -1e18, 0
    for lag in range(-lag_max, lag_max+1):
        if lag >= 0:
            u, v = aa[lag:], bb[:T-lag]
        else:
            u, v = aa[:T+lag], bb[-lag:]
        if len(u) <= 1: continue
        s = float((u*v).sum())
        if s > best:
            best, bestlag = s, lag
    return bestlag

def shift_with_zeros(x: np.ndarray, lag: int) -> np.ndarray:
    T, P = x.shape
    if lag > 0:
        pad = np.zeros((lag, P), dtype=x.dtype)
        y = np.vstack([pad, x])[:T]
    elif lag < 0:
        pad = np.zeros((-lag, P), dtype=x.dtype)
        y = np.vstack([x, pad])[-T:]
    else:
        y = x
    return y

def build_weight_from_hand(hand: np.ndarray, time_smooth_w: int, pitch_blur_w: int, gamma: float, floor: float) -> np.ndarray:
    h = np.clip(hand, 0.0, 1.0).astype(np.float32)
    if time_smooth_w > 1: h = smooth_time(h, time_smooth_w)
    if pitch_blur_w > 1:  h = blur_pitch(h, pitch_blur_w)
    if gamma != 1.0:      h = np.power(np.clip(h, 0.0, 1.0), float(gamma), dtype=np.float32)
    if floor > 0.0:       h = np.maximum(h, float(floor))
    return np.clip(h, 0.0, 1.0)

def _log_stats(name: str, x: np.ndarray):
    print(f"[stat] {name:14s} shape={tuple(x.shape)} min={float(x.min()):.4f} max={float(x.max()):.4f} mean={float(x.mean()):.4f}")

# ------------------------- rescue/suppress core (onset) -------------------------

def max_filter_time(x: np.ndarray, win: int = 3) -> np.ndarray:
    pad = win // 2
    xpad = np.pad(x, ((pad, pad), (0, 0)), mode='edge')
    out = np.empty_like(x)
    for t in range(out.shape[0]):
        out[t] = np.max(xpad[t:t+win], axis=0)
    return out

def shift_pitch(x: np.ndarray, d: int) -> np.ndarray:
    if d == 0: return x
    if d > 0:
        return np.pad(x[:, :-d], ((0,0),(d,0)), mode='constant', constant_values=0.0)
    d = -d
    return np.pad(x[:, d:], ((0,0),(0,d)), mode='constant', constant_values=0.0)

def fuse_onset_rescue_suppress(
    p_o, p_h, key_mask=None, conf=None,
    a=1.0, b=1.0, c=0.0,
    theta_o=0.50, theta_hard=0.85, theta_rescue=0.35,
    tau_hand=0.60, tau_fp=0.40,
    bonus_peak=0.35, bonus_neighbor=0.15,
    conf_lo=0.30, conf_hi=0.80,
):
    T, K = p_o.shape
    p_o = clip01(p_o); p_h = clip01(p_h)
    z = a*logit(p_o) + b*logit(p_h) + c

    # anchors: Omnizartの局所ピーク
    peaks = (p_o >= max_filter_time(p_o, win=3))
    z = z + bonus_peak * peaks.astype(z.dtype)

    # neighbor support from hand (±1鍵が強ければ加点)
    neigh = np.maximum(shift_pitch(p_h, +1), shift_pitch(p_h, -1))
    z = z + bonus_neighbor * (neigh > tau_hand).astype(z.dtype)

    # key mask
    if key_mask is not None:
        z = np.where(key_mask > 0, z, z - 2.0)

    # conf
    if conf is not None:
        conf = conf.reshape(T, 1)
        z = np.where(conf < conf_lo, z - 0.7, z)
        z = np.where(conf > conf_hi, z + 0.2, z)

    p_f = sigmoid(z)

    # 初期判定: Omnizart基準
    dec = (p_o >= theta_o).astype(np.uint8)

    # rescue: 救出帯 & 手強め & 鍵域OK & confOK
    ok_hand = (p_h >= tau_hand)
    ok_key  = (key_mask > 0) if key_mask is not None else np.ones_like(dec, dtype=bool)
    ok_conf = (conf.reshape(T,1) >= conf_lo) if conf is not None else np.ones_like(dec, dtype=bool)
    rescue_band = (p_o >= theta_rescue) & (p_o < theta_o)
    rescue = rescue_band & ok_hand & ok_key & ok_conf
    dec = np.where(rescue, 1, dec)

    # suppress: 高いが手が弱い/鍵域外/conf低 → ただし Omni 非常に高いのは保護
    weak_hand = (p_h < tau_fp)
    bad_conf  = (conf.reshape(T,1) < conf_lo) if conf is not None else np.zeros_like(dec, dtype=bool)
    suppress = (p_o >= theta_o) & (weak_hand | (~ok_key) | bad_conf)
    protect  = (p_o >= theta_hard)
    dec = np.where(suppress & (~protect), 0, dec)

    return p_f, dec

# ------------------------- CLI / main -------------------------

def main():
    ap = argparse.ArgumentParser(description="音響×手 融合 + Onset救出/抑圧 → OmnizartでMIDI化")

    # 入力
    ap.add_argument('--onset', required=True); ap.add_argument('--duration', required=True); ap.add_argument('--offset', required=True)
    ap.add_argument('--t-unit', type=float, required=True)

    ap.add_argument('--hand-probs', required=True, help='手モデル (T_aux,88) onset系の確率')
    ap.add_argument('--hand-hop', type=float, default=0.04)

    ap.add_argument('--fusion-hop', type=float, default=0.04)
    ap.add_argument('--lag-max', type=int, default=12)

    ap.add_argument('--onset-invert', choices=['auto','yes','no'], default='auto')

    # 重み W （連続値）関連（従来機能）
    ap.add_argument('--time-smooth', type=int, default=3)
    ap.add_argument('--pitch-blur', type=int, default=1)
    ap.add_argument('--weight-gamma', type=float, default=1.0)
    ap.add_argument('--weight-floor', type=float, default=0.0)
    ap.add_argument('--apply-to', choices=['onset_dura','all'], default='onset_dura')
    ap.add_argument('--tail-mode', choices=['edge','audio_only'], default='edge')

    # 救出＆抑圧（有効化で適用）
    ap.add_argument('--rescue-mode', choices=['enable','disable'], default='disable')
    ap.add_argument('--key-mask', type=str, default=None, help='(T,88) in {0,1}')
    ap.add_argument('--conf', type=str, default=None, help='(T,) or (T,1) in [0,1]')

    # 閾値・係数
    ap.add_argument('--onset-th', type=float, default=0.50)   # 基準Omni閾値
    ap.add_argument('--dura-th', type=float, default=1.0)     # Omnizart側のdura_th (normalize前提)
    ap.add_argument('--theta-hard', type=float, default=0.7)
    ap.add_argument('--theta-rescue', type=float, default=0.2)
    ap.add_argument('--tau-hand', type=float, default=0.60)
    ap.add_argument('--tau-fp', type=float, default=0.40)
    ap.add_argument('--bonus-peak', type=float, default=0)
    ap.add_argument('--bonus-neighbor', type=float, default=0)
    ap.add_argument('--conf-lo', type=float, default=0.30)
    ap.add_argument('--conf-hi', type=float, default=0.80)

    # ロジット線形係数
    ap.add_argument('--a-logit', type=float, default=0.9)
    ap.add_argument('--b-logit', type=float, default=1.2)
    ap.add_argument('--c-logit', type=float, default=0.08)

    # Omnizart呼び出し
    ap.add_argument('--mode', choices=['note','note-stream'], default='note')
    ap.add_argument('--normalize', dest='normalize', action='store_true')
    ap.add_argument('--no-normalize', dest='normalize', action='store_false')
    ap.set_defaults(normalize=True)

    ap.add_argument('--out-piece', default='fused_piece3.npy')
    ap.add_argument('--out-midi',  default='fused.mid')
    ap.add_argument('--debug', action='store_true')

    args = ap.parse_args()

    # ---- load audio probs ----
    onset = _load_2d(args.onset)
    duration = _load_2d(args.duration)
    offset = _load_2d(args.offset)

    _log_stats('onset(raw)', onset); _log_stats('duration(raw)', duration); _log_stats('offset(raw)', offset)

    # onset invert
    if args.onset_invert == 'yes':
        onset = 1.0 - onset; print('[info] onset inverted: yes')
    elif args.onset_invert == 'auto':
        if float(onset.mean()) > 0.5:
            onset = 1.0 - onset; print('[info] onset inverted: auto (mean>0.5)')
        else:
            print('[info] onset inverted: auto (kept)')

    # ---- unify hop (audio) ----
    t_unit = float(args.t_unit); f_hop = float(args.fusion-hop) if hasattr(args,'fusion-hop') else float(args.fusion_hop)
    ratio = f_hop / t_unit
    if abs(ratio - round(ratio)) < 1e-9:
        k = int(round(ratio))
        onset_u    = _pool_time(onset,    k, 'max')
        offset_u   = _pool_time(offset,   k, 'max')
        duration_u = _pool_time(duration, k, 'mean')
        print(f"[info] audio pooling: k={k} (onset/offset=max, duration=mean)")
    else:
        onset_u    = resample_time_linear(onset,    t_unit, f_hop)
        duration_u = resample_time_linear(duration, t_unit, f_hop)
        offset_u   = resample_time_linear(offset,   t_unit, f_hop)
        print(f"[info] audio resample: {t_unit} -> {f_hop} (linear)")

    T_audio = onset_u.shape[0]

    # ---- hand side ----
    hand = _load_2d(args.hand_probs)
    _log_stats('hand(raw)', hand)
    hand_u = resample_time_linear(hand, float(args.hand_hop), f_hop, T_ref=T_audio)

    cov = np.ones((hand.shape[0], 1), dtype=np.float32)                # coverageトラッキング
    cov_u = resample_time_linear(cov, float(args.hand_hop), f_hop, T_ref=T_audio)

    lag = compute_best_lag(onset_u, hand_u, int(args.lag_max))
    print(f"[align] best lag={lag} (@fusion-hop)")
    hand_a = shift_with_zeros(hand_u, lag)
    cov_a  = shift_with_zeros(cov_u,  lag)

    # ---- build continuous weight W (従来機能) ----
    W = build_weight_from_hand(hand_a, args.time_smooth, args.pitch_blur, args.weight_gamma, args.weight_floor)
    if args.tail_mode == 'audio_only':
        mask = (cov_a <= 1e-6).astype(np.float32)
        W = W * (1.0 - mask) + 1.0 * mask
        print('[info] tail-mode=audio_only: coverage outside -> weight=1.0')
    else:
        print('[info] tail-mode=edge: keep resampled/shifted values')
    W = np.clip(W, 0.0, 1.0)
    _log_stats('weight(W)', W)

    # ---- apply W ----
    onset_w    = np.clip(onset_u,    0.0, 1.0)
    duration_w = np.clip(duration_u, 0.0, 1.0)
    offset_w   = np.clip(offset_u,   0.0, 1.0)
    if args.apply_to == 'onset_dura':
        onset_w    = onset_w * W
        duration_w = duration_w * W
        print('[apply] weight -> onset, duration')
    else:
        onset_w    = onset_w * W; duration_w = duration_w * W; offset_w = offset_w * W
        print('[apply] weight -> onset, duration, offset')

    # ---- rescue/suppress on onset (binary decision + fused prob) ----
    key_mask = None
    if args.key_mask is not None:
        key_mask = _load_optional_2d(args.key_mask)
        if key_mask.shape[1] != 88:
            raise ValueError("key_mask must be (T,88)")
        # resample/shift to audio timeline
        key_mask = resample_time_linear(key_mask, t_unit, f_hop, T_ref=T_audio) if key_mask.shape[0] != T_audio else key_mask
        key_mask = (key_mask >= 0.5).astype(np.uint8)

    conf = None
    if args.conf is not None:
        conf = _load_optional_2d(args.conf)  # (T,1) 推奨
        if conf.shape[0] != T_audio:
            conf = resample_time_linear(conf, t_unit, f_hop, T_ref=T_audio)

    onset_f, onset_dec = fuse_onset_rescue_suppress(
        p_o=onset_w, p_h=hand_a, key_mask=key_mask, conf=conf,
        a=args.a_logit, b=args.b_logit, c=args.c_logit,
        theta_o=args.onset_th, theta_hard=args.theta_hard, theta_rescue=args.theta_rescue,
        tau_hand=args.tau_hand, tau_fp=args.tau_fp,
        bonus_peak=args.bonus_peak, bonus_neighbor=args.bonus_neighbor,
        conf_lo=args.conf_lo, conf_hi=args.conf_hi
    )

    # piece3 は連続値で保存（解析・可視化用）
    piece3 = np.stack([onset_f, duration_w, offset_w], axis=-1).astype(np.float32)
    np.save(args.out_piece, piece3)
    print(f"[save] piece: {args.out_piece} shape={piece3.shape}")

    # ---- Omnizart MIDI ----
    if mz_inf is None:
        print('[warn] omnizart import に失敗。MIDI生成はスキップされました。')
        return
    try:
        midi = mz_inf.multi_inst_note_inference(
            piece3,
            mode=args.mode,
            onset_th=args.onset_th,    # Omnizart内部のしきい値
            dura_th=args.dura_th,
            t_unit=f_hop,
            normalize=args.normalize
        )
        midi.write(args.out_midi)
        print(f"[save] MIDI: {args.out_midi} (mode={args.mode}, t_unit={f_hop})")
    except Exception as e:
        print(f"[ERROR] omnizart inference failed: {e}")

if __name__ == '__main__':
    main()
