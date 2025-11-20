import numpy as np
import tensorflow as tf
from omnizart.feature.cfp import extract_cfp
from omnizart.music import prediction as mz_pred
from omnizart.music import inference  as mz_inf
from pathlib import Path

wav_path   = "/home/yamada/MIDI_TEST_SET/miditest_videos/27/27.wav"
stem = Path(wav_path).stem
model_dir  = "/home/yamada/GNN/omnizart/omnizart/checkpoints/music/music_piano-v2"
out_dir   = "/home/yamada/MIDI_TEST_SET/miditest_videos/27"
# 1) モデル読込（SavedModel でOK）
model = tf.keras.models.load_model(model_dir, compile=False)

# 保存ユーティリティ
Path(out_dir).mkdir(parents=True, exist_ok=True)
stem = Path(wav_path).stem  # 例: "5"

def outpath(name: str) -> str:
    """保存先パスを生成（例: /home/yamada/OUTPUTS/5_onset_prob_88.npy）"""
    return str(Path(out_dir) / f"{stem}_{name}")

def save_npy(name: str, arr: np.ndarray):
    """float32で保存 & ログ"""
    p = outpath(name)
    np.save(p, arr.astype(np.float32))
    print(f"[save] {name} {arr.shape} -> {p}")


# 1) モデル読込（SavedModel）
print("[load] model:", model_dir)
model = tf.keras.models.load_model(model_dir, compile=False)

# 2) CFP 抽出（Settings 準拠）
print("[feature] extract CFP:", wav_path)
Z, tfrL0, tfrLF, tfrLQ, cen = extract_cfp(
    wav_path,
    down_fs=44100, hop=0.02, win_size=7939,
    fr=2.0, fc=27.5, tc=0.00022287,
    g=[0.24, 0.6, 1.0], bin_per_octave=48
)

# tfrL0/tfrLF/tfrLQ は (352, T) → (T, 352, 3) に整形
feat = np.stack([tfrL0.T, tfrLQ.T, tfrLF.T], axis=-1).astype(np.float32)
print("feat fixed:", feat.shape, feat.dtype)  # 期待: (T, 352, 3)

# timesteps/step_size をモデルから推定
def _guess_timesteps(keras_model, default=256):
    shp = getattr(keras_model, "input_shape", None)
    if isinstance(shp, (list, tuple)) and len(shp) >= 2 and isinstance(shp[1], int):
        return shp[1]
    try:
        return int(keras_model.layers[0].input_shape[1])
    except Exception:
        return default

timesteps = _guess_timesteps(model, default=256)  # 例: 256
step_size = max(1, min(64, timesteps // 4))       # 例: 64
batch_size = 4

# チェック
assert feat.ndim == 3 and feat.shape[1] == 352 and feat.shape[2] == 3, f"bad feat shape: {feat.shape}"
assert not np.isnan(feat).any(), "NaN in feature."

# 必要ならパディング（前方に合わせる）
T = feat.shape[0]
if T < timesteps:
    pad = np.repeat(feat[-1:, :, :], timesteps - T, axis=0)
    feat = np.concatenate([feat, pad], axis=0)

print("feat.shape      =", feat.shape, "dtype=", feat.dtype)
print("any NaN?        =", bool(np.isnan(feat).any()))
print("model.input_shape =", getattr(model, "input_shape", None))
print("timesteps/step_size =", timesteps, step_size)

# 3) 推論（スライス→結合を内部で実施）
print("[infer] predicting...")
pred = mz_pred.predict(feat, model, batch_size=batch_size, step_size=step_size)  # (T, ~354, C)

# 4) 88鍵にダウンサンプル
pred88 = mz_inf.down_sample(pred)  # (T, 88, C) or (T, 88)
print("pred88.shape =", pred88.shape)

# 生の multi-ch を保存（保険）
save_npy("pred88_raw.npy", pred88)

# ---- 便利関数 ----
def _interp_2d(x2: np.ndarray, ori=0.02, tar=0.01) -> np.ndarray:
    """(T,88) を時間0.02s→0.01sに補間"""
    return mz_inf.interpolation(x2.astype(np.float32), ori_t_unit=ori, tar_t_unit=tar)

def _extract_channels(pred88_arr: np.ndarray):
    """
    pred88: (T,88,C) or (T,88)
    返り値: {'onset','dura','offset','off','idx':dict}
    """
    if pred88_arr.ndim == 2:
        # 1ch の場合は duration のみ
        return {"onset": None, "dura": pred88_arr, "offset": None, "off": None,
                "idx": {"onset": None, "dura": 0, "offset": None, "off": None}}

    C = pred88_arr.shape[2]
    chans = [pred88_arr[..., i] for i in range(C)]

    if C == 3:
        # 典型: [onset, dura(frame), offset]
        onset_idx, dura_idx, offset_idx = 0, 1, 2
        off_idx = None
    elif C == 4:
        # ヒューリスティックで推定
        mad = [np.mean(np.abs(np.diff(ch, axis=0))) for ch in chans]
        dura_idx = int(np.argmin(mad))

        pos = [np.mean(np.clip(np.diff(ch, axis=0), 0, None)) for ch in chans]
        onset_idx = int(np.argmax(pos))

        neg = [np.mean(np.clip(-np.diff(ch, axis=0), 0, None)) for ch in chans]
        offset_idx = int(np.argmax(neg))

        used = {dura_idx, onset_idx, offset_idx}
        off_idx = next((i for i in range(C) if i not in used), None)
    else:
        # 想定外: dura = 最も滑らか、onset/offsetは無し
        mad = [np.mean(np.abs(np.diff(ch, axis=0))) for ch in chans]
        dura_idx = int(np.argmin(mad))
        onset_idx = None
        offset_idx = None
        off_idx = None

    dura   = pred88_arr[..., dura_idx]
    onset  = pred88_arr[..., onset_idx]  if onset_idx  is not None else None
    offset = pred88_arr[..., offset_idx] if offset_idx is not None else None
    off    = pred88_arr[..., off_idx]    if off_idx    is not None else None

    return {
        "onset": onset, "dura": dura, "offset": offset, "off": off,
        "idx": {"onset": onset_idx, "dura": dura_idx, "offset": offset_idx, "off": off_idx}
    }
# -------------------

# チャンネル抽出
ch = _extract_channels(pred88)
print("[channels] indices:", ch["idx"])

# onset が無い場合は duration から擬似生成
if ch["onset"] is None:
    d = ch["dura"].astype(np.float32)
    pseudo_onset = np.clip(d - np.roll(d, 1, axis=0), 0, 1)
    pseudo_onset[0] = 0.0
    ch["onset"] = pseudo_onset
    print("[info] onset channel not found → pseudo onset from duration")

# 0.01s に補間して保存
if ch["onset"] is not None:
    onset_001 = _interp_2d(ch["onset"])
    save_npy("onset_prob_88.npy", onset_001)

dura_001 = _interp_2d(ch["dura"])
save_npy("duration_prob_88.npy", dura_001)

if ch["offset"] is not None:
    offset_001 = _interp_2d(ch["offset"])
    save_npy("offset_prob_88.npy", offset_001)

if ch["off"] is not None:
    off_001 = _interp_2d(ch["off"])
    save_npy("off_prob_88.npy", off_001)

# 以降の処理で使いたいとき用に、変数としても保持
duration_prob_88 = dura_001
print("[done]")