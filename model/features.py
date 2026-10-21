"""Preprocessing + feature extraction. Mirrors the training notebook (swallow_classification_colab) exactly."""
import numpy as np
from scipy import signal as sg, stats
import pywt

FS_RAW = 4000
DECIM = 2
FS_W = FS_RAW // DECIM          # 2000 Hz working rate
WIN = 4000                      # 2 s at 2 kHz
BANDS = [(1, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 700), (700, 1000)]
_SOS_HP = sg.butter(4, 1.0, "hp", fs=FS_RAW, output="sos")


def preprocess(x, fs=FS_RAW):
    """x: (N, 4) raw samples. Returns (N/2, 4) float32: 1 Hz high-pass, decimate x2, per-channel median-abs scaling."""
    x = np.asarray(x, dtype=np.float64)
    if fs != FS_RAW:
        g = np.gcd(int(fs), FS_RAW)
        x = sg.resample_poly(x, FS_RAW // g, int(fs) // g, axis=0)
    x = sg.sosfiltfilt(_SOS_HP, x, axis=0)
    x = sg.decimate(x, DECIM, axis=0, zero_phase=True)
    x = x / (np.median(np.abs(x), axis=0) + 1e-6)
    return x.astype(np.float32)


def feats_one(w):
    """w: (4, WIN) -> 135 features (same order as training)."""
    f = []
    for ch in w:
        d = np.diff(ch); dd = np.diff(d)
        v0, v1, v2 = ch.var() + 1e-12, d.var() + 1e-12, dd.var() + 1e-12
        f += [np.abs(ch).mean(), np.sqrt(v0), stats.skew(ch), stats.kurtosis(ch), np.ptp(ch),
              ((ch[:-1] * ch[1:]) < 0).mean(), np.sqrt(v1 / v0), np.sqrt(v2 / v1) / np.sqrt(v1 / v0)]
        fr, p = sg.welch(ch, FS_W, nperseg=512); p = p + 1e-14; tot = p.sum()
        f += [np.log(p[(fr >= lo) & (fr < hi)].sum() / tot + 1e-12) for lo, hi in BANDS]
        pn = p / tot
        f += [(fr * pn).sum(), fr[p.argmax()], -(pn * np.log(pn)).sum()]
        co = pywt.wavedec(ch, "db4", level=6)
        e = np.array([np.sum(c ** 2) for c in co]); e = e / (e.sum() + 1e-12)
        f += list(np.log(e + 1e-9))
        seg = np.array_split(np.abs(ch), 8)
        f += [s.mean() / (np.abs(ch).mean() + 1e-9) for s in seg]
    cc = np.corrcoef(w[1:]); f += [cc[0, 1], cc[0, 2], cc[1, 2]]
    return f


def window_features(xp, hop=WIN):
    """xp: preprocessed (M, 4). Returns (starts, feature matrix) for every full 2 s window at the given hop (samples @2 kHz)."""
    starts = list(range(0, len(xp) - WIN + 1, hop))
    F = np.array([feats_one(xp[s:s + WIN].T) for s in starts], dtype=np.float32)
    return starts, np.nan_to_num(F)
