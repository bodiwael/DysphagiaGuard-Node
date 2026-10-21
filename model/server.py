"""Swallow vs non-swallow inference service. Run: python model/server.py  (port 5001)"""
import json, os, sys
import numpy as np, joblib
from flask import Flask, request, jsonify
sys.path.insert(0, os.path.dirname(__file__))
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
THR = float(CFG.get("thr", 0.5))
GBM = joblib.load(os.path.join(HERE, "swallow_gbm.joblib"))
assert GBM.n_features_in_ == 135

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


def run(x, fs, hop_s):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2: raise ValueError("expected 4 channels")
    if x.shape[0] == 4 and x.shape[1] != 4: x = x.T          # accept channel-major
    if x.shape[1] != 4: raise ValueError(f"need 4 channels (mic + 3-axis accelerometer), got {x.shape[1]}")
    if not np.isfinite(x).all(): raise ValueError("non-finite samples")
    if x.shape[0] / fs < 2.0: raise ValueError("need at least 2 s of signal")
    xp = F.preprocess(x, fs)
    starts, Fm = F.window_features(xp, hop=max(1, int(hop_s * F.FS_W)))
    p = GBM.predict_proba(Fm)[:, 1]
    return [dict(t=s / F.FS_W, p=float(v), swallow=bool(v > THR)) for s, v in zip(starts, p)]


@app.get("/health")
def health(): return jsonify(ok=True, thr=THR, n_features=int(GBM.n_features_in_))


@app.post("/predict")
def predict():
    try:
        j = request.get_json(force=True)
        res = run(j["samples"], float(j.get("fs", CFG["fs"])), float(j.get("hop", 2.0)))
        return jsonify(thr=THR, windows=res, swallow_any=any(r["swallow"] for r in res), p_max=max(r["p"] for r in res))
    except (KeyError, ValueError) as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("MODEL_PORT", 5001)))
