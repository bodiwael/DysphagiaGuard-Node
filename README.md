# DysphagiaGuard Web

Web port of the Flutter app: live IMU feed from Firebase Realtime Database → 30 s calibration → swallow classification → animated dashboard.

## Run
    npm install
    npm start        # http://localhost:3000

Or open `public/index.html` through any static server (Firebase Hosting: `firebase deploy`).

## Swallow ML model (swallow_gbm.joblib)
Two processes: `npm start` (site, :3000) and the model service:

    python -m venv .venv && . .venv/bin/activate
    pip install -r model/requirements.txt     # scikit-learn must be 1.6.1 (the version it was trained with)
    npm run model                             # :5001, proxied by the site at /api/swallow

- Model: HistGradientBoosting, 135 features, **swallow vs non-swallow** on 2 s windows of 4 channels (mic + 3-axis accelerometer, 4 kHz), threshold 0.5. Trained on Zenodo 4539695.
- The ESP32 firmware has no microphone and samples at 100 Hz, so this model cannot run on its live data. It is used through **Upload CSV** only.
- Offline: use **Upload CSV** (4 columns) in the Swallow Detector card.
- It does not replace the IMU heuristic (Normal/Delayed/Aspiration); the two run side by side.

## Live feeds (both active)
1. **Original (continuous):** project `abdelrahman-wael`, `devices/<id>/live` (device box, default `dysphagiaguard-01`). This is the feed that worked before.
2. **ESP32 firmware windows (extra):** project `bioflex-237aa`, `dysphagiaGuard/device01/latest` (0.5 s bursts). Needs public read in that project's rules; if it fails, the console shows a warning and source 1 is unaffected.


## Notes
- Firebase config is in the `FIREBASE` object at the top of `public/index.html`.
- The wearable writes `{ax,ay,az,gx,gy,gz,t}` to `devices/<deviceId>/live`.
- No hardware? Press **Run demo** — it simulates a wearable (calibrates in 6 s).
- `database.rules.json` is open for testing. Lock it down before real patient use.
- This uses the heuristic classifier from the app (the .tflite model wasn't in the zip).
- `WINDOW_SAMPLES` (200) is a guess; set it to `window_samples` from your feature_stats.json.
