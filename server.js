const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;
app.disable('x-powered-by');
const MODEL_URL = process.env.MODEL_URL || 'http://127.0.0.1:5001';
// Proxy to the Python swallow model (model/server.py)
app.post('/api/swallow', express.json({ limit: '64mb' }), async (req, res) => {
  try {
    const r = await fetch(MODEL_URL + '/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req.body) });
    res.status(r.status).json(await r.json());
  } catch (e) { res.status(503).json({ error: 'Model service unreachable. Start it with: python model/server.py' }); }
});
app.use(express.static(path.join(__dirname, 'public'), { extensions: ['html'], etag: false, setHeaders: r => r.setHeader('Cache-Control', 'no-store') }));
app.get('/healthz', (_req, res) => res.json({ ok: true }));
app.listen(PORT, () => console.log(`DysphagiaGuard running → http://localhost:${PORT}`));
