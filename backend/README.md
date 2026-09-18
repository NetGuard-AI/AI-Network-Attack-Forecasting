# Module 4 — Backend MVP

FastAPI + SQLite integration layer for the network attack forecasting MVP.

## Upstream artifacts

Create `backend/artifacts/` and copy in the REAL files:

- Module 1: `feature_schema.json`, `scaler.pkl`, `encoders.pkl`
- Module 2: `trained_model.pkl`
- Module 3: `forecast_model.pkl`

No placeholder model/artifact generators are included. The backend must not
run on fake/random artifacts.

## MVP contracts

### Detection (`POST /predict`)
Returns only:
`attack_label`, `confidence`, `is_attack`.

### Forecast (`GET /forecast` / live WebSocket message)
Returns only:
`forecast_probability`, `risk_level`, `trend`, `forecast_window_minutes`.

`forecast_window_minutes` must be declared by Module 1 in
`feature_schema.json`; Module 4 does not invent a minute value. The current
M1 implementation's flow-window values (lookback/horizon) are flow counts,
not automatically minutes, so the schema must be updated if the team wants a
literal 5-minute claim.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Swagger: `http://localhost:8000/docs`

Replay from the local M1 output:

```bash
python replay_simulator.py --file "PATH_TO_REPLAY_STREAM.csv" --loop
```

The replay posts real prepared rows to `/ingest`. `/ingest` runs Module 2
detection, feeds the rolling chronological window, and then runs Module 3
forecasting when enough flows are buffered.

## Routes

- `POST /predict`
- `POST /ingest`
- `GET /forecast`
- `GET /traffic`
- `GET /stats`
- `GET /alerts`
- `POST /inject-attack` — accepts a real feature record and sends it through
  the same detection pipeline; it does not fabricate a model result.
- `WS /ws/live`

## Deliberately excluded from MVP

No ARIMA/LSTM, multiple forecasting models, advanced risk calculations,
PostgreSQL, authentication, Docker, cloud deployment, SHAP, or mitigation.
