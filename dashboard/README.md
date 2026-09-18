# NetraShield AI — Module 5 MVP Dashboard

One-page judge-facing dashboard for the Module 4 FastAPI integration layer.

## Run locally

1. Start Module 4 at `http://localhost:8000`.
2. Copy `.env.example` to `.env` if Module 4 uses a different URL.
3. Install dependencies:
   `npm install`
4. Start the dashboard:
   `npm run dev`

The frontend uses `VITE_API_BASE_URL` and does not contain ML models or generated traffic data.

## Module 4 contract

The dashboard consumes:

- `POST /predict` — current attack detection for a traffic record
- `POST /inject-attack` — replays a real attack-classified feature record through Module 4
- `GET /traffic` — recent traffic records
- `GET /forecast` — latest future forecast
- `GET /stats` — backend/model metrics supplied by the integrated backend
- `GET /alerts` — generated detection/forecast alerts
- optional `WS /ws/live` — live detection/forecast/alert events

## Integration behavior

- Current detection and future forecast are displayed separately.
- REST data refreshes every 3 seconds.
- WebSocket is optional; REST remains the source of truth.
- The **Inject real attack** action searches recent real traffic records, verifies one with the actual `/predict` endpoint, and sends that record to `/inject-attack`. It does not fabricate a label or confidence value.
- Forecast history contains successive backend forecast outputs only; the dashboard does not invent model predictions.
- Missing metrics are shown as `—` rather than zero.

## Expected final flow

`M1 replay/preprocessing → M4 ingest → M2 detection → M3 forecasting → M4 API → M5 dashboard`

Before the full demo, Module 4 must contain the real M1/M2/M3 artifacts in `backend/artifacts/`.
