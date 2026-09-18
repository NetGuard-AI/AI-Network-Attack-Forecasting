"""Module 4 MVP FastAPI integration layer.

This backend intentionally contains no placeholder ML artifacts. Put the real
M1/M2/M3 artifacts in backend/artifacts/ before starting a full demo.
"""
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import db
import feature_utils

DETECT_AVAILABLE = False
FORECAST_AVAILABLE = False
LIVE_FORECAST_WINDOW = None

try:
    from inference import detect, scale_record, MissingFeatureError
    DETECT_AVAILABLE = True
except Exception as e:
    print(f"[startup] detection model unavailable: {e}")
    class MissingFeatureError(ValueError): pass
    def detect(record: dict) -> dict: raise RuntimeError("Detection model not loaded")
    def scale_record(record: dict): raise RuntimeError("Detection model not loaded")

try:
    from forecast import forecast_from_row, get_model_features, load_model
    _forecast_model = load_model()
    FORECAST_AVAILABLE = True
    LIVE_FORECAST_WINDOW = feature_utils.make_live_forecast_window(get_model_features(_forecast_model))
except Exception as e:
    print(f"[startup] forecast model unavailable: {e}")
    def forecast_from_row(features: dict) -> dict: raise RuntimeError("Forecast model not loaded")

HIGH_FORECAST_THRESHOLD = 0.70
DETECTION_ALERT_CONFIDENCE = 0.90
_last_forecast_probability: Optional[float] = None

class ConnectionManager:
    def __init__(self): self.active = []
    async def connect(self, ws):
        await ws.accept(); self.active.append(ws)
    def disconnect(self, ws):
        if ws in self.active: self.active.remove(ws)
    async def broadcast(self, message):
        dead = []
        for ws in self.active:
            try: await ws.send_json(message)
            except Exception: dead.append(ws)
        for ws in dead: self.disconnect(ws)

manager = ConnectionManager()
app = FastAPI(title="Network Attack Forecasting - Module 4 MVP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup(): db.init_db()

def label_response(label, confidence):
    return {"attack_label": label, "confidence": float(confidence), "is_attack": label != "BENIGN"}

def detection_alert(label, confidence):
    if label != "BENIGN" and confidence > DETECTION_ALERT_CONFIDENCE:
        msg = f"High-confidence {label} detection ({confidence:.2f})"
        db.insert_alert("high", msg, "detection")
        return msg
    return None

class PredictResponse(BaseModel):
    attack_label: str
    confidence: float
    is_attack: bool

@app.post("/predict", response_model=PredictResponse)
async def predict(record: dict):
    if not DETECT_AVAILABLE: raise HTTPException(503, "Detection model not loaded")
    try: result = detect(record)
    except MissingFeatureError as e: raise HTTPException(422, str(e))
    return label_response(result["label"], result["confidence"])

@app.post("/ingest")
async def ingest(record: dict):
    flow_seq = record.get("flow_seq")
    traffic_id = db.insert_traffic(record, flow_seq=flow_seq)
    if not DETECT_AVAILABLE: raise HTTPException(503, "Detection model not loaded")
    try: result = detect(record)
    except MissingFeatureError as e: raise HTTPException(422, str(e))
    label, confidence = result["label"], result["confidence"]
    is_attack = label != "BENIGN"
    db.insert_detection(traffic_id, label, confidence, is_attack)
    forecast_ready = False
    if LIVE_FORECAST_WINDOW is not None:
        scaled = scale_record(record)
        LIVE_FORECAST_WINDOW.add_flow(scaled, int(is_attack))
        features = LIVE_FORECAST_WINDOW.build_features()
        if features is not None:
            forecast_ready = True
            await _run_live_forecast(features)
    msg = detection_alert(label, confidence)
    payload = label_response(label, confidence)
    await manager.broadcast({"type":"detection", "data":payload})
    if msg: await manager.broadcast({"type":"alert", "data":{"severity":"high","message":msg}})
    return {"stored": True, "detection": payload, "forecast_ready": forecast_ready}

class InjectAttackRequest(BaseModel):
    features: dict = Field(..., description="A real traffic feature record from the replay/dataset")

@app.post("/inject-attack")
async def inject_attack(body: InjectAttackRequest):
    """Run a supplied real feature record through the same detection pipeline.

    The MVP deliberately does not fabricate a fake model result. The dashboard
    can send a real attack row selected from the replay/test data.
    """
    if not DETECT_AVAILABLE: raise HTTPException(503, "Detection model not loaded")
    traffic_id = db.insert_traffic(body.features, flow_seq=body.features.get("flow_seq"))
    try: result = detect(body.features)
    except MissingFeatureError as e: raise HTTPException(422, str(e))
    label, confidence = result["label"], result["confidence"]
    is_attack = label != "BENIGN"
    db.insert_detection(traffic_id, label, confidence, is_attack, injected=True)
    if is_attack and confidence > DETECTION_ALERT_CONFIDENCE:
        db.insert_alert("high", f"[INJECTED] {label} detection ({confidence:.2f})", "manual_inject")
    payload = label_response(label, confidence)
    await manager.broadcast({"type":"detection","data":payload})
    return {"injected": True, "detection": payload}

@app.get("/traffic")
async def traffic(limit: int = 50): return db.recent_traffic(limit=limit)

@app.get("/forecast")
async def forecast():
    latest = db.latest_forecast()
    if not latest:
        return {"forecast_probability":0.0,"risk_level":"Low","trend":"stable","forecast_window_minutes":0}
    return {"forecast_probability":latest["forecast_probability"],"risk_level":latest["risk_level"],"trend":latest["trend"],"forecast_window_minutes":int(latest["forecast_window_minutes"])}

@app.get("/stats")
async def stats(): return db.get_stats()

@app.get("/alerts")
async def alerts(limit: int = 50, active_only: bool = False): return db.recent_alerts(limit=limit, active_only=active_only)

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: manager.disconnect(websocket)

def _trend(new, old):
    if old is None: return "stable"
    if new - old > 0.05: return "rising"
    if new - old < -0.05: return "falling"
    return "stable"

async def _run_live_forecast(features: dict):
    global _last_forecast_probability
    if not FORECAST_AVAILABLE: return
    try: result = forecast_from_row(features)
    except Exception as e:
        print(f"[live_forecast] forecast failed: {e}"); return
    probability = float(result["forecast_probability"])
    trend = _trend(probability, _last_forecast_probability)
    _last_forecast_probability = probability
    risk = result["risk_level"]
    window_minutes = int(result["forecast_window_minutes"])
    db.insert_forecast(probability, risk, trend, window_minutes)
    payload = {"forecast_probability":probability,"risk_level":risk,"trend":trend,"forecast_window_minutes":window_minutes}
    await manager.broadcast({"type":"forecast","data":payload})
    if probability > HIGH_FORECAST_THRESHOLD:
        msg = f"Forecast risk HIGH ({probability:.2f})"
        db.insert_alert("high", msg, "forecast")
        await manager.broadcast({"type":"alert","data":{"severity":"high","message":msg}})

@app.get("/")
async def root():
    return {"service":"Module 4 - Network Attack Forecasting MVP","detection_model_loaded":DETECT_AVAILABLE,"forecast_model_loaded":FORECAST_AVAILABLE,"docs":"/docs"}
