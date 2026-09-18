import { createFileRoute } from "@tanstack/react-router";
import { Activity, BellRing, Radio, ShieldAlert, ShieldCheck, TrendingUp, UploadCloud } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "NetraShield AI — Network Attack Forecasting" },
      { name: "description", content: "Judge-facing network attack detection and forecasting dashboard." },
    ],
  }),
  component: Dashboard,
});

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type AlertItem = { id?: number; severity?: string; message?: string; ts?: number; source?: string };
type TrafficRow = Record<string, unknown> & { id?: number; ts?: number; raw_json?: string; flow_seq?: string };
type ForecastPoint = { time: string; probability: number };

function Panel({ children, className }: { children: React.ReactNode; className?: string }) {
  return <section className={cn("rounded-lg border border-border bg-card p-4 shadow-sm", className)}>{children}</section>;
}

function PanelTitle({ icon: Icon, title, subtitle, action }: { icon: typeof Activity; title: string; subtitle: string; action?: React.ReactNode }) {
  return <div className="mb-4 flex items-start justify-between gap-3"><div className="flex items-start gap-2.5"><span className="mt-0.5 grid size-8 place-items-center rounded-md bg-accent text-primary"><Icon className="size-4" /></span><div><h2 className="text-sm font-semibold text-foreground">{title}</h2><p className="mt-0.5 font-mono text-[10px] text-muted-foreground">{subtitle}</p></div></div>{action}</div>;
}

function Dashboard() {
  const [traffic, setTraffic] = useState<TrafficRow[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [current, setCurrent] = useState({ attack_label: "—", confidence: null as number | null, is_attack: false });
  const [forecastCurrent, setForecastCurrent] = useState({ forecast_probability: null as number | null, risk_level: "—", trend: "—", forecast_window_minutes: null as number | null });
  const [stats, setStats] = useState<Record<string, unknown>>({});
  const [backendReachable, setBackendReachable] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastSync, setLastSync] = useState("—");
  const [error, setError] = useState("");
  const [injecting, setInjecting] = useState(false);
  const [injectStatus, setInjectStatus] = useState("");

  const api = useCallback(async (path: string, init?: RequestInit) => {
    const res = await fetch(`${API_BASE}${path}`, init);
    if (!res.ok) throw new Error(`${path} returned ${res.status}`);
    return res.json();
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [trafficData, alertsData, forecastData, statsData] = await Promise.all([
        api("/traffic?limit=25"),
        api("/alerts?limit=20"),
        api("/forecast"),
        api("/stats"),
      ]);
      setTraffic(Array.isArray(trafficData) ? trafficData : []);
      setAlerts(Array.isArray(alertsData) ? alertsData : []);
      const nextForecast = forecastData ?? {};
      setForecastCurrent({
        forecast_probability: nextForecast.forecast_probability == null ? null : Number(nextForecast.forecast_probability),
        risk_level: nextForecast.risk_level == null ? "—" : String(nextForecast.risk_level),
        trend: nextForecast.trend == null ? "—" : String(nextForecast.trend),
        forecast_window_minutes: nextForecast.forecast_window_minutes == null ? null : Number(nextForecast.forecast_window_minutes),
      });
      setStats(statsData ?? {});
      const latest = Array.isArray(trafficData) ? trafficData[0] : null;
      if (latest?.raw_json) {
        try {
          const record = JSON.parse(String(latest.raw_json));
          const prediction = await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(record) });
          setCurrent({
            attack_label: String(prediction.attack_label ?? "—"),
            confidence: prediction.confidence == null ? null : Number(prediction.confidence),
            is_attack: Boolean(prediction.is_attack),
          });
        } catch {
          setCurrent({ attack_label: "UNAVAILABLE", confidence: null, is_attack: false });
        }
      }
      if (nextForecast.forecast_probability != null && Number(nextForecast.forecast_window_minutes ?? 0) > 0) {
        const probability = Number(nextForecast.forecast_probability) * 100;
        setForecast((points) => {
          const last = points[points.length - 1];
          if (last && Math.abs(last.probability - probability) < 0.0001) return points;
          return [...points, {
            time: new Date().toLocaleTimeString("en-IN", { hour12: false }),
            probability,
          }].slice(-30);
        });
      }
      setBackendReachable(true);
      setError("");
      setLastSync(new Date().toLocaleTimeString("en-IN", { hour12: false }));
    } catch (e) {
      setBackendReachable(false);
      setError(e instanceof Error ? e.message : "Backend unavailable");
    }
  }, [api]);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    let ws: WebSocket | undefined;
    try {
      ws = new WebSocket(`${API_BASE.replace(/^http/, "ws")}/ws/live`);
      ws.onopen = () => setWsConnected(true);
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === "forecast" && message.data?.forecast_probability != null) {
            const probability = Number(message.data.forecast_probability);
            setForecastCurrent({
              forecast_probability: probability,
              risk_level: String(message.data.risk_level ?? "—"),
              trend: String(message.data.trend ?? "—"),
              forecast_window_minutes: message.data.forecast_window_minutes == null ? null : Number(message.data.forecast_window_minutes),
            });
            setForecast((points) => [...points, {
              time: new Date().toLocaleTimeString("en-IN", { hour12: false }),
              probability: probability * 100,
            }].slice(-30));
          }
        } catch {
          // REST polling remains the source of truth when a WS message is malformed.
        }
        void refresh();
      };
      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
    } catch {
      setWsConnected(false);
    }
    return () => ws?.close();
  }, [refresh]);

  const injectAttack = async () => {
    setInjecting(true);
    setInjectStatus("");
    try {
      const rows = await api("/traffic?limit=100");
      if (!Array.isArray(rows)) throw new Error("/traffic did not return traffic records");

      let attackRecord: Record<string, unknown> | null = null;
      for (const row of rows) {
        if (!row?.raw_json) continue;
        try {
          const record = JSON.parse(String(row.raw_json)) as Record<string, unknown>;
          const prediction = await api("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(record),
          });
          if (prediction?.is_attack) {
            attackRecord = record;
            break;
          }
        } catch {
          // Skip malformed/incompatible traffic rows and continue searching the real stream.
        }
      }

      if (!attackRecord) {
        throw new Error("No attack-classified traffic record is available to replay yet");
      }

      await api("/inject-attack", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: attackRecord }),
      });
      setInjectStatus("Real attack-classified traffic replayed through Module 4");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Attack replay failed");
    } finally {
      setInjecting(false);
    }
  };

  const probability = forecastCurrent.forecast_probability == null ? null : forecastCurrent.forecast_probability * 100;
  const risk = String(forecastCurrent.risk_level ?? "—").toUpperCase();
  const metric = (key: string) => {
    const value = stats[key];
    return value === undefined || value === null ? "—" : String(value);
  };

  return <main className="min-h-screen bg-background text-foreground">
    <div className="mx-auto max-w-[1500px] px-3 py-4 sm:px-5 lg:px-7">
      <header className="flex flex-col gap-4 border-b border-border pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3"><div className="grid size-11 place-items-center rounded-lg bg-primary text-primary-foreground"><ShieldAlert className="size-6" /></div><div><h1 className="text-xl font-bold">NetraShield <span className="text-primary">AI</span></h1><p className="font-mono text-[10px] text-muted-foreground">NETWORK ATTACK FORECASTING · MODULE 5 MVP</p></div></div>
        <div className="flex flex-wrap items-center gap-2">
          <div className={cn("flex h-9 items-center gap-2 rounded-md border px-3 font-mono text-[10px]", backendReachable ? "border-success/30 bg-success/10 text-success" : "border-critical/30 bg-critical/10 text-critical")}><span className="size-2 rounded-full bg-current" />{backendReachable ? `BACKEND CONNECTED${wsConnected ? " · WS LIVE" : " · REST"}` : "BACKEND OFFLINE"}</div>
          <div className="h-9 rounded-md border border-border bg-card px-3 py-1.5 text-right font-mono"><p className="text-[8px] text-muted-foreground">LAST SYNC</p><p className="text-[11px]">{lastSync}</p></div>
          <Button onClick={injectAttack} disabled={injecting} size="sm"><UploadCloud className="mr-1.5 size-4" />{injecting ? "Replaying…" : "Inject real attack"}</Button>
        </div>
      </header>

      {error && <div className="mt-3 rounded-md border border-critical/30 bg-critical/10 px-3 py-2 font-mono text-[10px] text-critical">Backend integration: {error}</div>}
      {injectStatus && <div className="mt-3 rounded-md border border-success/30 bg-success/10 px-3 py-2 font-mono text-[10px] text-success">{injectStatus}</div>}

      <div className="mt-4 grid gap-4 xl:grid-cols-12">
        <Panel className="border-critical/30 xl:col-span-4"><PanelTitle icon={ShieldAlert} title="Current attack detection" subtitle="LIVE INFERENCE · /predict" /><div className="flex items-end justify-between gap-3"><div><p className={cn("text-2xl font-bold", current.is_attack ? "text-critical" : "text-success")}>{current.attack_label}</p><p className="mt-1 text-xs text-muted-foreground">{current.is_attack ? "Attack detected in latest traffic record" : "No attack detected in latest traffic record"}</p></div><div className="text-right"><p className="font-mono text-3xl font-semibold">{current.confidence == null ? "—" : Math.round(current.confidence * 100)}<span className="text-sm text-muted-foreground">%</span></p><p className="font-mono text-[9px] text-muted-foreground">CONFIDENCE</p></div></div></Panel>

        <Panel className="border-warning/30 xl:col-span-4"><PanelTitle icon={TrendingUp} title="Forecast risk" subtitle="FUTURE FORECAST · /forecast" /><div className="flex items-end justify-between gap-3"><div><p className="text-2xl font-bold text-warning">{risk} · {forecastCurrent.trend}</p><p className="mt-1 text-xs text-muted-foreground">Module 4 · {forecastCurrent.forecast_window_minutes == null ? "forecast window unavailable" : `${forecastCurrent.forecast_window_minutes} min forecast window`}</p></div><div className="text-right"><p className="font-mono text-3xl font-semibold">{probability == null ? "—" : Math.round(probability)}<span className="text-sm text-muted-foreground">%</span></p><p className="font-mono text-[9px] text-muted-foreground">PROBABILITY</p></div></div><div className="mt-4 h-2 overflow-hidden rounded-full bg-muted"><div className="h-full rounded-full bg-warning" style={{ width: `${probability == null ? 0 : Math.min(100, Math.max(0, probability))}%` }} /></div><div className="mt-2 flex justify-between font-mono text-[9px] text-muted-foreground"><span>0%</span><span>70% high-risk threshold</span><span>100%</span></div></Panel>

        <Panel className="xl:col-span-4"><PanelTitle icon={ShieldCheck} title="Model metrics" subtitle="ONLY VALUES PROVIDED BY BACKEND" /><div className="grid grid-cols-2 gap-2"><Score label="Precision" value={metric("precision")} /><Score label="Recall" value={metric("recall")} /><Score label="F1 score" value={metric("f1")} /><Score label="Forecast accuracy" value={metric("forecast_accuracy")} /></div><p className="mt-2 font-mono text-[9px] text-muted-foreground">Metrics are not fabricated; Module 4 must expose evaluation values when supplied by upstream modules.</p></Panel>

        <Panel className="xl:col-span-7"><PanelTitle icon={Activity} title="Live traffic" subtitle="LATEST RECORDS · /traffic" /><div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left"><thead><tr className="border-b border-border font-mono text-[9px] uppercase text-muted-foreground"><th className="pb-2 font-medium">Time</th><th className="pb-2 font-medium">Flow</th><th className="pb-2 font-medium">Source</th><th className="pb-2 font-medium">Destination</th><th className="pb-2 font-medium">Protocol</th></tr></thead><tbody>{traffic.slice(0, 10).map((row, i) => { let r: Record<string, unknown> = {}; try { r = row.raw_json ? JSON.parse(String(row.raw_json)) : {}; } catch {} return <tr key={String(row.id ?? i)} className="border-b border-border/60 font-mono text-[10px] last:border-0"><td className="py-2.5 text-muted-foreground">{row.ts ? new Date(Number(row.ts) * 1000).toLocaleTimeString("en-IN", { hour12: false }) : "—"}</td><td className="py-2.5">{String(row.flow_seq ?? r.flow_seq ?? "—")}</td><td className="py-2.5">{String(r["Source IP"] ?? r.src_ip ?? r.source ?? "—")}</td><td className="py-2.5">{String(r["Destination IP"] ?? r.dst_ip ?? r.destination ?? "—")}</td><td className="py-2.5">{String(r.Protocol ?? r.protocol ?? "—")}</td></tr>})}</tbody></table></div></Panel>

        <Panel className="xl:col-span-5"><PanelTitle icon={TrendingUp} title="Forecast trend" subtitle="SUCCESSIVE BACKEND FORECASTS" action={<span className="font-mono text-[9px] text-critical">70% THRESHOLD</span>} /><div className="h-60"><ResponsiveContainer width="100%" height="100%"><LineChart data={forecast}><CartesianGrid stroke="var(--grid-line)" vertical={false}/><XAxis dataKey="time" stroke="var(--muted-foreground)" fontSize={10} tickLine={false} axisLine={false}/><YAxis domain={[0, 100]} stroke="var(--muted-foreground)" fontSize={10} tickLine={false} axisLine={false}/><Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 11 }}/><ReferenceLine y={70} stroke="var(--critical)" strokeDasharray="5 5"/><Line type="monotone" dataKey="probability" stroke="var(--warning)" strokeWidth={3} dot={false} /></LineChart></ResponsiveContainer></div></Panel>

        <Panel className="xl:col-span-5"><PanelTitle icon={BellRing} title="Alerts" subtitle="NEWEST FIRST · /alerts" /><div className="space-y-2">{alerts.slice(0, 8).map((alert, i) => <div key={String(alert.id ?? i)} className={cn("rounded-md border p-2.5", String(alert.severity).toLowerCase() === "high" ? "border-critical/35 bg-critical/10" : "border-warning/35 bg-warning/10")}><p className="text-xs font-medium">{alert.message ?? "Alert"}</p><p className="mt-0.5 font-mono text-[9px] uppercase text-muted-foreground">{alert.severity ?? "unknown"} · {alert.ts ? new Date(Number(alert.ts) * 1000).toLocaleTimeString("en-IN", { hour12: false }) : "—"}</p></div>)}</div></Panel>

        <Panel className="xl:col-span-7"><PanelTitle icon={Radio} title="Backend status" subtitle="MVP INTEGRATION CONTRACT" /><div className="grid grid-cols-2 gap-2 sm:grid-cols-4"><Score label="Traffic records" value={metric("total_traffic")} /><Score label="Detections" value={metric("total_detections")} /><Score label="Attacks" value={metric("total_attacks")} /><Score label="Active alerts" value={metric("active_alerts")} /></div></Panel>
      </div>
      <footer className="mt-4 border-t border-border pt-3 font-mono text-[9px] text-muted-foreground">MODULE 4: {API_BASE} · REST polling 3s · WebSocket optional · ONE-PAGE MVP</footer>
    </div>
  </main>;
}

function Score({ label, value }: { label: string; value: string }) {
  return <div className="rounded-md bg-surface-raised p-3"><p className="font-mono text-[9px] uppercase text-muted-foreground">{label}</p><p className="mt-1 text-xl font-semibold">{value}</p></div>;
}
