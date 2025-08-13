
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
import random, os, time

st.set_page_config(page_title="TPPR AI — Digital Twin", layout="wide")

# -------------------------- THEME / STYLES --------------------------
st.markdown("""
<style>
/* Global tweaks */
:root { --card-bg: #0f172a; --panel:#0b1220; --border:#1e293b; --text:#e2e8f0; }
html, body, [class*="css"]  { color: var(--text) !important; }
.block-container { padding-top: 1.2rem; }

/* Cards */
.card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset; }
.hero { background: linear-gradient(135deg,#0b1220, #0f172a 50%, #0b1220); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px; }

/* Badges */
.badge {display:inline-block; padding:4px 10px; border-radius:12px; font-weight:700; margin-right:6px; letter-spacing:.3px;}
.badge-warning { background: rgba(255,184,77,0.2); color:#ffb84d; border:1px solid rgba(255,184,77,0.5); }
.badge-critical { background: rgba(255,76,76,0.18); color:#ff6b6b; border:1px solid rgba(255,76,76,0.5); }

/* Flash animation */
@keyframes flash {0% {opacity: 1;} 50% {opacity:.35;} 100% {opacity:1;}}
.flash { animation: flash 1.2s infinite; }

/* Buttons */
.bigbtn button { width:100%; padding:10px 14px; font-weight:700; border-radius:12px; }
.sim button { background:#1d4ed8; color:white; }
.reset button { background:#334155; color:#e2e8f0; }

/* Side panel */
.rightpanel { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 10px 12px; }

/* Code summary */
.summary pre { background:#0a0f1a !important; border:1px solid #152036; }
</style>
""", unsafe_allow_html=True)

# -------------------------- CONFIG --------------------------
ROOMS = [
    {"id":"mixing","name":"Mixing Room","icon":"🧪","x":0,"y":0},
    {"id":"pack","name":"Packaging","icon":"📦","x":6,"y":0},
    {"id":"boiler","name":"Boiler Room","icon":"🔥","x":0,"y":6},
    {"id":"waste","name":"Waste Handling","icon":"♻️","x":6,"y":6},
]
ROOM_SIZE = 5.0
ROOM_HEIGHT = 0.8
DEFAULT_COLOR = "#223147"
WARNING_COLOR = "#8a6b2a"
CRITICAL_COLOR = "#7a1e1e"

GASES = ["CH4","H2S","CO"]
BASELINES = {"CH4":(8,2), "H2S":(2,0.7), "CO":(5,1.5)}
THRESHOLDS = {"CH4":{"warning":25,"critical":50},
              "H2S":{"warning":10,"critical":20},
              "CO":{"warning":30,"critical":60}}
COLOR_MAP = {"CH4":"#60a5fa", "H2S":"#34d399", "CO":"#f87171"}

# -------------------------- INIT STATE --------------------------
if "df" not in st.session_state:
    if os.path.exists("sample_data.csv"):
        df = pd.read_csv("sample_data.csv", parse_dates=["timestamp"])
    else:
        now = pd.Timestamp.now().floor("min")
        rows = []
        for minute in range(180):
            ts = now + pd.Timedelta(minutes=minute)
            for r in ROOMS:
                for g in GASES:
                    mu, sd = BASELINES[g]
                    val = np.random.normal(mu, sd)
                    rows.append({"timestamp":ts,"room":r["id"],"gas":g,"ppm":float(round(val,2))})
        df = pd.DataFrame(rows)
    st.session_state.df = df
    st.session_state.selected_room = None
    st.session_state.room_colors = {r["id"]: DEFAULT_COLOR for r in ROOMS}
    st.session_state.incident_history = []  # newest first
    st.session_state.alert_active = {}

df = st.session_state.df

# -------------------------- HELPERS --------------------------
def latest_room_avg(room_id:str)->float:
    d = df[df["room"]==room_id]
    if d.empty: return 0.0
    latest = d.sort_values("timestamp").groupby("gas").tail(1)
    return float(round(latest["ppm"].mean(),2))

def room_any_status(room_id:str):
    d = df[df["room"]==room_id].sort_values("timestamp")
    if d.empty: return None
    latest = d.groupby("gas").tail(1)
    status = None
    for _,row in latest.iterrows():
        thr = THRESHOLDS[row["gas"]]
        if row["ppm"] >= thr["critical"]:
            return "critical"
        if row["ppm"] >= thr["warning"]:
            status = "warning"
    return status

def spike_profile(gas:str, i:int, dur:int, peak:float):
    t = i/max(dur-1,1)
    if gas=="CH4": return peak*(0.2+0.8*t**1.5)
    if gas=="H2S": return peak*(0.3+0.7*np.sin(np.pi*t))
    if gas=="CO": return peak*(0.6 if t<0.2 else 1.0)
    return peak*t

def add_threshold_lines(fig, gases, opacity=0.75, annotate=True):
    shapes, annotations = [], []
    colors = {"warning": f"rgba(255,184,77,{opacity})", "critical": f"rgba(255,76,76,{opacity})"}
    for g in gases:
        w = THRESHOLDS[g]["warning"]; c = THRESHOLDS[g]["critical"]
        shapes+= [
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=w, y1=w, line=dict(color=colors["warning"], width=2)),
            dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=c, y1=c, line=dict(color=colors["critical"], width=2, dash="dash")),
        ]
        if annotate:
            annotations+= [
                dict(x=1.0, xref="paper", y=w, yref="y", text=f"{g} Warning: {w} ppm", showarrow=False, xanchor="right",
                     font=dict(size=10, color="#ffb84d"), bgcolor="rgba(255,184,77,0.08)"),
                dict(x=1.0, xref="paper", y=c, yref="y", text=f"{g} Critical: {c} ppm", showarrow=False, xanchor="right",
                     font=dict(size=10, color="#ff6b6b"), bgcolor="rgba(255,76,76,0.08)"),
            ]
    fig.update_layout(shapes=shapes, annotations=annotations)

def status_label(g, ppm):
    thr = THRESHOLDS[g]
    if ppm >= thr["critical"]: return "Critical"
    if ppm >= thr["warning"]: return "Warning"
    return "Safe"

def build_incident_summary(room_id:str, when:pd.Timestamp):
    room = next(r for r in ROOMS if r["id"]==room_id)
    room_df = df[(df["room"]==room_id) & (df["timestamp"]<=when)].sort_values("timestamp")
    if room_df.empty:
        return f"Incident — {room['name']} {room['icon']}\nTime: {pd.Timestamp.now():%Y-%m-%d %H:%M}\nNo data."
    last_ts = room_df["timestamp"].max()
    snapshot = room_df[room_df["timestamp"]==last_ts]
    day_start = last_ts.floor("D")
    lines = [f"Incident — {room['name']} {room['icon']}", f"Time: {last_ts:%Y-%m-%d %H:%M}"]
    for g in GASES:
        row = snapshot[snapshot["gas"]==g]
        ppm = float(row["ppm"].iloc[0]) if not row.empty else 0.0
        label = status_label(g, ppm)
        gday = df[(df["room"]==room_id) & (df["gas"]==g) & (df["timestamp"]>=day_start) & (df["timestamp"]<=last_ts)]
        w = int((gday["ppm"] >= THRESHOLDS[g]["warning"]).sum())
        c = int((gday["ppm"] >= THRESHOLDS[g]["critical"]).sum())
        extra = f" — {c} critical event(s) today" if label=="Critical" else (f" — {w} warning event(s) today" if label=="Warning" else "")
        lines.append(f"{g}: {ppm} ppm ({label}){extra}")
    return "\n".join(lines)

def ai_text(room_id:str):
    room = next(r for r in ROOMS if r["id"]==room_id)
    room_df = df[df["room"]==room_id].sort_values("timestamp")
    if room_df.empty: return "No data available yet."
    latest = room_df.groupby("gas").tail(1)
    parts = []
    worst = "Safe"
    for _,row in latest.iterrows():
        g = row["gas"]; ppm = float(row["ppm"])
        label = status_label(g, ppm)
        if label=="Critical": worst="Critical"
        elif label=="Warning" and worst!="Critical": worst="Warning"
        parts.append(f"{g} at {ppm} ppm ({label})")
    head = f"{room['name']}: " + ", ".join(parts) + "."
    if worst=="Critical":
        tail = "Immediate action recommended: evacuate or increase ventilation; investigate leak."
    elif worst=="Warning":
        tail = "Elevated levels detected: increase ventilation and monitor closely."
    else:
        tail = "All gases within safe limits."
    return head + " " + tail

def forecast_series(room_id:str, gas:str, steps:int=5):
    room_df = df[df["room"]==room_id].sort_values("timestamp")
    gdf = room_df[room_df["gas"]==gas].tail(30)
    if len(gdf) < 3:
        return None
    x = np.arange(len(gdf)); y = gdf["ppm"].values.astype(float)
    m,b = np.polyfit(x,y,1)
    future_x = np.arange(len(gdf), len(gdf)+steps)
    future_y = m*future_x + b
    times = [gdf["timestamp"].iloc[-1] + pd.Timedelta(minutes=i) for i in range(1, steps+1)]
    return pd.DataFrame({"timestamp": times, "ppm": future_y, "gas": gas, "type": "predicted"})

def update_alert_badges(room_id):
    room_df = df[df["room"]==room_id].sort_values("timestamp")
    for g in GASES:
        gdf = room_df[room_df["gas"]==g]
        if gdf.empty: continue
        latest = float(gdf["ppm"].iloc[-1])
        thr = THRESHOLDS[g]
        key = (room_id, g)
        if latest >= thr["critical"]:
            st.session_state.alert_active[key] = "critical"
        elif latest >= thr["warning"]:
            st.session_state.alert_active[key] = "warning"
        else:
            if key in st.session_state.alert_active: del st.session_state.alert_active[key]
    # render
    for (r,g), level in st.session_state.alert_active.items():
        if r!=room_id: continue
        cls = "badge-critical flash" if level=="critical" else "badge-warning flash"
        label = "CRITICAL" if level=="critical" else "WARNING"
        st.markdown(f"<span class='badge {cls}'>{g} {label}</span>", unsafe_allow_html=True)

def camera_for_room(r):
    return dict(eye=dict(x=r["x"]+7,y=r["y"]+7,z=4),
                center=dict(x=r["x"]+ROOM_SIZE/2, y=r["y"]+ROOM_SIZE/2, z=ROOM_HEIGHT/2),
                up=dict(x=0,y=0,z=1))

def default_camera():
    return dict(eye=dict(x=10,y=10,z=8))

# -------------------------- SIDEBAR: INCIDENT HISTORY --------------------------
with st.sidebar:
    st.header("Incident History")
    if len(st.session_state.incident_history)==0:
        st.caption("No incidents yet. Simulate one to see it here.")
    else:
        for s in st.session_state.incident_history:
            st.code(s, language=None)

# -------------------------- TOP BAR --------------------------
st.markdown("<div class='hero'><b>TPPR AI Safety Assistant</b> — Real-time gas monitoring, incident summaries, and short-term predictions.</div>", unsafe_allow_html=True)
top_cols = st.columns([1,1,1,1,1])

with top_cols[0]:
    st.metric("🧪 Mixing avg", f"{latest_room_avg('mixing')} ppm")
with top_cols[1]:
    st.metric("📦 Packaging avg", f"{latest_room_avg('pack')} ppm")
with top_cols[2]:
    st.metric("🔥 Boiler avg", f"{latest_room_avg('boiler')} ppm")
with top_cols[3]:
    st.metric("♻️ Waste avg", f"{latest_room_avg('waste')} ppm")
with top_cols[4]:
    st.markdown("<div class='bigbtn sim'>", unsafe_allow_html=True)
    if st.button("Simulate Live Gas Event"):
        room = random.choice(ROOMS)
        gases_to_spike = random.sample(GASES, k=random.choice([1,2,3]))
        severity = random.choices(["warning","critical"], weights=[0.6,0.4])[0]
        dur = 8 if severity=="warning" else 14
        peaks = {"CH4": (30 if severity=="warning" else 70),
                 "H2S": (12 if severity=="warning" else 26),
                 "CO":  (40 if severity=="warning" else 80)}
        last_ts = df["timestamp"].max()
        start = last_ts + pd.Timedelta(minutes=1)
        new_rows = []
        for i in range(dur):
            ts = start + pd.Timedelta(minutes=i)
            for r in ROOMS:
                for g in GASES:
                    mu, sd = BASELINES[g]
                    val = np.random.normal(mu, sd)
                    if r["id"] == room["id"] and g in gases_to_spike:
                        val += spike_profile(g, i, dur, peaks[g]) + np.random.normal(0, sd*0.4)
                    new_rows.append({"timestamp": ts, "room": r["id"], "gas": g, "ppm": float(round(val,2))})
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame(new_rows)], ignore_index=True)
        df = st.session_state.df

        # Color status
        status = room_any_status(room["id"])
        st.session_state.room_colors[room["id"]] = CRITICAL_COLOR if status=="critical" else (WARNING_COLOR if status=="warning" else DEFAULT_COLOR)

        # Auto-create incident summary
        st.session_state.selected_room = room["id"]
        summary = build_incident_summary(room["id"], df["timestamp"].max())
        st.session_state.incident_history.insert(0, summary)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------- MAIN AREA --------------------------
map_col, detail_col = st.columns([2.2, 2.8])

# 3D map
with map_col:
    mesh_traces = []
    marker_x, marker_y, marker_z, marker_text, marker_room_ids = [], [], [], [], []
    def add_room_mesh(room_id, name, icon, x0, y0, color=DEFAULT_COLOR):
        size = ROOM_SIZE; height = ROOM_HEIGHT
        vx = [x0, x0+size, x0+size, x0,   x0, x0+size, x0+size, x0]
        vy = [y0, y0,      y0+size, y0+size, y0, y0,      y0+size, y0+size]
        vz = [0, 0, 0, 0,  height, height, height, height]
        faces = [[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
        i, j, k = zip(*faces)
        avg = latest_room_avg(room_id)
        ht = f"{icon} {name} — avg {avg} ppm"
        mesh_traces.append(go.Mesh3d(x=vx,y=vy,z=vz,i=i,j=j,k=k, color=color, opacity=0.96, name=name, hovertext=ht, hoverinfo="text"))
        cx, cy, cz = x0+size/2, y0+size/2, height/2
        marker_x.append(cx); marker_y.append(cy); marker_z.append(cz)
        marker_text.append(f"{icon} {name}"); marker_room_ids.append(room_id)

    for r in ROOMS:
        add_room_mesh(r["id"], r["name"], r["icon"], r["x"], r["y"], color=st.session_state.room_colors[r["id"]])

    marker_trace = go.Scatter3d(x=marker_x,y=marker_y,z=marker_z, mode="markers+text",
                                marker=dict(size=28, color="rgba(0,0,0,0)"), text=marker_text, textposition="top center", hoverinfo="text")
    fig = go.Figure(data=mesh_traces+[marker_trace])
    cam = default_camera()
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                      height=560, margin=dict(l=0,r=0,t=10,b=0), scene_camera=cam)
    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=560)
    st.plotly_chart(fig, use_container_width=True)

    if clicked and isinstance(clicked, list) and len(clicked)>0:
        pt = clicked[0]
        curve = pt.get("curveNumber", None); pnum = pt.get("pointNumber", None)
        marker_curve_idx = len(mesh_traces)
        if curve == marker_curve_idx and pnum is not None and 0 <= pnum < len(marker_room_ids):
            rid = marker_room_ids[pnum]
            st.session_state.selected_room = rid if st.session_state.selected_room != rid else None
            st.rerun()

# Detail / Room dashboard
with detail_col:
    sel = st.session_state.selected_room
    if sel is None:
        st.markdown("### Inspector")
        st.caption("Click a room on the left to open its live dashboard.")
    else:
        room = next(r for r in ROOMS if r["id"]==sel)
        room_df = df[df["room"]==sel].sort_values("timestamp")

        # AI Assistant box
        st.markdown("#### AI Safety Assistant")
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write(ai_text(sel))
            # Prediction snippet
            preds = []
            for g in GASES:
                f = forecast_series(sel, g, steps=5)
                if f is not None:
                    preds.append((g, float(f['ppm'].iloc[-1])))
            if preds:
                best = max(preds, key=lambda x:x[1])
                st.caption(f"Prediction: {best[0]} may reach ~{round(best[1],1)} ppm in ~5 min (trend estimate).")
            # TTS button (SpeechSynthesis)
            summary_text = ai_text(sel)
            tts_html = f"""
            <script>
            function speakText(){{
                const text = `{summary_text}`;
                const u = new SpeechSynthesisUtterance(text);
                u.lang = 'en-US';
                speechSynthesis.cancel();
                speechSynthesis.speak(u);
            }}
            </script>
            <button onclick="speakText()" style="margin-top:6px;padding:8px 10px;border-radius:10px;border:1px solid #1e293b;background:#0b1220;color:#e2e8f0;cursor:pointer;">🔊 Speak summary</button>
            """
            st.components.v1.html(tts_html, height=60)
            st.markdown("</div>", unsafe_allow_html=True)

        # Alert badges
        update_alert_badges(sel)

        # Metrics
        c1,c2,c3 = st.columns(3)
        for col, gas in zip([c1,c2,c3], GASES):
            gdf = room_df[room_df["gas"]==gas]
            latest = float(gdf["ppm"].iloc[-1]) if not gdf.empty else 0.0
            prev = float(gdf["ppm"].iloc[-2]) if len(gdf)>=2 else latest
            col.metric(f"{gas}", f"{latest} ppm", f"{round(latest-prev,2)} ppm")

        # Chart with predictions
        recent = room_df.groupby("gas").tail(150)
        base_fig = px.line(recent, x="timestamp", y="ppm", color="gas",
                           color_discrete_map=COLOR_MAP, markers=False,
                           title=f"{room['name']} — Live readings & 5-min AI forecast")
        add_threshold_lines(base_fig, GASES, opacity=0.75, annotate=True)

        # Add per-gas predictions as dashed lines with "pulsing" markers (style cue)
        for g in GASES:
            f = forecast_series(sel, g, steps=5)
            if f is not None:
                base_fig.add_trace(go.Scatter(x=f["timestamp"], y=f["ppm"], mode="lines+markers",
                                              name=f"Predicted {g}", line=dict(color=COLOR_MAP[g], dash="dash"),
                                              marker=dict(symbol="circle-open-dot", size=8)))

        base_fig.update_layout(height=330, margin=dict(l=0,r=0,t=40,b=0), legend=dict(orientation="h"))
        st.plotly_chart(base_fig, use_container_width=True)

        # Latest Incident Summary + copy
        st.markdown("#### Latest Incident Summary")
        room_summaries = [s for s in st.session_state.incident_history if f"— {room['name']}" in s or room['name'] in s]
        if room_summaries:
            s = room_summaries[0]
            st.markdown("<div class='summary'>", unsafe_allow_html=True)
            st.code(s, language=None)
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("Copy incident summary"):
                st.success("Summary ready — select the text above and press Ctrl+C / Cmd+C.")
        else:
            st.caption("No incidents logged for this room yet. Simulate an event to create one.")

# Fade glow back after 1.5s (if any room colored by alert)
if any(c in [WARNING_COLOR, CRITICAL_COLOR] for c in st.session_state.room_colors.values()):
    time.sleep(1.2)
    st.session_state.room_colors = {r["id"]: DEFAULT_COLOR for r in ROOMS}
    st.rerun()
