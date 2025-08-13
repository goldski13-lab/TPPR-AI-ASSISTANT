
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
import streamlit.components.v1 as components
import time, random, datetime

st.set_page_config(page_title="TPPR AI — Lab Digital Twin", layout="wide")
st.title("TPPR AI Safety Assistant — 3D Lab Digital Twin (Loop Layout)")

# --------- CONFIG ---------
ROOMS = [
    {"id": "mixing", "name": "Mixing Area", "icon": "🧪", "x": 0,  "y": 0},
    {"id": "pack",   "name": "Packaging",   "icon": "📦", "x": 6,  "y": 0},
    {"id": "boiler", "name": "Boiler Room", "icon": "🔥", "x": 0,  "y": 6},
    {"id": "waste",  "name": "Waste Treat.", "icon": "♻️", "x": 6, "y": 6},
]
ROOM_SIZE = 5.0
ROOM_HEIGHT = 0.8
DEFAULT_COLOR = "#cfe8ff"
WARNING_COLOR = "#ffb84d"
CRITICAL_COLOR = "#ff4c4c"

GASES = ["CH4", "H2S", "CO"]
BASELINES = {
    "CH4": (8, 2),
    "H2S": (2, 0.7),
    "CO" : (5, 1.5),
}
THRESHOLDS = {
    "CH4": {"warning": 25, "critical": 50},
    "H2S": {"warning": 10, "critical": 20},
    "CO" : {"warning": 30, "critical": 60},
}
COLOR_MAP = {
    "CH4": "#1f77b4",
    "H2S": "#2ca02c",
    "CO":  "#d62728",
}

# --------- INIT STATE ---------
if "df" not in st.session_state:
    now = pd.Timestamp.now().floor("min")
    rows = []
    for minute in range(180):  # 3 hours baseline
        ts = now + pd.Timedelta(minutes=minute)
        for r in ROOMS:
            for g in GASES:
                mu, sd = BASELINES[g]
                val = np.random.normal(mu, sd)
                rows.append({"timestamp": ts, "room": r["id"], "gas": g, "ppm": float(round(val, 2))})
    st.session_state.df = pd.DataFrame(rows)
    st.session_state.selected_room = None
    st.session_state.room_colors = {r["id"]: DEFAULT_COLOR for r in ROOMS}
    st.session_state.sim_history = []
    st.session_state.alert_active = {}  # (room, gas) -> 'warning'/'critical'
    st.session_state.incidents = []     # list of dicts with time, room, summary
    st.session_state.counters = {}      # (date_str, room, gas, level) -> count
    st.session_state.last_summary_text = None
    st.session_state.last_summary_room = None

df = st.session_state.df

# --------- HELPERS ---------
def latest_room_avg(room_id: str) -> float:
    d = df[df["room"] == room_id]
    if d.empty:
        return 0.0
    latest = d.sort_values("timestamp").groupby("gas").tail(1)
    return float(round(latest["ppm"].mean(), 2))

def room_any_status(room_id: str):
    d = df[df["room"] == room_id].sort_values("timestamp")
    if d.empty:
        return None
    latest = d.groupby("gas").tail(1)
    status = None
    for _, row in latest.iterrows():
        thr = THRESHOLDS[row["gas"]]
        if row["ppm"] >= thr["critical"]:
            return "critical"
        if row["ppm"] >= thr["warning"]:
            status = "warning"
    return status

def camera_for_room(room_id: str):
    r = next(rr for rr in ROOMS if rr["id"] == room_id)
    return dict(eye=dict(x=r["x"] + 7, y=r["y"] + 7, z=4),
                center=dict(x=r["x"] + ROOM_SIZE/2, y=r["y"] + ROOM_SIZE/2, z=ROOM_HEIGHT/2),
                up=dict(x=0, y=0, z=1))

def default_camera():
    return dict(eye=dict(x=10, y=10, z=8))

def spike_profile(gas: str, i: int, dur: int, peak: float):
    t = i / max(dur-1, 1)
    if gas == "CH4":
        return peak * (0.2 + 0.8 * t**1.5)
    if gas == "H2S":
        return peak * (0.3 + 0.7 * np.sin(np.pi * t))
    if gas == "CO":
        return peak * (0.6 if t < 0.2 else 1.0)
    return peak * t

def status_for(gas: str, value: float) -> str:
    thr = THRESHOLDS[gas]
    if value >= thr["critical"]:
        return "Critical"
    if value >= thr["warning"]:
        return "Warning"
    return "Safe"

def count_and_get_n(date_str, room_id, gas, level):
    key = (date_str, room_id, gas, level)
    st.session_state.counters[key] = st.session_state.counters.get(key, 0) + 1
    return st.session_state.counters[key]

def build_incident_summary(room_id: str) -> str:
    room = next(r for r in ROOMS if r["id"] == room_id)
    d = df[df["room"] == room_id]
    if d.empty:
        return f"Incident Report — {room['name']} {room['icon']}\nNo data."
    latest_ts = d["timestamp"].max()
    snap = d[d["timestamp"] == latest_ts]
    date_str = pd.to_datetime(latest_ts).strftime("%Y-%m-%d")
    lines = [f"Incident Report — {room['name']} {room['icon']}", f"Time: {pd.to_datetime(latest_ts).strftime('%Y-%m-%d %H:%M')}"]
    for g in GASES:
        gd = snap[snap["gas"] == g]
        if gd.empty:
            continue
        v = float(gd["ppm"].iloc[0])
        s = status_for(g, v)
        # increment counters if warning/critical
        suffix = ""
        if s == "Warning":
            n = count_and_get_n(date_str, room_id, g, "warning")
            suffix = f" — {n}{ordinal(n)} warning event today"
        elif s == "Critical":
            n = count_and_get_n(date_str, room_id, g, "critical")
            suffix = f" — {n}{ordinal(n)} critical event today"
        lines.append(f"{g}: {round(v,2)} ppm ({s}){suffix}")
    return "\n".join(lines)

def ordinal(n:int) -> str:
    return "th" if 11<=n%100<=13 else {1:"st",2:"nd",3:"rd"}.get(n%10, "th")

# --------- TOP METRICS ---------
st.subheader("Live Room Averages")
mcols = st.columns(len(ROOMS))
for i, r in enumerate(ROOMS):
    avg_val = latest_room_avg(r["id"])
    drr = df[(df["room"] == r["id"])].sort_values("timestamp")
    prev_avg = 0.0
    if len(drr) >= len(GASES)*2:
        latest = drr.groupby("gas").tail(1)["ppm"].mean()
        prev = drr.groupby("gas").nth(-2)["ppm"].mean()
        prev_avg = prev
    delta = round(avg_val - prev_avg, 2)
    mcols[i].metric(f"{r['icon']} {r['name']}", f"{avg_val} ppm", f"{delta} ppm")

st.markdown("---")
left, center, right = st.columns([1.2, 2.6, 1.6])

# --------- CONTROLS + LEGEND + INCIDENT HISTORY ---------
with left:
    st.header("Controls")
    if st.button("Simulate Live Gas Event"):
        room = random.choice(ROOMS)
        gases_to_spike = random.sample(GASES, k=random.choice([1, 2, 3]))
        severity = random.choices(["warning", "critical"], weights=[0.6, 0.4])[0]
        dur = 8 if severity == "warning" else 14
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
                        val += spike_profile(g, i, dur, peaks[g]) + np.random.normal(0, sd*0.5)
                    new_rows.append({"timestamp": ts, "room": r["id"], "gas": g, "ppm": float(round(val, 2))})
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame(new_rows)], ignore_index=True)

        status = room_any_status(room["id"])
        if status == "critical":
            st.session_state.room_colors[room["id"]] = CRITICAL_COLOR
        elif status == "warning":
            st.session_state.room_colors[room["id"]] = WARNING_COLOR
        else:
            st.session_state.room_colors[room["id"]] = WARNING_COLOR

        st.session_state.selected_room = room["id"]

        # Auto-build and store incident summary
        summary = build_incident_summary(room["id"])
        st.session_state.last_summary_text = summary
        st.session_state.last_summary_room = room["id"]
        st.session_state.incidents.append({"time": pd.Timestamp.now(), "room": room["id"], "summary": summary})
        # Limit history to last 20 to keep tidy
        st.session_state.incidents = st.session_state.incidents[-20:]

        st.session_state.sim_history.append({
            "time": pd.Timestamp.now(),
            "room": room["id"],
            "gases": gases_to_spike,
            "severity": severity
        })
        st.experimental_rerun()

    if st.button("Reset Data"):
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("**Simulation history (latest 10)**")
    for h in list(reversed(st.session_state.sim_history[-10:])):
        gases_str = ", ".join(h["gases"]) if isinstance(h.get("gases", []), list) else "-"
        st.write(f"- {h['time'].strftime('%H:%M:%S')} — {h['room']} — {gases_str} — {h['severity']}")

    st.markdown("---")
    st.header("Legend")
    thr_df = pd.DataFrame({
        "Gas": ["CH4", "H2S", "CO"],
        "Warning": [THRESHOLDS["CH4"]["warning"], THRESHOLDS["H2S"]["warning"], THRESHOLDS["CO"]["warning"]],
        "Critical": [THRESHOLDS["CH4"]["critical"], THRESHOLDS["H2S"]["critical"], THRESHOLDS["CO"]["critical"]],
        "Color": ["CH4", "H2S", "CO"]
    })
    st.dataframe(thr_df, hide_index=True, use_container_width=True)
    st.caption("Room color: amber = warning, red = critical.")
    st.markdown("---")
    st.caption("Tip: Click a room in the 3D map to zoom in. Click the same room again to zoom back out.")

# Sidebar Incident History
st.sidebar.header("Incident History")
if st.session_state.incidents:
    for inc in reversed(st.session_state.incidents):
        room = next(r for r in ROOMS if r["id"] == inc["room"])
        with st.sidebar.expander(f"{inc['time'].strftime('%H:%M:%S')} — {room['name']} {room['icon']}"):
            st.code(inc["summary"])
else:
    st.sidebar.info("No incidents yet in this session.")

# --------- 3D LOOP LAYOUT ---------
mesh_traces = []
marker_x, marker_y, marker_z, marker_text, marker_room_ids = [], [], [], [], []

def add_room_mesh(room_id, name, icon, x0, y0, size=ROOM_SIZE, height=ROOM_HEIGHT, color=DEFAULT_COLOR):
    vx = [x0, x0+size, x0+size, x0,   x0, x0+size, x0+size, x0]
    vy = [y0, y0,      y0+size, y0+size, y0, y0,      y0+size, y0+size]
    vz = [0, 0, 0, 0,  height, height, height, height]
    faces = [[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    i, j, k = zip(*faces)
    avg_ppm = latest_room_avg(room_id)
    ht = f"{icon} {name} — avg {avg_ppm} ppm"
    mesh_traces.append(go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k,
                                 color=color, opacity=0.95,
                                 name=name, hovertext=ht, hoverinfo="text"))
    cx, cy, cz = x0 + size/2, y0 + size/2, height/2
    marker_x.append(cx); marker_y.append(cy); marker_z.append(cz)
    marker_text.append(f"{icon} {name}")
    marker_room_ids.append(room_id)

for r in ROOMS:
    add_room_mesh(r["id"], r["name"], r["icon"], r["x"], r["y"], color=st.session_state.room_colors[r["id"]])

marker_trace = go.Scatter3d(
    x=marker_x, y=marker_y, z=marker_z,
    mode="markers+text",
    marker=dict(size=28, color="rgba(0,0,0,0)"),
    text=marker_text, textposition="top center",
    hoverinfo="text"
)

fig = go.Figure(data=mesh_traces + [marker_trace])
cam = camera_for_room(st.session_state.selected_room) if st.session_state.selected_room else default_camera()
fig.update_layout(
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    height=560, margin=dict(l=0, r=0, t=10, b=0),
    scene_camera=cam
)

with st.container():
    st.subheader("3D Floorplan (loop layout) — Click a room")
    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=560)
    st.plotly_chart(fig, use_container_width=True)

# Handle clicks
if clicked and isinstance(clicked, list) and len(clicked) > 0:
    pt = clicked[0]
    curve = pt.get("curveNumber", None)
    pnum = pt.get("pointNumber", None)
    marker_curve_idx = len(mesh_traces)
    if curve == marker_curve_idx and pnum is not None and 0 <= pnum < len(marker_room_ids):
        rid = marker_room_ids[pnum]
        if st.session_state.selected_room == rid:
            st.session_state.selected_room = None
        else:
            st.session_state.selected_room = rid
        st.experimental_rerun()

# --------- CSS for flashing badges ---------
st.markdown("""
<style>
.badge { display:inline-block; padding:4px 8px; border-radius:12px; font-weight:600; margin-right:6px; }
.badge-warning { background: rgba(255,184,77,0.25); color:#8a4b00; border:1px solid #ffb84d; }
.badge-critical { background: rgba(255,76,76,0.25); color:#7a0000; border:1px solid #ff4c4c; }
@keyframes flash { 0% {opacity: 1;} 50% {opacity: 0.35;} 100% {opacity: 1;} }
.flash { animation: flash 1.2s infinite; }
.copy-btn { padding:6px 10px; border:1px solid #ddd; border-radius:8px; background:#fafafa; cursor:pointer; }
.copy-btn:hover { background:#f0f0f0; }
</style>
""", unsafe_allow_html=True)

def add_threshold_lines(fig, gases, opacity=0.7, annotate=True):
    shapes = []
    annotations = []
    colors = {"warning": "rgba(255,184,77,{})".format(opacity),
              "critical": "rgba(255,76,76,{})".format(opacity)}
    for g in gases:
        w = THRESHOLDS[g]["warning"]
        c = THRESHOLDS[g]["critical"]
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=w, y1=w,
                           line=dict(color=colors["warning"], width=2, dash="solid")))
        shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=c, y1=c,
                           line=dict(color=colors["critical"], width=2, dash="dash")))
        if annotate:
            annotations.append(dict(x=1.0, xref="paper", y=w, yref="y",
                                    text=f"{g} Warning: {w} ppm", showarrow=False, xanchor="right",
                                    font=dict(size=10, color="#8a4b00"), bgcolor="rgba(255,184,77,0.15)"))
            annotations.append(dict(x=1.0, xref="paper", y=c, yref="y",
                                    text=f"{g} Critical: {c} ppm", showarrow=False, xanchor="right",
                                    font=dict(size=10, color="#7a0000"), bgcolor="rgba(255,76,76,0.15)"))
    fig.update_layout(shapes=shapes, annotations=annotations)

def update_alert_badges(room_id):
    room_df = df[df["room"] == room_id].sort_values("timestamp")
    for g in GASES:
        gdf = room_df[room_df["gas"] == g]
        if gdf.empty:
            continue
        latest = float(gdf["ppm"].iloc[-1])
        thr = THRESHOLDS[g]
        key = (room_id, g)
        if latest >= thr["critical"]:
            st.session_state.alert_active[key] = "critical"
        elif latest >= thr["warning"]:
            st.session_state.alert_active[key] = "warning"
        else:
            if key in st.session_state.alert_active:
                del st.session_state.alert_active[key]

    # Render badges
    for (r, g), level in st.session_state.alert_active.items():
        if r != room_id:
            continue
        cls = "badge-critical flash" if level == "critical" else "badge-warning flash"
        label = "CRITICAL" if level == "critical" else "WARNING"
        st.markdown(f"<span class='badge {cls}'>{g} {label}</span>", unsafe_allow_html=True)

def copy_to_clipboard(summary_text: str, key: str):
    # Use a lightweight HTML button to access Clipboard API
    safe = summary_text.replace("\n", "\\n").replace("'", "\'")
    html_btn = f"""
        <button class="copy-btn" onclick="navigator.clipboard.writeText('{safe}')">Copy incident summary</button>
    """
    components.html(html_btn, height=40)

# --------- DETAIL VIEW ---------
sel = st.session_state.selected_room
if sel is None:
    st.header("Inspector")
    st.write("Click a room to see its detectors (CH₄, H₂S, CO), live readings, forecasts, and incident summary.")
else:
    room = next(r for r in ROOMS if r["id"] == sel)
    st.header(f"Room: {room['icon']} {room['name']}")
    room_df = df[df["room"] == sel].sort_values("timestamp")

    # Alert badges (flash while out of range)
    update_alert_badges(sel)

    # Auto-show last summary if it belongs to this room
    if st.session_state.last_summary_room == sel and st.session_state.last_summary_text:
        st.subheader("Latest incident summary")
        st.code(st.session_state.last_summary_text)
        copy_to_clipboard(st.session_state.last_summary_text, key=f"copy_{sel}")

    # Metric cards per gas
    c1, c2, c3 = st.columns(3)
    for col, gas in zip([c1, c2, c3], GASES):
        gdf = room_df[room_df["gas"] == gas]
        latest = float(gdf["ppm"].iloc[-1]) if not gdf.empty else 0.0
        prev = float(gdf["ppm"].iloc[-2]) if len(gdf) >= 2 else latest
        delta = round(latest - prev, 2)
        thr = THRESHOLDS[gas]
        status = "OK"
        if latest >= thr["critical"]:
            status = "CRITICAL"
        elif latest >= thr["warning"]:
            status = "WARNING"
        col.metric(f"{gas} — {status}", f"{latest} ppm", f"{delta} ppm")

    # Line chart of all gases (measured) + threshold overlays
    recent = room_df.groupby("gas").tail(150)
    fig_line = px.line(recent, x="timestamp", y="ppm", color="gas",
                       color_discrete_map=COLOR_MAP, markers=False,
                       title="Recent readings (with thresholds)")
    add_threshold_lines(fig_line, GASES, opacity=0.7, annotate=True)
    fig_line.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_line, use_container_width=True)

    # Mini-forecast per gas (linear) + threshold overlays full opacity
    st.subheader("Short-term forecast (next 5 mins)")
    fcols = st.columns(3)
    for col, gas in zip(fcols, GASES):
        gdf = room_df[room_df["gas"] == gas].tail(30)
        if len(gdf) >= 3:
            x = np.arange(len(gdf))
            y = gdf["ppm"].values.astype(float)
            m, b = np.polyfit(x, y, 1)
            future_x = np.arange(len(gdf), len(gdf) + 5)
            future_y = m * future_x + b
            times = list(gdf["timestamp"]) + [gdf["timestamp"].iloc[-1] + pd.Timedelta(minutes=i) for i in range(1, 6)]
            dfp = pd.DataFrame({
                "time": times,
                "ppm": list(y) + list(future_y),
                "type": ["measured"] * len(y) + ["predicted"] * 5
            })
            fc = px.line(dfp, x="time", y="ppm", color="type", markers=True,
                         title=f"{gas} forecast (with thresholds)",
                         color_discrete_map={"measured": COLOR_MAP[gas], "predicted": "#7f7f7f"})
            add_threshold_lines(fc, [gas], opacity=1.0, annotate=True)
            fc.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            col.plotly_chart(fc, use_container_width=True)
        else:
            col.info(f"Not enough data to forecast {gas} yet.")

# --------- GLOW FADE ---------
if any(c in [WARNING_COLOR, CRITICAL_COLOR] for c in st.session_state.room_colors.values()):
    time.sleep(1.5)
    st.session_state.room_colors = {r["id"]: DEFAULT_COLOR for r in ROOMS}
    st.experimental_rerun()
