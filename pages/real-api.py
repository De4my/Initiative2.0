import random
import numpy as np
import json
import re
import openrouteservice

import streamlit as st
from google.transit import gtfs_realtime_pb2
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
from agentic_ai import create_bus_agent, get_replacement_plan

# =========================
# CONFIG
# =========================
PAGE_REFRESH_MS = 4 * 60 * 1000
LIVE_DATA_REFRESH_SECONDS = 10

client = openrouteservice.Client(
    key="5b3ce3597851110001cf624833a4fa1e81bc489086625844b46df18c" 
)

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Live Bus Map", layout="wide")
st.title("⭕ Live Bus Tracker (GTFS Realtime)")

st_autorefresh(interval=PAGE_REFRESH_MS, key="page_refresh")

# =========================
# SESSION STATE
# =========================
if "passenger_counts" not in st.session_state:
    st.session_state.passenger_counts = {}

if "analysis_map" not in st.session_state:
    st.session_state.analysis_map = {}

if "breakdown_map" not in st.session_state:
    st.session_state.breakdown_map = {}

if "live_df" not in st.session_state:
    st.session_state.live_df = pd.DataFrame()

if "agent_error" not in st.session_state:
    st.session_state.agent_error = None

# =========================
# FETCH DATA
# =========================
@st.cache_data(ttl=10)
def fetch_data():
    URL = "https://api.data.gov.my/gtfs-realtime/vehicle-position/prasarana?category=rapid-bus-kl"

    try:
        response = requests.get(URL, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()

        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)

        vehicles = []
        for entity in feed.entity:
            if entity.HasField("vehicle"):
                v = entity.vehicle
                vehicles.append({
                    "vehicle_id": v.vehicle.id if v.vehicle else "unknown",
                    "route_id": v.trip.route_id if v.trip else "unknown",
                    "lat": v.position.latitude,
                    "lon": v.position.longitude
                })

        return pd.DataFrame(vehicles)

    except:
        return pd.DataFrame()

# =========================
# HELPERS
# =========================
def enrich_passenger_counts(df):
    if df.empty:
        df["passenger"] = []
        return df

    counts = st.session_state.passenger_counts

    for bus_id in df["vehicle_id"]:
        counts.setdefault(bus_id, random.randint(0, 39))

    df["passenger"] = df["vehicle_id"].map(counts)
    return df


def get_route(start, end):
    try:
        coords = [(start[1], start[0]), (end[1], end[0])]
        route = client.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson'
        )
        return route["features"][0]["geometry"]["coordinates"]
    except:
        return None


def convert_coords(coords):
    return [(lat, lon) for lon, lat in coords]


def parse_agent_response(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        cleaned = match.group(0)
        parsed = json.loads(cleaned)
        return parsed
    except:
        return None

# =========================
# MAP RENDER
# =========================
def render_map(df):
    if df.empty:
        center = [3.1390, 101.6869]  # KL fallback
    else:
        center = [df.iloc[0]["lat"], df.iloc[0]["lon"]]

    m = folium.Map(location=center, zoom_start=12)

    breakdown_map = st.session_state.breakdown_map

    all_replacements = [
        r for repl in breakdown_map.values() for r in repl
    ]

    for _, row in df.iterrows():
        bus_id = row["vehicle_id"]

        if bus_id in breakdown_map:
            color = "red"
            radius = 12
        elif bus_id in all_replacements:
            color = "green"
            radius = 10
        else:
            color = "#4E545C"
            radius = 8

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=f"{bus_id} | Route: {row['route_id']} | P: {row['passenger']}"
        ).add_to(m)

    # 🔗 ROUTES (correct mapping)
    for broken_bus, replacements in breakdown_map.items():

        broken_row = df[df["vehicle_id"] == broken_bus]
        if broken_row.empty:
            continue

        broken_coords = [
            float(broken_row.iloc[0]["lat"]),
            float(broken_row.iloc[0]["lon"]),
        ]

        for repl in replacements:
            repl_row = df[df["vehicle_id"] == repl]
            if repl_row.empty:
                continue

            repl_coords = [
                float(repl_row.iloc[0]["lat"]),
                float(repl_row.iloc[0]["lon"]),
            ]

            route = get_route(broken_coords, repl_coords)

            if route:
                folium.PolyLine(
                    locations=convert_coords(route),
                    color="green",
                    weight=3,
                    opacity=0.6,
                    tooltip=f"{broken_bus} → {repl}"
                ).add_to(m)

    st_folium(m, width=1200, height=600)

# =========================
# LOAD DATA
# =========================
df = fetch_data()

if df.empty and not st.session_state.live_df.empty:
    df = st.session_state.live_df.copy()
else:
    df = enrich_passenger_counts(df)
    st.session_state.live_df = df.copy()

bus_list = df["vehicle_id"].tolist() if not df.empty else []

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🚨 Bus Breakdown")

    selected_bus = st.selectbox(
        "Select Bus",
        ["Select"] + sorted(bus_list)
    )

    if st.button("Add Breakdown"):
        if selected_bus != "Select":

            if selected_bus not in st.session_state.breakdown_map:

                try:
                    agent = create_bus_agent(df)
                    result = agent.invoke({
                        "messages": [{"role": "user", "content": f"bus={selected_bus}"}]
                    })

                    parsed = parse_agent_response(
                        result["messages"][-1].content
                    )

                except:
                    parsed = get_replacement_plan(df, selected_bus)

            if parsed:
                # Store replacements
                if parsed["type"] == "single":
                    st.session_state.breakdown_map[selected_bus] = [parsed["bus_id"]]
                else:
                    st.session_state.breakdown_map[selected_bus] = parsed["bus_ids"]

                # ✅ Store FULL analysis (THIS IS WHAT YOU LOST)
                st.session_state.analysis_map[selected_bus] = parsed

    if st.button("Clear All"):
        st.session_state.breakdown_map = {}

    st.write("### Current Breakdown Map")
    st.json(st.session_state.breakdown_map)

# =========================
# LIVE MAP
# =========================
if hasattr(st, "fragment"):
    @st.fragment(run_every=f"{LIVE_DATA_REFRESH_SECONDS}s")
    def live():
        live_df = enrich_passenger_counts(fetch_data())
        st.session_state.live_df = live_df.copy()

        render_map(live_df)

        with st.expander("📊 Data"):
            st.dataframe(live_df)

    live()
else:
    render_map(df)

# =========================
# RESULTS SECTION (RESTORED)
# =========================
if st.session_state.breakdown_map:

    st.markdown("---")
    st.markdown("## 🔍 Bus Replacement Analysis")

    for broken_bus, parsed in st.session_state.analysis_map.items():

        st.markdown(f"### 🚨 Broken Bus: {broken_bus}")

        if parsed["type"] == "single":
            st.success(f"✅ Single Bus Solution: **{parsed['bus_id']}**")

            st.info(f"📝 Reason: {parsed.get('reason', 'Bus selected as replacement')}")

            replacement_bus = df[df["vehicle_id"] == parsed["bus_id"]]

            if not replacement_bus.empty:
                r = replacement_bus.iloc[0]

                col1, col2, col3 = st.columns(3)
                col1.metric("Passengers", int(r["passenger"]))
                col2.metric("Available Seats", int(40 - r["passenger"]))
                col3.metric("Route", r["route_id"])

        elif parsed["type"] == "multi":
            st.warning(f"⚠️ Multiple Buses Solution: {len(parsed['bus_ids'])} buses")

            st.info(f"📝 Reason: {parsed.get('reason', 'Single bus cannot fit all passengers')}")

            cols = st.columns(len(parsed["bus_ids"]))

            for idx, bus_id in enumerate(parsed["bus_ids"]):
                with cols[idx]:
                    replacement_bus = df[df["vehicle_id"] == bus_id]

                    if not replacement_bus.empty:
                        r = replacement_bus.iloc[0]

                        st.markdown(f"**Bus {idx+1}: {bus_id}**")
                        st.write(f"Route: {r['route_id']}")
                        st.write(f"Passengers: {int(r['passenger'])}")
                        st.write(f"Available: {int(40 - r['passenger'])}")