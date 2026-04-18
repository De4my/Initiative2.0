import streamlit as st
import pandas as pd
import pydeck as pdk
import time
import numpy as np
from agentic_ai import create_bus_agent
from dotenv import load_dotenv
load_dotenv()

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("RouteA_Final.xlsx", sheet_name="Route_Coordinates")
df2 = pd.read_excel("RouteA_Final.xlsx", sheet_name="Bus_Stops")


#new variable
# Clean route dataframe
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
df = df.dropna(subset=["lat", "lon"])
df["datetime"] = pd.to_datetime(df["datetime"])

# Clean stops dataframe
df2.columns = df2.columns.str.strip().str.lower()
df2 = df2.rename(columns={"latitude": "lat", "longitude": "lon", "lng": "lon"})
df2 = df2.dropna(subset=["lat", "lon"])

agent = create_bus_agent(df)
# =========================
# COLOR PER BUS
# =========================
bus_list = df["bus id"].unique()

bus_colors = {
    bus: [int(x) for x in np.random.randint(0, 256, 3)]
    for bus in bus_list
}


# =========================
# SESSION STATE
# =========================
if "playing" not in st.session_state:
    st.session_state.playing = False

if "current_seq" not in st.session_state:
    st.session_state.current_seq = int(df["sequence"].min())

if "sim_time" not in st.session_state:
    start_row = df[df["sequence"] == df["sequence"].min()].iloc[0]
    st.session_state.sim_time = pd.to_datetime(start_row["datetime"])

if "stopped_buses" not in st.session_state:
    st.session_state.stopped_buses = {}  # {bus_id: seq}

if "selected_bus" not in st.session_state:
    st.session_state.selected_bus = []


# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.header("Controls")

    play = st.button("▶ Play")
    stop = st.button("⏹ Stop")

    speed = st.slider(
        "Speed", min_value=0.1, max_value=2.0, value=0.5, step=0.1
    )

    selected_seq = st.slider(
        "Sequence",
        min_value=int(df["sequence"].min()),
        max_value=int(df["sequence"].max()),
        value=st.session_state.current_seq,
        step=1
    )

    st.markdown("### Breakdown bus")
    seq_input = st.text_input("Enter Bus ID")
    go_button = st.button("Send")
    if go_button:
        if seq_input in bus_list:

            st.session_state.stopped_buses[seq_input] = st.session_state.current_seq

            query = f"sequence={st.session_state.current_seq},bus={seq_input}"

            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            })

            response_text = result["messages"][-1].content
            st.session_state.agent_result = response_text

            import json

            try:
                parsed = json.loads(response_text)

                if parsed["type"] == "single":
                    st.session_state.selected_buses = [parsed["bus_id"]]

                elif parsed["type"] == "multi":
                    st.session_state.selected_buses = parsed["bus_ids"]

            except:
                st.session_state.selected_buses = []


# =========================
# BUTTON LOGIC
# =========================
if play:
    st.session_state.playing = True

if stop:
    st.session_state.playing = False
    st.session_state.current_seq = int(df["sequence"].min())

    start_row = df[df["sequence"] == df["sequence"].min()].iloc[0]
    st.session_state.sim_time = pd.to_datetime(start_row["datetime"])

if not st.session_state.playing:
    st.session_state.current_seq = selected_seq


# =========================
# MAIN UI
# =========================
st.title("Bus Route Simulation")

clock_placeholder = st.empty()
map_placeholder = st.empty()
info_placeholder = st.empty()


# =========================
# MAP FUNCTION
# =========================
def render_map(seq):

    layers = []

    for bus in bus_list:

        df_bus = df[df["bus id"] == bus]

        df_path = df_bus[df_bus["sequence"] <= seq]
        # if this bus is stopped → freeze at stored sequence
        if bus in st.session_state.stopped_buses:
            freeze_seq = st.session_state.stopped_buses[bus]

            df_current = df_bus[df_bus["sequence"] == freeze_seq]
            df_path = df_bus[df_bus["sequence"] <= freeze_seq]
        else:
            df_current = df_bus[df_bus["sequence"] == seq]
            df_path = df_bus[df_bus["sequence"] <= seq]

        # DEFAULT color
        color = bus_colors.get(bus, [100, 100, 100])
        radius = 60

        # ❌ BROKEN BUS → RED + BIG
        if bus in st.session_state.stopped_buses:
            color = [255, 0, 0]
            radius = 100

        # 🚑 SELECTED BUS → GREEN + BIGGER
        if bus in st.session_state.get("selected_buses", []):
            color = [0, 255, 0]
            radius = 110

        # past route
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_path,
                get_position="[lon, lat]",
                get_radius=10,
                get_fill_color=color,
                pickable=True   # ✅ IMPORTANT
            )
        )

        # current position
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_current,
                get_position="[lon, lat]",
                get_radius=radius,
                get_fill_color=color,
                pickable=True   # ✅ IMPORTANT
            )
        )

    # stops
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df2,
            get_position="[lon, lat]",
            get_radius=25,
            get_fill_color=[255, 0, 0],
            pickable=True   # ✅ IMPORTANT
        )
    )

    view_state = pdk.ViewState(
        latitude=df["lat"].mean(),
        longitude=df["lon"].mean(),
        zoom=12
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="road",
        tooltip={
            "html": """
            <b>Bus:</b> {bus id} <br/>
            <b>Sequence:</b> {sequence} <br/>
            <b>Passengers:</b> {total passenger}
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    )

    # CLOCK
    time_str = st.session_state.sim_time.strftime("%H:%M:%S")

    clock_placeholder.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: flex-end;
            font-size: 36px;
            font-weight: bold;
            font-family: monospace;
            margin-bottom: -40px;
        ">
            {time_str}
        </div>
        """,
        unsafe_allow_html=True
    )

    map_placeholder.pydeck_chart(deck)

    info_placeholder.write(f"Sequence: {seq}")

if "agent_result" in st.session_state:
    st.subheader("🤖 AI Recommendation")
    st.write(st.session_state.agent_result)


# =========================
# AUTO PLAY
# =========================
if st.session_state.playing:

    sequences = sorted(df["sequence"].unique())
    start_idx = sequences.index(st.session_state.current_seq)

    for seq in sequences[start_idx:]:

        if not st.session_state.playing:
            break

        st.session_state.current_seq = seq
        st.session_state.sim_time += pd.Timedelta(seconds=1)

        render_map(seq)
        time.sleep(speed)

    st.session_state.playing = False

else:
    render_map(st.session_state.current_seq)