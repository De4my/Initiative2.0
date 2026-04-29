import json
from deepagents import create_deep_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
import geopy.distance
from geopy.distance import geodesic


# =========================
# TOOL WRAPPER
# =========================
def bus_analysis_tool_wrapper(df):

    @tool
    def bus_analysis_tool(query: str):
        """
        Analyze bus data and return all bus states.

        Input format:
        "bus=A1"

        Returns:
        JSON list of buses with:
        - bus_id
        - lat, lon
        - passenger count
        - is_broken
        """

        try:
            bus_id = query.split("=")[1]

           #get lat lon of the bus that broken down
            broken_bus = df[df["vehicle_id"] == bus_id]
            broken_busLatLong = (broken_bus["lat"].values[0], broken_bus["lon"].values[0])
            
            CAPACITY = 40  # max seats

            broken_passenger = float(broken_bus["passenger"].values[0])

            result = []

            for bus in df["vehicle_id"].unique():

                df_bus = df[df["vehicle_id"] == bus]
                row = df_bus

                if not row.empty:
                    r = row.iloc[0]

                    # current bus location
                    current_latlon = (float(r["lat"]), float(r["lon"]))

                    # distance to broken bus
                    distance_km = geodesic(broken_busLatLong, current_latlon).km

                    # available seats
                    passenger = float(r.get("passenger", 0))
                    available_seat = CAPACITY - passenger

                    result.append({
                        "bus_id": bus,
                        "lat": current_latlon[0],
                        "lon": current_latlon[1],
                        "passenger": passenger,
                        "broken_passenger": broken_passenger,
                        "available_seat": available_seat,
                        "distance_km": distance_km,
                        "is_broken": bus == bus_id
                    })

            return json.dumps(result)

        except Exception as e:
            return f"Error: {str(e)}"

    return bus_analysis_tool


def get_replacement_plan(df, broken_bus_id):
    broken_bus = df[df["vehicle_id"] == broken_bus_id]
    if broken_bus.empty:
        raise ValueError(f"Bus '{broken_bus_id}' not found")

    broken_row = broken_bus.iloc[0]
    broken_latlon = (float(broken_row["lat"]), float(broken_row["lon"]))
    required_passengers = float(broken_row.get("passenger", 0))
    capacity = 40

    candidates = []
    for _, row in df.iterrows():
        bus_id = row["vehicle_id"]
        if bus_id == broken_bus_id:
            continue

        passenger = float(row.get("passenger", 0))
        available_seat = capacity - passenger
        if available_seat <= 0:
            continue

        distance_km = geodesic(
            broken_latlon,
            (float(row["lat"]), float(row["lon"]))
        ).km

        candidates.append({
            "bus_id": bus_id,
            "distance_km": distance_km,
            "passenger": passenger,
            "available_seat": available_seat,
        })

    candidates.sort(key=lambda x: (x["distance_km"], x["passenger"], -x["available_seat"]))

    for bus in candidates:
        if bus["available_seat"] >= required_passengers:
            return {
                "type": "single",
                "bus_id": bus["bus_id"],
                "reason": (
                    f"Selected the nearest bus with enough capacity "
                    f"({int(bus['available_seat'])} available seats)."
                ),
            }

    selected = []
    total_available = 0
    for bus in candidates:
        selected.append(bus)
        total_available += bus["available_seat"]
        if total_available >= required_passengers:
            return {
                "type": "multi",
                "bus_ids": [bus["bus_id"] for bus in selected],
                "reason": (
                    f"No single bus had enough capacity, so the nearest combination of "
                    f"{len(selected)} buses was selected with {int(total_available)} total available seats."
                ),
            }

    return {
        "type": "multi",
        "bus_ids": [bus["bus_id"] for bus in selected],
        "reason": "Not enough available capacity across the current fleet.",
    }


# =========================
# AGENT CREATION
# =========================
def create_bus_agent(df):

    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
    )

    tool = bus_analysis_tool_wrapper(df)

    system_prompt = """
    You are a smart bus operation AI.

    A bus has broken down.

    Use the tool to retrieve current bus data.

    Then choose the BEST replacement bus based on:
    - closest distance
    - lowest passenger load
    - not broken
Your task:

1. FIRST try to find ONE bus that:
   - is closest
   - has enough available seats to take ALL passengers

2. IF no single bus can handle all passengers:
   - choose MULTIPLE buses (2 or more)
   - their TOTAL available seats must be >= required passengers
   - prioritize closest + least loaded buses

    Rules:
    - Never select broken bus
    - Minimize number of buses used
    - Prefer fewer buses over many

    Return STRICT JSON.
    Output only one JSON object.
    Do not use markdown.
    Do not wrap the JSON in code fences.
    Do not add explanation before or after the JSON.

    IF one bus:
    {
    "type": "single",
    "bus_id": "B2",
    "reason": "..."
    }

    IF multiple buses:
    {
    "type": "multi",
    "bus_ids": ["B2", "C1"],
    "reason": "..."
    }
    
    """

    agent = create_deep_agent(
        model=llm,
        tools=[tool],
        system_prompt=system_prompt,
    )

    return agent
