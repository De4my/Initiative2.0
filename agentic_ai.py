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
        Analyze bus data at a given sequence and return all bus states.

        Input format:
        "sequence=120,bus=A1"

        Returns:
        JSON list of buses with:
        - bus_id
        - lat, lon
        - passenger count
        - is_broken
        """

        try:
            parts = query.split(",")
            seq = int(parts[0].split("=")[1])
            bus_id = parts[1].split("=")[1]

           #get lat lon of the bus that broken down
            broken_bus = df[(df["sequence"] == seq) & (df["bus id"] == bus_id)]
            broken_busLatLong = (broken_bus["lat"].values[0], broken_bus["lon"].values[0])
            
            CAPACITY = 40  # max seats

            broken_passenger = float(broken_bus["total passenger"].values[0])

            result = []

            for bus in df["bus id"].unique():

                df_bus = df[df["bus id"] == bus]
                row = df_bus[df_bus["sequence"] == seq]

                if not row.empty:
                    r = row.iloc[0]

                    # current bus location
                    current_latlon = (float(r["lat"]), float(r["lon"]))

                    # distance to broken bus
                    distance_km = geodesic(broken_busLatLong, current_latlon).km

                    # available seats
                    passenger = float(r.get("total passenger", 0))
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

    Return STRICT JSON:

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