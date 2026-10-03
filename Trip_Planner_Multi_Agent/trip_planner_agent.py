"""
Trip Planner Multi-Agent System
===============================
Pattern: Supervisor / Subagents-as-Tools Pattern with:
  1. Stateful Persistence via LangGraph SqliteSaver (`trip_planner_memory.db`)
  2. Human-in-the-Loop Approval Gates via LangChain `HumanInTheLoopMiddleware`

Workflow Stages:
  - Stage 1: Main Orchestrator Agent captures basic user info (Name, Budget, Source, Destination).
  - Stage 2 [HITL Gate #1]: Main Agent notifies user and triggers `find_hotels_subagent`.
    `HumanInTheLoopMiddleware` pauses for Plan Approval (`approve` or `edit`).
    Once approved/edited, the Hotel Sub-Agent searches available hotels.
  - Stage 3 [HITL Gate #2]: Main Agent displays found hotels and triggers `book_hotel_subagent`.
    `HumanInTheLoopMiddleware` pauses BEFORE booking so user can `approve` or `edit` the hotel choice.
    Once approved, the Hotel Sub-Agent books the hotel and returns a Booking ID.
  - Stage 4 [Conversational + HITL Gate #3]: Main Agent asks user for Flight Dates.
    Once dates are provided, Main Agent calls `search_flights_subagent` to fetch flight details,
    displays the flights, and triggers `book_flight_subagent`.
    `HumanInTheLoopMiddleware` pauses BEFORE booking so user can `approve` or `edit` the flight ticket.
  - Stage 5: Main Agent reverts to the user with the complete confirmed itinerary (Hotel + Flight + Budget).
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

# Load environment variables (.env in project root)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
load_dotenv()


# =====================================================================
# 1. MOCK DATABASES (HOTELS, FLIGHTS, AND CONFIRMED BOOKINGS)
# =====================================================================

HOTEL_DB: Dict[str, List[Dict[str, Any]]] = {
    "goa": [
        {
            "hotel_id": "H-GOA-01",
            "name": "Taj Exotica Resort & Spa Goa",
            "price_per_night": 220.0,
            "rating": 4.9,
            "amenities": "Private Beach, Pool, Breakfast Included, Spa",
        },
        {
            "hotel_id": "H-GOA-02",
            "name": "Lemon Tree Amarante Beach Resort",
            "price_per_night": 110.0,
            "rating": 4.5,
            "amenities": "Near Candolim Beach, Pool, Free Wi-Fi, Breakfast",
        },
        {
            "hotel_id": "H-GOA-03",
            "name": "Zostel Goa Calangute",
            "price_per_night": 45.0,
            "rating": 4.2,
            "amenities": "Backpacker Hostel, Cafe, Common Lounge, Free Wi-Fi",
        },
    ],
    "tokyo": [
        {
            "hotel_id": "H-TYO-01",
            "name": "Park Hyatt Shinjuku Tokyo",
            "price_per_night": 450.0,
            "rating": 4.9,
            "amenities": "Skyline Views, Luxury Spa, Fine Dining, Indoor Pool",
        },
        {
            "hotel_id": "H-TYO-02",
            "name": "Hotel Sunroute Plaza Shinjuku",
            "price_per_night": 140.0,
            "rating": 4.5,
            "amenities": "2 mins to Shinjuku Station, Free Wi-Fi, Buffet Breakfast",
        },
        {
            "hotel_id": "H-TYO-03",
            "name": "Shibuya Capsule Inn",
            "price_per_night": 65.0,
            "rating": 4.1,
            "amenities": "Modern Pods, Lockers, Lounge, Central Shibuya",
        },
    ],
    "paris": [
        {
            "hotel_id": "H-PAR-01",
            "name": "Pullman Paris Tour Eiffel",
            "price_per_night": 310.0,
            "rating": 4.7,
            "amenities": "Eiffel Tower Views, Fitness Center, French Brasserie",
        },
        {
            "hotel_id": "H-PAR-02",
            "name": "Ibis Styles Paris Montmartre",
            "price_per_night": 135.0,
            "rating": 4.4,
            "amenities": "Continental Breakfast Included, Free Wi-Fi, Metro Access",
        },
        {
            "hotel_id": "H-PAR-03",
            "name": "Generator Paris Hostel",
            "price_per_night": 60.0,
            "rating": 4.1,
            "amenities": "Rooftop Terrace, Cafe, Shared & Private Rooms",
        },
    ],
    "dubai": [
        {
            "hotel_id": "H-DXB-01",
            "name": "Atlantis The Palm Dubai",
            "price_per_night": 380.0,
            "rating": 4.9,
            "amenities": "Aquaventure Waterpark, Private Beach, Underwater Suites",
        },
        {
            "hotel_id": "H-DXB-02",
            "name": "Rove Downtown Dubai",
            "price_per_night": 125.0,
            "rating": 4.6,
            "amenities": "Burj Khalifa View, Outdoor Pool, 24/7 Gym, Free Wi-Fi",
        },
        {
            "hotel_id": "H-DXB-03",
            "name": "Premier Inn Dubai Airport",
            "price_per_night": 70.0,
            "rating": 4.3,
            "amenities": "Free Airport Shuttle, Rooftop Pool, Family Rooms",
        },
    ],
}

FLIGHT_DB: List[Dict[str, Any]] = [
    {
        "flight_number": "AI-805",
        "airline": "Air India",
        "departure_time": "08:30 AM",
        "arrival_time": "11:15 AM",
        "class": "Economy",
        "price": 180.0,
    },
    {
        "flight_number": "6E-2042",
        "airline": "IndiGo",
        "departure_time": "01:45 PM",
        "arrival_time": "04:20 PM",
        "class": "Economy Saver",
        "price": 135.0,
    },
    {
        "flight_number": "UK-981",
        "airline": "Vistara Premier",
        "departure_time": "07:00 PM",
        "arrival_time": "09:45 PM",
        "class": "Premium Economy",
        "price": 240.0,
    },
]

# Stores active bookings made during runs
BOOKINGS_DB: Dict[str, List[Dict[str, Any]]] = {
    "hotels": [],
    "flights": [],
}


# =====================================================================
# 2. DOMAIN TOOLS FOR SPECIALIZED SUB-AGENTS
# =====================================================================

@tool
def search_hotels(city: str, max_total_budget: float) -> str:
    """Search available hotels in the destination city and filter options suitable for the user's budget."""
    city_key = city.lower().strip()
    hotels = HOTEL_DB.get(city_key)

    # Provide fallback options if user enters a city not in static dict
    if not hotels:
        hotels = [
            {
                "hotel_id": f"H-{city_key[:3].upper()}-01",
                "name": f"Grand Plaza Hotel {city.title()}",
                "price_per_night": 210.0,
                "rating": 4.8,
                "amenities": "Luxury Spa, Pool, Breakfast Included, City Center",
            },
            {
                "hotel_id": f"H-{city_key[:3].upper()}-02",
                "name": f"Comfort Suites {city.title()}",
                "price_per_night": 115.0,
                "rating": 4.5,
                "amenities": "Free Wi-Fi, Complimentary Breakfast, Gym",
            },
            {
                "hotel_id": f"H-{city_key[:3].upper()}-03",
                "name": f"Traveler's Rest Inn {city.title()}",
                "price_per_night": 55.0,
                "rating": 4.2,
                "amenities": "Budget Friendly, Metro Access, 24h Front Desk",
            },
        ]

    lines = [f"Available Hotels in {city.title()} (Total Trip Budget: ${max_total_budget:.2f}):"]
    for h in hotels:
        est_3_nights = h["price_per_night"] * 3
        budget_status = "Within Budget" if est_3_nights <= max_total_budget * 0.75 else "High/Stretch"
        lines.append(
            f"- [{h['hotel_id']}] {h['name']} | ${h['price_per_night']:.2f}/night "
            f"(Est. 3 nights: ${est_3_nights:.2f} - {budget_status}) | "
            f"Rating: {h['rating']}★ | Amenities: {h['amenities']}"
        )
    return "\n".join(lines)


@tool
def book_hotel(
    guest_name: str,
    city: str,
    hotel_name: str,
    price_per_night: float,
    nights: int = 3,
) -> str:
    """Book a specific hotel in the destination city for the guest and generate a confirmation ID."""
    total_hotel_cost = round(price_per_night * nights, 2)
    booking_ref = f"HTL-{abs(hash(f'{guest_name}-{hotel_name}-{city}')) % 90000 + 10000}"

    record = {
        "booking_ref": booking_ref,
        "guest_name": guest_name,
        "city": city.title(),
        "hotel_name": hotel_name,
        "price_per_night": price_per_night,
        "nights": nights,
        "total_cost": total_hotel_cost,
        "status": "CONFIRMED",
    }
    BOOKINGS_DB["hotels"].append(record)

    return (
        f"HOTEL BOOKING CONFIRMED!\n"
        f"- Booking Reference: {booking_ref}\n"
        f"- Guest Name: {guest_name}\n"
        f"- Hotel: {hotel_name} ({city.title()})\n"
        f"- Duration: {nights} nights @ ${price_per_night:.2f}/night\n"
        f"- Total Hotel Cost: ${total_hotel_cost:.2f}"
    )


@tool
def search_flights(source: str, destination: str, travel_date: str) -> str:
    """Search available flights between source and destination for a specific travel date."""
    lines = [
        f"Available Flights from {source.title()} to {destination.title()} on {travel_date}:"
    ]
    for f in FLIGHT_DB:
        lines.append(
            f"- Flight {f['flight_number']} ({f['airline']}) | "
            f"Departs: {f['departure_time']} -> Arrives: {f['arrival_time']} | "
            f"Class: {f['class']} | Fare: ${f['price']:.2f}"
        )
    return "\n".join(lines)


@tool
def book_flight(
    passenger_name: str,
    source: str,
    destination: str,
    travel_date: str,
    flight_number: str,
    airline: str,
    price: float,
) -> str:
    """Reserve a flight ticket for the passenger on the specified date and issue a PNR."""
    pnr = f"FL-PNR-{abs(hash(f'{passenger_name}-{flight_number}-{travel_date}')) % 90000 + 10000}"

    record = {
        "pnr": pnr,
        "passenger_name": passenger_name,
        "source": source.title(),
        "destination": destination.title(),
        "travel_date": travel_date,
        "flight_number": flight_number,
        "airline": airline,
        "price": price,
        "status": "TICKETED",
    }
    BOOKINGS_DB["flights"].append(record)

    return (
        f"FLIGHT TICKET CONFIRMED!\n"
        f"- PNR Number: {pnr}\n"
        f"- Passenger: {passenger_name}\n"
        f"- Route: {source.title()} -> {destination.title()}\n"
        f"- Travel Date: {travel_date}\n"
        f"- Flight: {flight_number} ({airline})\n"
        f"- Ticket Fare: ${price:.2f}"
    )


# =====================================================================
# 3. BUILD SPECIALIZED SUB-AGENTS (HOTEL SUB-AGENT & FLIGHT SUB-AGENT)
# =====================================================================

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

HOTEL_SUBAGENT_PROMPT = """You are the specialized Hotel Sub-Agent.
You have two tools:
1. `search_hotels`: Use this when asked to find/search hotels in a destination city. Return ALL hotel options with their exact prices, ratings, and amenities.
2. `book_hotel`: Use this ONLY when explicitly instructed to book a specific hotel for a guest. Return the exact booking reference and total cost.
"""

hotel_subagent = create_agent(
    model=llm,
    tools=[search_hotels, book_hotel],
    system_prompt=HOTEL_SUBAGENT_PROMPT,
)

FLIGHT_SUBAGENT_PROMPT = """You are the specialized Flight Sub-Agent.
You have two tools:
1. `search_flights`: Use this when asked to search flights between a source and destination on a specific date. Return ALL available flights with flight numbers, airlines, departure/arrival times, and fares.
2. `book_flight`: Use this ONLY when explicitly instructed to book a specific flight for a passenger. Return the confirmed PNR and ticket details.
"""

flight_subagent = create_agent(
    model=llm,
    tools=[search_flights, book_flight],
    system_prompt=FLIGHT_SUBAGENT_PROMPT,
)


def _extract_text(response: dict) -> str:
    """Extract clean text string from a subagent response."""
    last_msg = response["messages"][-1]
    if isinstance(last_msg.content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in last_msg.content
        ]
        return "\n".join(p for p in parts if p)
    return str(last_msg.content)


# =====================================================================
# 4. DELEGATION TOOLS EXPOSED TO MAIN ORCHESTRATOR AGENT
# =====================================================================

@tool
def find_hotels_subagent(
    user_name: str,
    source: str,
    destination: str,
    total_budget: float,
) -> str:
    """Step 1 Delegation (Requires Plan Approval via HITL Middleware):
    Once basic user info (user_name, source, destination, total_budget) is captured,
    call this tool to approve the trip plan and spin up the Hotel Sub-Agent to find hotels."""
    instruction = (
        f"Search for available hotels in {destination} for traveler {user_name} "
        f"(traveling from {source} to {destination}) with a total trip budget of ${total_budget}."
    )
    res = hotel_subagent.invoke({"messages": [{"role": "user", "content": instruction}]})
    return _extract_text(res)


@tool
def book_hotel_subagent(
    user_name: str,
    destination: str,
    hotel_name: str,
    price_per_night: float,
    nights: int,
    available_hotels_summary: str,
) -> str:
    """Step 2 Delegation (Requires Hotel Booking Approval via HITL Middleware):
    After `find_hotels_subagent` returns hotel options, call this tool with the recommended hotel
    and a summary of all available hotels (`available_hotels_summary`) so the user can approve or edit the hotel booking."""
    instruction = (
        f"Book the hotel '{hotel_name}' in {destination} for guest '{user_name}' "
        f"for {nights} nights at ${price_per_night} per night."
    )
    res = hotel_subagent.invoke({"messages": [{"role": "user", "content": instruction}]})
    return _extract_text(res)


@tool
def search_flights_subagent(
    source: str,
    destination: str,
    travel_date: str,
) -> str:
    """Step 3 Delegation (Read-Only Search after user provides Flight Dates):
    Once the hotel is booked and the user provides their travel_date, call this tool to
    spin up the Flight Sub-Agent and retrieve available flights and their details."""
    instruction = (
        f"Search for available flights from {source} to {destination} on travel date {travel_date}."
    )
    res = flight_subagent.invoke({"messages": [{"role": "user", "content": instruction}]})
    return _extract_text(res)


@tool
def book_flight_subagent(
    passenger_name: str,
    source: str,
    destination: str,
    travel_date: str,
    flight_number: str,
    airline: str,
    price: float,
    available_flights_summary: str,
) -> str:
    """Step 4 Delegation (Requires Flight Booking Approval via HITL Middleware):
    After `search_flights_subagent` returns available flights, call this tool with the recommended
    flight details and `available_flights_summary` so the user can approve or edit the flight booking."""
    instruction = (
        f"Book flight {flight_number} ({airline}) from {source} to {destination} "
        f"on {travel_date} for passenger '{passenger_name}' at fare ${price}."
    )
    res = flight_subagent.invoke({"messages": [{"role": "user", "content": instruction}]})
    return _extract_text(res)


# =====================================================================
# 5. HUMAN-IN-THE-LOOP MIDDLEWARE & SQLITE CHECKPOINTER SETUP
# =====================================================================

hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        # Approval Gate 1: Plan Approval (Approve or Edit basic plan before spinning up Hotel Finder)
        "find_hotels_subagent": {
            "allowed_decisions": ["approve", "edit", "reject"],
        },
        # Approval Gate 2: Hotel Booking Approval (Approve or Edit hotel selection before booking)
        "book_hotel_subagent": {
            "allowed_decisions": ["approve", "edit", "reject"],
        },
        # Approval Gate 3: Flight Booking Approval (Approve or Edit flight selection before booking)
        "book_flight_subagent": {
            "allowed_decisions": ["approve", "edit", "reject"],
        },
    },
    description_prefix="HUMAN-IN-THE-LOOP APPROVAL REQUIRED",
)

MAIN_ORCHESTRATOR_PROMPT = """You are the Main Trip Planner Orchestrator Agent.
You coordinate a multi-agent travel planning workflow with a Hotel Sub-Agent and a Flight Sub-Agent.

Follow this EXACT sequential workflow:

STAGE 1 — CAPTURE BASIC USER INFORMATION:
- Check if you have all 4 mandatory pieces of basic information from the user:
  1. User Name (`user_name`)
  2. Total Trip Budget (`total_budget` in USD)
  3. Source City (`source`)
  4. Destination City (`destination`)
- If ANY of these 4 items are missing, politely ask the user to provide the missing details. Do NOT call any tools yet.

STAGE 2 — NOTIFY USER & PLAN APPROVAL (VIA `find_hotels_subagent`):
- As soon as you have all 4 items (`user_name`, `total_budget`, `source`, `destination`), write a clear message notifying the user that you have captured all their trip details and are submitting the Trip Plan for their approval before spinning up the Hotel Finder Sub-Agent.
- In the SAME response turn, call `find_hotels_subagent(user_name, source, destination, total_budget)`.
  (Note: `HumanInTheLoopMiddleware` will automatically pause execution on this tool call so the user can Approve or Edit the plan!)

STAGE 3 — SHOW HOTEL OPTIONS & REQUEST HOTEL BOOKING APPROVAL (VIA `book_hotel_subagent`):
- Once `find_hotels_subagent` returns the available hotels in the destination city:
  1. In your message content, clearly list ALL the available hotels returned by the sub-agent (including nightly price, 3-night cost, rating, and amenities) and state which hotel you recommend based on their budget.
  2. In the SAME response turn, call `book_hotel_subagent` with the recommended hotel's details (`user_name`, `destination`, `hotel_name`, `price_per_night`, `nights=3`, and `available_hotels_summary` listing all options).
  (Note: `HumanInTheLoopMiddleware` will automatically pause execution BEFORE booking so the user can Approve your recommended hotel or Edit the arguments to choose another hotel!)

STAGE 4 — CONFIRM HOTEL & ASK FOR FLIGHT DATES, THEN SEARCH & BOOK FLIGHT:
- Once `book_hotel_subagent` succeeds and returns the Hotel Booking Confirmation:
  1. If the user has NOT yet provided their preferred **Flight Date (`travel_date`)**, present the Hotel Booking Confirmation ID to the user and ask them for their preferred **Flight Travel Date** (e.g., YYYY-MM-DD). Do NOT call `search_flights_subagent` until the user provides a flight date.
  2. Once the user provides the flight travel date, call `search_flights_subagent(source, destination, travel_date)` to fetch available flights from the Flight Sub-Agent.
  3. Immediately after `search_flights_subagent` returns the flight options:
     - In your message content, display ALL available flight options with their flight numbers, airlines, departure/arrival times, and ticket prices.
     - In the SAME response turn, call `book_flight_subagent` with the best flight option (including `available_flights_summary` listing all flights).
     (Note: `HumanInTheLoopMiddleware` will automatically pause execution BEFORE booking the flight so the user can Approve or Edit the flight selection!)

STAGE 5 — FINAL TRIP CONFIRMATION:
- Once `book_flight_subagent` returns the confirmed Flight PNR, provide a comprehensive, well-formatted **Final Trip Confirmation Summary** to the user including:
  - Traveler Name & Route (Source -> Destination)
  - Confirmed Hotel Details (Hotel Name, Nights, Cost, Booking Reference ID)
  - Confirmed Flight Details (Airline, Flight Number, Travel Date, Timings, Fare, PNR)
  - Total Cost Spent vs. Total Trip Budget & Remaining Balance.
"""


def create_trip_planner_system(db_path: Optional[str] = None):
    """Instantiate the Main Orchestrator Agent backed by persistent SqliteSaver."""
    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "trip_planner_memory.db",
        )
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    main_agent = create_agent(
        model=llm,
        tools=[
            find_hotels_subagent,
            book_hotel_subagent,
            search_flights_subagent,
            book_flight_subagent,
        ],
        system_prompt=MAIN_ORCHESTRATOR_PROMPT,
        middleware=[hitl_middleware],
        checkpointer=checkpointer,
    )
    return main_agent, checkpointer, conn


# =====================================================================
# 6. HELPER FUNCTIONS FOR MULTI-TURN & HITL RESUME EXECUTION
# =====================================================================

def format_agent_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract clean assistant messages, tool outputs, and active HITL interrupt details."""
    messages = result.get("messages", [])

    # Find latest AI text message and latest ToolMessage
    latest_ai_text = ""
    latest_tool_text = ""

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not latest_ai_text:
            if isinstance(msg.content, list):
                text_parts = [
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in msg.content
                ]
                candidate = "\n".join(p for p in text_parts if p).strip()
            else:
                candidate = str(msg.content).strip()
            if candidate:
                latest_ai_text = candidate
        elif isinstance(msg, ToolMessage) and not latest_tool_text:
            latest_tool_text = str(msg.content).strip()

    interrupts = result.get("__interrupt__", [])
    interrupt_payload = None

    if interrupts:
        raw_int = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        action_requests = raw_int.get("action_requests", [])
        review_configs = raw_int.get("review_configs", [])

        if action_requests:
            act = action_requests[0]
            rev = review_configs[0] if review_configs else {}
            tool_name = act.get("name", "")

            # Determine friendly stage name for UI/Students
            stage_map = {
                "find_hotels_subagent": "Stage 2: Trip Plan Approval (Before Spinning Up Hotel Finder Sub-Agent)",
                "book_hotel_subagent": "Stage 3: Hotel Booking Approval (Before Booking Selected Hotel)",
                "book_flight_subagent": "Stage 4: Flight Booking Approval (Before Booking Selected Flight)",
            }
            interrupt_payload = {
                "stage": stage_map.get(tool_name, f"Approval Required for {tool_name}"),
                "tool_name": tool_name,
                "proposed_args": act.get("args", {}),
                "description": act.get("description", ""),
                "allowed_decisions": rev.get("allowed_decisions", ["approve", "edit", "reject"]),
                "latest_subagent_findings": latest_tool_text,
            }

    return {
        "status": "interrupted_for_approval" if interrupt_payload else "waiting_for_user",
        "assistant_message": latest_ai_text,
        "latest_subagent_output": latest_tool_text,
        "interrupt": interrupt_payload,
        "total_messages": len(messages),
    }
