import os
import uuid
from datetime import date
from typing import Annotated, Sequence, TypedDict

import operator
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
import streamlit as st

load_dotenv()

def configured_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except (FileNotFoundError, KeyError, AttributeError):
        return None


OPENAI_API_KEY = configured_value("OPENAI_API_KEY")
TAVILY_API_KEY = configured_value("TAVILY_API_KEY")
DUFFEL_ACCESS_TOKEN = configured_value("DUFFEL_ACCESS_TOKEN")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


@tool
def get_current_date_tool() -> str:
    """Return today's date so the assistant can plan future travel requests."""
    return f"Today's date is {date.today().isoformat()}."


if TAVILY_API_KEY:
    tavily_search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
else:
    @tool
    def tavily_search_tool(query: str) -> str:
        """Search the web, or explain that web search is unavailable without an API key."""
        return "Tavily API key is missing. Add TAVILY_API_KEY to your .env file to enable web search."


DUFFEL_API_BASE_URL = "https://api.duffel.com"
if DUFFEL_ACCESS_TOKEN:
    duffel_headers = {
        "Authorization": f"Bearer {DUFFEL_ACCESS_TOKEN}",
        "Duffel-Version": "v2",
        "Content-Type": "application/json",
    }
else:
    duffel_headers = None


@tool
def search_flights_tool(
    origin_code: str,
    destination_code: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    travel_class: str = "economy",
    currency: str = "USD",
    max_offers: int = 5,
) -> str:
    """Search flight offers using Duffel's Air Offer Requests API."""
    if not duffel_headers:
        return "Duffel access token is missing. Add DUFFEL_ACCESS_TOKEN to your .env file to enable flight search."

    from datetime import datetime

    try:
        departure = datetime.strptime(departure_date, "%Y-%m-%d").date()
        return_trip = datetime.strptime(return_date, "%Y-%m-%d").date() if return_date else None
    except ValueError:
        return "Invalid date. Use YYYY-MM-DD, for example 2027-06-07."

    if departure < date.today():
        return f"Departure date {departure_date} is in the past. Please provide a future date."
    if return_trip and return_trip < departure:
        return "Return date must be on or after the departure date."

    slices = [{
        "origin": origin_code.upper(),
        "destination": destination_code.upper(),
        "departure_date": departure_date,
    }]
    if return_date:
        slices.append({
            "origin": destination_code.upper(),
            "destination": origin_code.upper(),
            "departure_date": return_date,
        })

    payload = {
        "data": {
            "slices": slices,
            "passengers": [{"type": "adult"} for _ in range(adults)],
            "cabin_class": travel_class.lower(),
        }
    }

    response = requests.post(
        f"{DUFFEL_API_BASE_URL}/air/offer_requests",
        headers=duffel_headers,
        json=payload,
        timeout=60,
    )
    if not response.ok:
        try:
            details = response.json().get("errors", response.text)
        except ValueError:
            details = response.text
        return f"Duffel flight search failed ({response.status_code}): {details}"

    offers = response.json().get("data", {}).get("offers", [])
    if not offers:
        return f"No Duffel flight offers found for {origin_code} to {destination_code}."

    results = []
    for offer in offers[:max_offers]:
        owner = offer.get("owner", {}).get("name", "Unknown airline")
        price = offer.get("total_amount", "N/A")
        currency = offer.get("total_currency", currency)
        offer_slices = offer.get("slices", [])
        itinerary = offer_slices[0] if offer_slices else {}
        segments = itinerary.get("segments", [])
        if segments:
            departure_time = segments[0].get("departing_at", "N/A")[:16].replace("T", " ")
            arrival_time = segments[-1].get("arriving_at", "N/A")[:16].replace("T", " ")
        else:
            departure_time, arrival_time = "N/A", "N/A"
        results.append(
            f"{owner} | {departure_time} -> {arrival_time} | {price} {currency} | "
            f"Offer ID: {offer.get('id', 'N/A')}"
        )

    return "Found Duffel flight offers:\n- " + "\n- ".join(results)


@tool
def search_hotels_tool(city_name: str, check_in_date: str, check_out_date: str, adults: int = 1) -> str:
    """Search hotels through Duffel Stays if the token has Stays access enabled."""
    if not duffel_headers:
        return "Duffel access token is missing. Add DUFFEL_ACCESS_TOKEN to your .env file to enable hotel search."

    payload = {
        "data": {
            "location": {"geography": {"city_name": city_name}},
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "rooms": [{"adults": adults}],
        }
    }

    response = requests.post(
        f"{DUFFEL_API_BASE_URL}/stays/search",
        headers=duffel_headers,
        json=payload,
        timeout=60,
    )

    if response.status_code == 403:
        return (
            "Duffel Stays access is forbidden for this API token (HTTP 403). "
            "Enable Stays in the Duffel dashboard or use a token with Stays access."
        )

    if not response.ok:
        try:
            details = response.json().get("errors", response.text)
        except ValueError:
            details = response.text
        return f"Duffel hotel search failed (HTTP {response.status_code}): {details}"

    stays = response.json().get("data", {}).get("results", [])
    if not stays:
        return f"No Duffel stays found in {city_name} for the requested dates."

    results = []
    for stay in stays[:5]:
        accommodation = stay.get("accommodation", {})
        name = accommodation.get("name", "Unknown accommodation")
        room = stay.get("rooms", [{}])[0]
        price = room.get("total_amount", stay.get("total_amount", "N/A"))
        currency = room.get("total_currency", stay.get("total_currency", ""))
        results.append(f"{name} | {price} {currency}")

    return "Found Duffel stay options:\n- " + "\n- ".join(results)


def build_graph_one_tool(tools_list, api_key=None):
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key is required before starting the travel agent.")

    model = ChatOpenAI(model="gpt-4o-mini", api_key=key)
    model_with_tools = model.bind_tools(tools_list)

    def call_node_fn(state):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "action"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_node_fn)
    workflow.add_node("action", ToolNode(tools_list))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    workflow.add_edge("action", "agent")
    return workflow.compile()


tools = [
    tavily_search_tool,
    search_flights_tool,
    get_current_date_tool,
    search_hotels_tool,
]
app_travel_agent = None


def friendly_tool_name(raw_name: str) -> str:
    mapping = {
        "tavily_search_results_json": "Web Search",
        "search_flights_tool": "Flight Search",
        "search_hotels_tool": "Hotel Search",
        "get_current_date_tool": "Date Check",
    }
    return mapping.get(raw_name, raw_name)


def travel_agent_chat(user_input: str, history=None):
    global app_travel_agent

    if app_travel_agent is None:
        yield "Please enter your OpenAI API key to unlock the travel assistant."
        return

    tools_used = []
    response_text = ""
    try:
        stream = app_travel_agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"recursion_limit": 15, "configurable": {"thread_id": str(uuid.uuid4())}},
        )
    except Exception as exc:
        app_travel_agent = None
        yield f"❌ OpenAI authentication failed. Please enter a valid API key. Details: {exc}"
        return

    for chunk in stream:
        if isinstance(chunk, dict):
            for _, node in chunk.items():
                if isinstance(node, dict) and "messages" in node:
                    for msg in node["messages"]:
                        if isinstance(msg, ToolMessage):
                            display_name = friendly_tool_name(msg.name)
                            if display_name not in tools_used:
                                tools_used.append(display_name)
                            response_text += f"\n\n**Tool:** {display_name}\n{msg.content}\n\n---\n\n"
                            yield response_text
                        elif isinstance(msg, AIMessage) and msg.content:
                            response_text += msg.content
                            yield response_text

    if tools_used:
        response_text += f"\n\n**Tools used this session:** {', '.join(tools_used)}"
        yield response_text


st.set_page_config(page_title="LangGraph AI Travel Agent", page_icon="✈️", layout="wide")

st.title("✈️ LangGraph AI Travel Agent 🌍")
st.caption("Plan trips with live web, flight, hotel, and date tools.")

with st.sidebar:
    st.header("Assistant setup")
    configured_key = OPENAI_API_KEY or ""
    entered_key = st.text_input(
        "OpenAI API key",
        value="" if configured_key else st.session_state.get("openai_api_key", ""),
        type="password",
        help="For Streamlit Cloud, add OPENAI_API_KEY under App settings > Secrets.",
    )
    if entered_key:
        st.session_state.openai_api_key = entered_key.strip()
    if configured_key:
        st.success("OpenAI key loaded from app configuration.")
    st.divider()
    st.caption("Optional tools")
    st.caption("TAVILY_API_KEY enables web search. DUFFEL_ACCESS_TOKEN enables flights and hotels.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

active_key = st.session_state.get("openai_api_key") or OPENAI_API_KEY
if not active_key:
    st.info("Add an OpenAI API key in the sidebar or configure OPENAI_API_KEY in Streamlit Secrets to begin.")
    st.stop()

if st.session_state.get("agent_key") != active_key or app_travel_agent is None:
    try:
        app_travel_agent = build_graph_one_tool(tools, api_key=active_key)
        st.session_state.agent_key = active_key
    except Exception as exc:
        st.error(f"Could not initialize the travel assistant: {exc}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about flights, hotels, travel advisories, or an itinerary...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_text = ""
        for response_text in travel_agent_chat(prompt):
            response_placeholder.markdown(response_text)
        if not response_text:
            response_text = "I could not generate a response. Please try again."
            response_placeholder.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
