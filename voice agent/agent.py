"""
===============================================================================
🧠 AGENT MODULE — The LangChain AI Brain with Tools & Memory
===============================================================================
This module creates our intelligent conversational voice agent.
It is the "filling" of our Sandwich Architecture:

  1. Takes written text from Speech-to-Text (AssemblyAI).
  2. Uses SenseNova Large Language Model (LLM) to reason.
  3. Calls real Python functions (Tools) to look up menus and manage orders.
  4. Remembers conversation history using LangGraph's InMemorySaver checkpointer.
  5. Streams tokens word-by-word to the Text-to-Speech engine.
===============================================================================
"""

import os
from typing import AsyncIterator, List, Dict, Any
from dotenv import load_dotenv

# LangChain & LangGraph imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables from .env
load_dotenv(override=True)


# -----------------------------------------------------------------------------
# 1. LLM FACTORY (SenseNova OpenAI-Compatible Endpoint)
# -----------------------------------------------------------------------------
def get_llm() -> BaseChatModel:
    """
    Initialize and return the SenseNova ChatModel.
    SenseNova exposes a standard OpenAI-compatible API endpoint.
    
    Students: We use low temperature (0.1) so the agent is precise,
    factual, and doesn't hallucinate fake menu items or prices!
    """
    sensenova_api_key = os.getenv("SENSENOVA_API_KEY")
    if not sensenova_api_key:
        raise ValueError("SENSENOVA_API_KEY is not set in environment.")

    model = os.getenv("SENSENOVA_MODEL", "sensenova-6.8-flash-lite")
    return ChatOpenAI(
        base_url="https://token.sensenova.ai/v1",
        api_key=sensenova_api_key,
        model=model,
        temperature=0.1,
    )


# -----------------------------------------------------------------------------
# 2. APPLICATION STATE & DOMAIN TOOLS (Sandwich & Cafe Ordering)
# -----------------------------------------------------------------------------
# Simulated cart database (In-memory list of active order items)
CURRENT_ORDER: List[Dict[str, Any]] = []

# Price menu for sandwiches and beverages
MENU_ITEMS = {
    "classic club": 8.99,
    "grilled chicken panini": 9.50,
    "veggie avocado wrap": 7.99,
    "smoked turkey & brie": 10.25,
    "tuna melt": 8.50,
    "espresso": 3.00,
    "cappuccino": 4.50,
    "iced matcha latte": 5.00,
    "chips": 2.00,
    "chocolate chip cookie": 2.50,
}

@tool
def get_sandwich_menu() -> str:
    """Get the available menu items and their prices."""
    items = [f"{item.title()} (${price:.2f})" for item, price in MENU_ITEMS.items()]
    return "Our menu includes: " + ", ".join(items) + "."

@tool
def add_to_order(item: str, quantity: int = 1, special_instructions: str = "") -> str:
    """Add an item to the customer's sandwich/drink order.
    
    Args:
        item: The name of the food or beverage item.
        quantity: Quantity of the item.
        special_instructions: Any custom modifications like no mayo, extra cheese, oat milk.
    """
    cleaned_item = item.lower().strip()
    match = None
    for menu_key in MENU_ITEMS:
        if menu_key in cleaned_item or cleaned_item in menu_key:
            match = menu_key
            break

    item_name = match if match else cleaned_item
    price = MENU_ITEMS.get(item_name, 8.00)
    
    order_entry = {
        "item": item_name.title(),
        "quantity": max(1, quantity),
        "price": price,
        "instructions": special_instructions or "None"
    }
    CURRENT_ORDER.append(order_entry)
    
    return f"Added {quantity} {item_name.title()} to the order. Current item count: {len(CURRENT_ORDER)}."

@tool
def view_current_order() -> str:
    """View items currently in the customer's order and calculate the total cost."""
    if not CURRENT_ORDER:
        return "The order is currently empty."
    
    summary = []
    total = 0.0
    for item in CURRENT_ORDER:
        subtotal = item["price"] * item["quantity"]
        total += subtotal
        summary.append(f"{item['quantity']}x {item['item']} (${subtotal:.2f})")
        
    return f"Current order: {', '.join(summary)}. Total is ${total:.2f}."

@tool
def confirm_and_place_order(customer_name: str, order_type: str = "pickup") -> str:
    """Finalize the order and send it to the kitchen.
    
    Args:
        customer_name: The name of the customer for the order.
        order_type: Either 'pickup' or 'dine-in'.
    """
    if not CURRENT_ORDER:
        return "Cannot confirm an empty order. Please add items first."
    
    total = sum(i["price"] * i["quantity"] for i in CURRENT_ORDER)
    count = sum(i["quantity"] for i in CURRENT_ORDER)
    CURRENT_ORDER.clear()
    
    return f"Order confirmed for {customer_name}! {count} items for {order_type}. Total is ${total:.2f}. Preparing in kitchen now."


TOOLS = [
    get_sandwich_menu,
    add_to_order,
    view_current_order,
    confirm_and_place_order,
]

# -----------------------------------------------------------------------------
# 3. VOICE-SPECIFIC SYSTEM PROMPT
# -----------------------------------------------------------------------------
# CRITICAL FOR STUDENTS TO UNDERSTAND:
# Voice models cannot read asterisks (*), markdown tables, bullet points, or emojis!
# If you leave emojis, the TTS engine might literally say "grinning face emoji".
# So our system prompt enforces natural speech formatting.
SYSTEM_PROMPT = """You are Sam, the friendly and efficient voice assistant for 'The Daily Melt Deli & Cafe', a gourmet sandwich and artisan coffee shop.
Your goal is to converse naturally with guests, take customized sandwich and beverage orders, answer menu questions, and confirm orders.

VOICE CONVERSATION RULES (CRITICAL):
1. Greet guests warmly and keep answers conversational, natural, and concise (1 to 2 short sentences per turn).
2. DO NOT use markdown formatting (no asterisks, bolding, bullet points, headers, or code blocks).
3. DO NOT use emojis, icons, or special characters.
4. Pronounce prices naturally as words (e.g., say 'eight dollars and ninety-nine cents' or 'eight ninety-nine').
5. Your text is spoken directly by a text-to-speech voice engine, so phrase everything as pleasant spoken English.
"""


# -----------------------------------------------------------------------------
# 4. AGENT FACTORY (LangChain create_agent)
# -----------------------------------------------------------------------------
def build_voice_agent():
    """
    Create and return the LangChain agent.
    - Model: SenseNova LLM
    - Tools: Sandwich shop tools
    - Checkpointer: InMemorySaver (provides multi-turn conversation memory)
    """
    llm = get_llm()
    checkpointer = InMemorySaver()
    
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent

# Instantiate global agent instance
voice_agent = build_voice_agent()


# -----------------------------------------------------------------------------
# 5. STREAMING AGENT PIPELINE TRANSFORMER
# -----------------------------------------------------------------------------
async def agent_stream(
    event_stream: AsyncIterator[Any],
    thread_id: str = "voice-session-default",
) -> AsyncIterator[Any]:
    """
    Transform stream: Voice Events -> Voice Events (with streamed Agent Responses).
    
    Lifecycle:
    1. Passes through all upstream events.
    2. When 'stt_output' arrives (user finished sentence), invokes the LangChain Agent.
    3. Streams response tokens as 'agent_chunk' events for immediate TTS synthesis.
    4. Yields 'agent_complete' when the turn finishes.
    """
    from events import AgentChunkEvent, AgentCompleteEvent, StatusEvent

    async for event in event_stream:
        # Pass through incoming events
        yield event

        # Check if user finished a complete sentence
        if getattr(event, "type", None) == "stt_output" and getattr(event, "transcript", "").strip():
            transcript = event.transcript.strip()
            yield StatusEvent.create(status="thinking", message=f"Thinking: {transcript}")

            full_text_parts = []
            
            try:
                # Thread ID preserves multi-turn memory
                config = {"configurable": {"thread_id": thread_id}}
                input_msg = {"messages": [HumanMessage(content=transcript)]}

                # Stream response tokens incrementally
                async for chunk, metadata in voice_agent.astream(
                    input_msg,
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, AIMessage) and chunk.content:
                        text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        if text:
                            full_text_parts.append(text)
                            yield AgentChunkEvent.create(text=text)

                complete_response = "".join(full_text_parts).strip()
                if complete_response:
                    yield AgentCompleteEvent.create(response=complete_response)
                
            except Exception as e:
                err_msg = f"Sorry, I encountered an error: {str(e)}"
                yield AgentChunkEvent.create(text=err_msg)
                yield AgentCompleteEvent.create(response=err_msg)
