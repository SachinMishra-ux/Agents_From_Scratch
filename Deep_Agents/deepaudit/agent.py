"""DeepAudit-AI: Core Deep Agent assembly with FastMCP server database tools, subagents, progressive skills, and HITL."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment
ROOT_DIR = Path("/Users/sachinmishra/Desktop/Agents_From_Scratch")
load_dotenv(ROOT_DIR / ".env")

# Remap GOOGLE_API_KEY to GEMINI_API_KEY if needed
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent, SubAgent
from deepagents.backends import FilesystemBackend
import importlib.util

# Load Altman Z and Cash Flow ratios
ratios_path = Path(__file__).parent / "skills" / "forensic-accounting" / "ratios.py"
spec = importlib.util.spec_from_file_location("forensic_ratios", str(ratios_path))
ratios_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ratios_mod)
compute_altman_z_score = ratios_mod.compute_altman_z_score
compute_earnings_quality_ratio = ratios_mod.compute_earnings_quality_ratio

# Load Sloan Accrual & Beneish manipulation engine
beneish_path = Path(__file__).parent / "skills" / "forensic-accounting" / "beneish_sloan.py"
spec_b = importlib.util.spec_from_file_location("beneish_sloan", str(beneish_path))
beneish_mod = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(beneish_mod)
compute_sloan_accrual_ratio = beneish_mod.compute_sloan_accrual_ratio
compute_beneish_forensic_indices = beneish_mod.compute_beneish_forensic_indices

MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", 8001))


# ---------------------------------------------------------
# FastMCP Database Tool Loader via langchain_mcp_adapters
# ---------------------------------------------------------

async def get_mcp_database_tools_async(
    host: str = MCP_HOST,
    port: int = MCP_PORT
) -> List[BaseTool]:
    """Connect to the FastMCP Database Server over SSE (Terminal 1) or fallback to stdio."""
    sse_url = f"http://{host}:{port}/sse"
    try:
        client = MultiServerMCPClient({
            "financial_database": {
                "url": sse_url,
                "transport": "sse"
            }
        })
        tools = await client.get_tools()
        print(f"[MCP CLIENT] Connected to FastMCP SSE Server at {sse_url}. Loaded: {[t.name for t in tools]}", flush=True)
        return tools
    except Exception as e:
        print(f"[MCP CLIENT] FastMCP SSE server not reachable at {sse_url} ({e}).", flush=True)
        print("[MCP CLIENT] Spawning FastMCP server via stdio transport fallback...", flush=True)
        mcp_path = str(Path(__file__).parent / "mcp_server.py")
        client = MultiServerMCPClient({
            "financial_database": {
                "command": sys.executable,
                "args": [mcp_path, "--stdio"],
                "transport": "stdio"
            }
        })
        tools = await client.get_tools()
        print(f"[MCP CLIENT] Connected via stdio MCP. Loaded: {[t.name for t in tools]}", flush=True)
        return tools


def get_mcp_database_tools_sync(
    host: str = MCP_HOST,
    port: int = MCP_PORT
) -> List[BaseTool]:
    """Synchronous wrapper to retrieve MCP database tools."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(get_mcp_database_tools_async(host, port))
        else:
            return loop.run_until_complete(get_mcp_database_tools_async(host, port))
    except RuntimeError:
        return asyncio.run(get_mcp_database_tools_async(host, port))


# ---------------------------------------------------------
# Specialized Quantitative & Forensic Tools
# ---------------------------------------------------------

@tool
def calculate_altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    revenue: float,
    total_assets: float,
    total_liabilities: float
) -> str:
    """Calculate the Altman Z-Score solvency metric and distress zone classification."""
    res = compute_altman_z_score(
        working_capital,
        retained_earnings,
        ebit,
        market_cap,
        revenue,
        total_assets,
        total_liabilities
    )
    return json.dumps(res, indent=2)


@tool
def calculate_cfo_quality_ratio(operating_cash_flow: float, net_income: float) -> str:
    """Calculate the Operating Cash Flow to Net Income conversion ratio to detect accrual anomalies."""
    res = compute_earnings_quality_ratio(operating_cash_flow, net_income)
    return json.dumps(res, indent=2)


@tool
def calculate_sloan_accrual_anomaly(
    net_income: float,
    operating_cash_flow: float,
    total_assets: float,
    prior_total_assets: Optional[float] = None
) -> str:
    """Calculate Richard Sloan's Accrual Anomaly Ratio to evaluate if net income is supported by cash.
    
    Formula: (Net Income - Operating Cash Flow) / Average Total Assets.
    Values < -3% signify High Cash Quality; values > +10% indicate high artificial accrual distortion.
    """
    res = compute_sloan_accrual_ratio(
        net_income=net_income,
        operating_cash_flow=operating_cash_flow,
        total_assets=total_assets,
        prior_total_assets=prior_total_assets
    )
    return json.dumps(res, indent=2)


@tool
def calculate_beneish_forensic_indices(
    current_year_revenue: float,
    prior_year_revenue: float,
    current_year_net_income: float,
    current_year_cfo: float,
    current_year_assets: float,
    prior_year_assets: float,
    current_year_liabilities: float,
    prior_year_liabilities: float
) -> str:
    """Calculate Beneish forensic accounting indices (Sales Growth Index, Total Accruals to Assets, Asset Quality Index)."""
    res = compute_beneish_forensic_indices(
        current_year_revenue=current_year_revenue,
        prior_year_revenue=prior_year_revenue,
        current_year_net_income=current_year_net_income,
        current_year_cfo=current_year_cfo,
        current_year_assets=current_year_assets,
        prior_year_assets=prior_year_assets,
        current_year_liabilities=current_year_liabilities,
        prior_year_liabilities=prior_year_liabilities
    )
    return json.dumps(res, indent=2)


@tool
def web_search(query: str) -> str:
    """Search Google/Web for recent regulatory fines, SEC enforcement actions, or litigation."""
    serpapi_key = os.environ.get("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            from serpapi import GoogleSearch
            search = GoogleSearch({"q": query, "api_key": serpapi_key})
            results = search.get_dict().get("organic_results", [])
            if not results:
                return "No relevant search results found."
            summary = []
            for r in results[:3]:
                summary.append(f"Title: {r.get('title')}\nLink: {r.get('link')}\nSnippet: {r.get('snippet')}\n")
            return "\n".join(summary)
        except Exception as e:
            return f"Search query failed: {e}"
    return "Search completed: No active enforcement actions found in baseline records."


# ---------------------------------------------------------
# Subagent Specifications (Context Offloading with MCP)
# ---------------------------------------------------------

def create_audit_agent(
    model_name: str = "google_genai:gemini-2.5-flash",
    checkpointer: Optional[Any] = None,
    interrupt_on_write: bool = True,
    mcp_tools: Optional[List[BaseTool]] = None
):
    """Factory function to build a compiled DeepAgent graph powered by FastMCP database tools."""
    backend = FilesystemBackend(root_dir=str(ROOT_DIR), virtual_mode=True)
    
    if checkpointer is None:
        checkpointer = MemorySaver()

    interrupt_cfg = {"write_file": True} if interrupt_on_write else None

    # Load MCP database tools if not explicitly supplied
    if mcp_tools is None:
        mcp_tools = get_mcp_database_tools_sync()

    # Extract individual MCP tools
    mcp_query_db = next((t for t in mcp_tools if t.name == "query_financial_db"), None)
    mcp_company = next((t for t in mcp_tools if t.name == "get_company_profile"), None)
    mcp_financials = next((t for t in mcp_tools if t.name == "get_historical_financials"), None)

    # Subagent 1 Tools: MCP Database Queries + Solvency Ratios
    quant_tools = [t for t in [mcp_query_db, mcp_financials, calculate_altman_z_score, calculate_cfo_quality_ratio] if t]

    # Subagent 2 Tools: MCP Database Queries + Accruals / Beneish Forensics
    accrual_tools = [t for t in [mcp_query_db, mcp_financials, calculate_sloan_accrual_anomaly, calculate_beneish_forensic_indices] if t]

    # Lead Orchestrator Tools: MCP Database Tools + Web Search
    orchestrator_tools = [t for t in [mcp_query_db, mcp_company, mcp_financials, web_search] if t]

    # Subagent 1: Quant Solvency Auditor
    quant_subagent: SubAgent = {
        "name": "quant_auditor",
        "description": "Specialized quantitative analyst that runs Altman Z solvency math, checks regulatory capital ratios, and queries SQLite balance sheets via the FastMCP server.",
        "system_prompt": (
            "You are an elite quantitative forensic accountant. "
            "Your task is to execute calculations using `calculate_altman_z_score`, `calculate_cfo_quality_ratio`, "
            "and query historical tables using FastMCP tool `query_financial_db`. "
            "DO NOT dump raw database tables back to the parent agent. "
            "Always summarize your findings in a concise 3-4 bullet point executive report highlighting solvency health."
        ),
        "tools": quant_tools,
        "model": model_name
    }

    # Subagent 2: Accrual & Earnings Manipulation Forensic Auditor
    accrual_subagent: SubAgent = {
        "name": "accrual_forensic_auditor",
        "description": "Forensic accounting specialist that investigates earnings manipulation, Sloan Accrual Anomaly, Beneish M-Score indices, and multi-year cash-flow decoupling via FastMCP database queries.",
        "system_prompt": (
            "You are a specialized institutional Forensic Accrual & Earnings Manipulation Auditor. "
            "Your role is to detect accounting gimmickry, aggressive accrual recognition, and paper earnings distortion. "
            "Execute your investigation by:\n"
            "1. Querying multi-year statements via FastMCP tool `query_financial_db` (look closely at Net Income vs. Operating Cash Flow trends across 2021, 2022, and 2023).\n"
            "2. Calculating the Sloan Accrual Ratio using `calculate_sloan_accrual_anomaly`.\n"
            "3. Computing Beneish manipulation indices using `calculate_beneish_forensic_indices`.\n"
            "4. Returning a clear 3-4 bullet forensic evaluation summarizing: Cash Flow conversion quality, Sloan Accrual classification, and whether earnings manipulation risk is Low or High."
        ),
        "tools": accrual_tools,
        "model": model_name
    }

    # Configure Human-in-the-loop (HITL) Middleware
    # HumanInTheLoopMiddleware pauses execution before target tool calls (e.g. write_file)
    # allowing human approval, parameter editing, rejection, or response.
    custom_middleware = []
    if interrupt_on_write:
        hitl_middleware = HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            }
        )
        custom_middleware.append(hitl_middleware)

    agent = create_deep_agent(
        model=model_name,
        tools=orchestrator_tools,
        subagents=[quant_subagent, accrual_subagent],
        skills=["Deep_Agents/deepaudit/skills/"],
        backend=backend,
        checkpointer=checkpointer,
        middleware=custom_middleware if custom_middleware else None,
        system_prompt=(
            "You are DeepAudit-AI, an autonomous Lead Forensic Due Diligence Auditor. "
            "Your mission is to audit corporate annual reports and SEC filings using connected FastMCP database tools. "
            "\n"
            "Execution Principles:\n"
            "1. ALWAYS start by creating a detailed checklist using `write_todos`.\n"
            "2. DELEGATE specialized analysis to your 2 sub-agents via `task()` to keep your context window clean:\n"
            "   - Delegate solvency & Altman Z calculations to `quant_auditor`.\n"
            "   - Delegate Sloan accrual divergence, Beneish indices, and earnings manipulation screening to `accrual_forensic_auditor`.\n"
            "3. Use FastMCP tools `query_financial_db` and `get_company_profile` to inspect corporate metadata.\n"
            "4. Use `web_search` to verify regulatory compliance or SEC enforcement actions.\n"
            "5. Follow the progressive skills in `skills/` for analytical benchmarks.\n"
            "6. SYNTHESIZE all sub-agent findings into an institutional Investment Committee Memorandum and write it to `reports/JPM_2023_Due_Diligence_Memo.md` using `write_file`.\n"
            "   (Note: writing this file will trigger a Human-in-the-Loop review checkpoint for approval via HumanInTheLoopMiddleware)."
        )
    )
    return agent
