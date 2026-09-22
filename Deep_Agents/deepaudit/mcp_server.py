"""Model Context Protocol (MCP) Server for Financial Auditing & Database Queries.

Run in Terminal 1:
    python Deep_Agents/deepaudit/mcp_server.py

This starts the FastMCP server over SSE on http://127.0.0.1:8001/sse.
"""

import json
import os
import sys
import sqlite3
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import uvicorn

DB_PATH = Path(__file__).parent / "data" / "financial_audit.db"
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", 8001))

# Initialize FastMCP Server
mcp = FastMCP("financial_database", host=MCP_HOST, port=MCP_PORT)


def _execute_query(query: str, params: tuple = ()):
    """Execute a query against financial_audit.db and return list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        conn.close()
        return result
    except Exception as e:
        conn.close()
        return {"error": str(e)}


@mcp.tool()
def query_financial_db(sql_query: str) -> str:
    """Execute a read-only SQL query against the financial audit database.
    
    Available tables:
    - company_profiles (ticker, company_name, sector, reporting_currency, latest_fiscal_year, market_cap)
    - financial_statements (ticker, fiscal_year, total_revenue, net_income, operating_cash_flow, total_assets, total_liabilities, stockholders_equity, retained_earnings, ebit, cet1_ratio)
    - audit_flags (id, ticker, fiscal_year, flag_type, severity, description)
    """
    # Safety guard: only allow SELECT and WITH queries
    cleaned = sql_query.strip().upper()
    if not (cleaned.startswith("SELECT") or cleaned.startswith("WITH")):
        return json.dumps({"error": "Only read-only SELECT queries are permitted on the audit database."})

    res = _execute_query(sql_query)
    return json.dumps(res, indent=2)


@mcp.tool()
def get_company_profile(ticker: str) -> str:
    """Retrieve verified corporate metadata and sector classification for a given ticker."""
    res = _execute_query("SELECT * FROM company_profiles WHERE ticker = ?", (ticker.upper(),))
    return json.dumps(res, indent=2)


@mcp.tool()
def get_historical_financials(ticker: str) -> str:
    """Retrieve 3-year historical statements (revenue, net income, cash flows, balance sheet, capital ratios) for a company."""
    res = _execute_query(
        "SELECT * FROM financial_statements WHERE ticker = ? ORDER BY fiscal_year ASC",
        (ticker.upper(),)
    )
    return json.dumps(res, indent=2)


if __name__ == "__main__":
    # Check for CLI flags
    if "--stdio" in sys.argv:
        print("[FastMCP] Running in stdio transport mode...", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        # Check custom port if passed e.g. --port 8001
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                MCP_PORT = int(sys.argv[i + 1])

        print("\n" + "=" * 60)
        print("🚀 DeepAudit-AI FastMCP Database Server Active")
        print(f"🌐 Transport: SSE (Server-Sent Events)")
        print(f"📡 Endpoint:  http://{MCP_HOST}:{MCP_PORT}/sse")
        print("📦 Tools Exposed:")
        print("   • query_financial_db       (Read-Only SQL Executor)")
        print("   • get_company_profile      (Corporate Metadata)")
        print("   • get_historical_financials (3-Year Statements)")
        print("=" * 60 + "\n")

        app = mcp.sse_app()
        uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
