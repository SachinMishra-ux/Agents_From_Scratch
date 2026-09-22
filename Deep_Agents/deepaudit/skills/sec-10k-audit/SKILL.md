---
name: sec-10k-audit
description: Comprehensive Standard Operating Procedure for auditing SEC 10-K filings, navigating Item 8 financial statements, footnotes, and Item 7 MD&A disclosures.
allowed-tools: read_file grep query_financial_db calculate_altman_z_score
---

# SEC 10-K Annual Report Audit SOP

This skill guides the agent in conducting institutional-grade financial audits of SEC Form 10-K annual reports.

## When to Use
- When given a corporate annual filing or 10-K document (e.g., JPMorgan Chase, EY, or tech firms).
- When investigating revenue recognition patterns, asset quality, off-balance-sheet liabilities, and capital adequacy.
- When tasked with synthesizing multi-year performance across income, balance sheet, and cash flow statements.

## Audit Workflow Steps

### Step 1: Document Structure Identification
Locate the core sections of the 10-K filing:
- **Item 7: Management's Discussion and Analysis (MD&A)**: Read for executive outlook, net interest margin commentary, and known risks.
- **Item 8: Financial Statements and Supplementary Data**:
  - Consolidated Statements of Income (Revenue, Net Income, Non-interest expense)
  - Consolidated Balance Sheets (Assets, Loans, Deposits, Long-term debt, Equity)
  - Consolidated Statements of Cash Flows (Operating, Investing, Financing cash flows)
- **Item 8 Notes**: Footnotes on credit reserves, litigation contingencies, and fair value measurements.

### Step 2: Extracting Consolidated Metrics
Extract numbers for the 3 most recent fiscal years:
1. **Total Net Revenue** (or Net Interest Income + Non-interest Revenue for banks)
2. **Net Income / Net Earnings**
3. **Cash Flows from Operating Activities**
4. **Total Assets & Total Liabilities**
5. **Stockholders' Equity & Retained Earnings**

### Step 3: Divergence & Anomaly Screening
Compare Operating Cash Flow against Net Income:
- If Reported Net Income is steadily rising while Operating Cash Flow is declining, flag an **Accounting Quality Warning** (potential premature revenue recognition or uncollected receivables).
- Check Provision for Credit Losses against Net Charge-Offs. If provisions drop dramatically in a deteriorating macroeconomic environment, flag **Reserve Understatement Risk**.

### Step 4: Subagent Context Offloading
Do not retain full 100+ page transcripts in your conversational state. Always delegate raw tabular extraction to the `quant_auditor` subagent, and instruct it to return a concise markdown table with key ratios.
