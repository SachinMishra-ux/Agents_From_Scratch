---
name: ic-memo-generator
description: Guidelines and structure for synthesizing complex audit findings into an institutional Investment Committee Due Diligence Memorandum.
allowed-tools: write_file read_file
---

# Investment Committee (IC) Due Diligence Memorandum SOP

This skill guides the final synthesis of the audit into a boardroom-ready Investment Committee Memorandum.

## When to Use
- When all subagent analyses (Quant audit, SEC disclosure check, litigation scout) are complete.
- When drafting the final deliverable to be saved to `reports/`.
- Prior to triggering the Human-in-the-Loop review interrupt.

## Document Structure Template

```markdown
# EXECUTIVE DUE DILIGENCE MEMORANDUM

**Target Entity**: [Company Name & Ticker]  
**Audit Period**: [e.g. FY2021 – FY2023]  
**Lead Auditor**: DeepAudit Autonomous Agent Harness  
**Date**: [Current Date]  
**Audit Recommendation**: [APPROVED / CAUTION / REJECT]  

---

## 1. Executive Summary & Verdict
- High-level verdict summarizing solvency, earnings quality, and capital adequacy.
- Top 3 critical takeaways.

## 2. Multi-Year Financial Performance Matrix
A clean markdown table comparing:
- Total Net Revenue ($ Billions)
- Net Income ($ Billions)
- Cash Flow from Operations ($ Billions)
- Return on Tangible Common Equity (ROTCE) / Return on Equity (ROE)

## 3. Quantitative Solvency & Forensic Metrics
- **Altman Z-Score**: Value, Risk Zone (Safe / Grey / Distress), and Liquidity interpretation.
- **Cash Flow Conversion**: Operating Cash Flow / Net Income Ratio and divergence analysis.
- **Capital Adequacy (For Financial Institutions)**: Basel III CET1 Ratio vs. Regulatory Minimums.

## 4. Key Accounting & Disclosure Observations (Item 7 & 8)
- Credit Loss Provisioning trends.
- Notable acquisitions or balance sheet changes (e.g. First Republic transaction for JPM).
- Legal & Regulatory Contingencies.

## 5. Investment Committee Recommendation & Sign-Off
- Concluding sign-off paragraph.
- Sign-off placeholder for Human Lead Auditor.
```
