---
name: forensic-accounting
description: Forensic accounting formulas and analytical rules for calculating Altman Z-Score, Beneish M-Score, and earnings quality divergence metrics.
allowed-tools: calculate_altman_z_score query_financial_db read_file
---

# Forensic Accounting & Solvency Analysis Skill

This skill provides quantitative methodologies and threshold guidelines for evaluating corporate solvency and forensic accounting irregularities.

## When to Use
- Evaluating corporate bankruptcy risk or balance sheet distress.
- Checking whether financial statements show symptoms of aggressive earnings manipulation.
- Benchmarking multi-year liquidity, leverage, and capital adequacy.

## 1. Altman Z-Score Solvency Metric

Formula:
$$Z = 1.2 \cdot X_1 + 1.4 \cdot X_2 + 3.3 \cdot X_3 + 0.6 \cdot X_4 + 0.999 \cdot X_5$$

Where:
- $X_1 = \text{Working Capital} / \text{Total Assets}$ (Measures short-term liquidity)
- $X_2 = \text{Retained Earnings} / \text{Total Assets}$ (Measures cumulative profitability)
- $X_3 = \text{EBIT} / \text{Total Assets}$ (Measures asset productivity)
- $X_4 = \text{Market Cap} / \text{Total Liabilities}$ (Measures equity cushion)
- $X_5 = \text{Total Revenue} / \text{Total Assets}$ (Measures asset turnover)

### Interpretation Zones:
- **$Z > 2.99$**: **Safe Zone** (Low probability of insolvency within 2 years).
- **$1.81 \le Z \le 2.99$**: **Grey Zone** (Moderate risk; heightened scrutiny required).
- **$Z < 1.81$**: **Distress Zone** (High probability of default/restructuring).

## 2. Cash Flow Quality Ratio

$$\text{CFO Ratio} = \frac{\text{Cash Flow from Operating Activities}}{\text{Net Income}}$$

- **Benchmark $> 1.0$**: Healthy earnings backed by cash generation.
- **Benchmark $< 0.8$**: Red flag; net income may be inflated by non-cash accrued items or deferred costs.
- **Negative Ratio (Negative CFO with Positive Net Income)**: Severe red flag indicating poor revenue quality.

## 3. Financial Institution Capital Adequacy (CET1)
For banks and financial holdings (like J.P. Morgan Chase):
- **Common Equity Tier 1 (CET1) Ratio**: Regulatory benchmark $\ge 12.0\%$. JPM reported $15.0\%$ in 2023, representing robust capitalization above regulatory minimums.
- **Liquidity Coverage Ratio (LCR)**: Benchmark $\ge 100\%$.

## 4. Sloan Accrual Anomaly & Beneish Forensic Manipulation

### Sloan Accrual Ratio (1996)
$$\text{Accrual Ratio} = \frac{\text{Net Income} - \text{Operating Cash Flow}}{\text{Average Total Assets}} \times 100\%$$

- **$< -3.0\%$**: **High Cash Quality** (Operating cash flow comfortably outpaces net income).
- **$-3.0\%$ to $+8.0\%$**: **Normal Accruals** (Standard seasonal working capital drift).
- **$> +10.0\%$**: **Elevated Accrual Risk** (Net income driven by uncollected paper profits rather than cash).

### Beneish Forensic Indices
- **Sales Growth Index (SGI)**: Rapid sales acceleration ($> 1.30$) creates incentive for aggressive revenue recognition.
- **Asset Quality Index (AQI)**: Ratio $> 1.25$ indicates expansion of non-current intangible/illiquid assets, signaling deferred expense capitalization.
- **Total Accruals to Total Assets (TATA)**: Measures proportion of total assets constituted by non-cash net income.
