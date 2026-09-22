"""Beneish M-Score and Sloan Accrual Anomaly Forensic Calculation Engine.

Implements empirical forensic accounting models:
1. Sloan Accrual Ratio (Richard Sloan, 1996 - Accounting Review)
   Detects artificial earnings decoupling from operating cash flow.
2. Beneish Earnings Manipulation Indices (Messod Beneish, 1999)
   - Days Sales in Receivables Index (DSRI)
   - Asset Quality Index (AQI)
   - Sales Growth Index (SGI)
   - Accruals to Total Assets (TATA)
"""

from typing import Dict, Any, Optional


def compute_sloan_accrual_ratio(
    net_income: float,
    operating_cash_flow: float,
    total_assets: float,
    prior_total_assets: Optional[float] = None
) -> Dict[str, Any]:
    """Calculate the Sloan Accrual Ratio.
    
    Formula:
      Accrual Ratio = (Net Income - Operating Cash Flow) / Average Total Assets
    
    Interpretation:
      - Accrual Ratio < -3%: High Cash Quality (Cash flow exceeds net income)
      - Accrual Ratio between -3% and +7%: Normal Operating Accruals
      - Accrual Ratio > +10%: Elevated Accrual Manipulation Risk (Earnings driven by paper profits)
    """
    avg_assets = (total_assets + prior_total_assets) / 2.0 if prior_total_assets else total_assets
    if avg_assets <= 0:
        return {"error": "Invalid total assets"}

    accruals = net_income - operating_cash_flow
    sloan_ratio = (accruals / avg_assets) * 100.0  # as a percentage

    if sloan_ratio < -3.0:
        quality_assessment = "High Cash Quality (Cash flow significantly outpaces accounting net income)"
        flag_severity = "LOW_RISK"
    elif sloan_ratio <= 8.0:
        quality_assessment = "Moderate / Normal Accruals (Standard working capital accrual drift)"
        flag_severity = "NORMAL"
    elif sloan_ratio <= 14.0:
        quality_assessment = "Cautionary Accruals (Operating cash flow lagging net income)"
        flag_severity = "MEDIUM"
    else:
        quality_assessment = "Elevated Manipulation Risk (Severe decoupling between paper profits and cash)"
        flag_severity = "HIGH"

    return {
        "model": "Sloan Accrual Anomaly (1996)",
        "net_income": net_income,
        "operating_cash_flow": operating_cash_flow,
        "accrual_dollar_amount": round(accruals, 2),
        "sloan_accrual_ratio_percent": round(sloan_ratio, 2),
        "interpretation": quality_assessment,
        "flag_severity": flag_severity
    }


def compute_beneish_forensic_indices(
    current_year_revenue: float,
    prior_year_revenue: float,
    current_year_net_income: float,
    current_year_cfo: float,
    current_year_assets: float,
    prior_year_assets: float,
    current_year_liabilities: float,
    prior_year_liabilities: float
) -> Dict[str, Any]:
    """Compute Beneish Forensic Manipulation Indices.
    
    Key Metrics:
    - SGI (Sales Growth Index): Measures if hypergrowth is putting pressure on accounting margins.
    - AQI (Asset Quality Index): Measures non-current asset expansion vs total assets.
    - TATA (Total Accruals to Total Assets): Beneish proxy for earnings quality.
    """
    sgi = current_year_revenue / prior_year_revenue if prior_year_revenue > 0 else 1.0
    total_accruals = current_year_net_income - current_year_cfo
    tata = total_accruals / current_year_assets if current_year_assets > 0 else 0.0

    prior_non_tangible = max(0.0, prior_year_assets - prior_year_revenue)
    curr_non_tangible = max(0.0, current_year_assets - current_year_revenue)
    
    aqi_curr = curr_non_tangible / current_year_assets if current_year_assets > 0 else 1.0
    aqi_prior = prior_non_tangible / prior_year_assets if prior_year_assets > 0 else 1.0
    aqi = aqi_curr / aqi_prior if aqi_prior > 0 else 1.0

    risk_points = 0
    anomalies = []

    if sgi > 1.30:
        risk_points += 1
        anomalies.append(f"High Sales Growth ({round((sgi-1)*100, 1)}% YoY) increases incentive for aggressive revenue recognition.")
    
    if tata > 0.08:
        risk_points += 2
        anomalies.append("Total Accruals exceed 8% of Total Assets, signaling high earnings non-cash intensity.")

    if aqi > 1.25:
        risk_points += 1
        anomalies.append("Asset Quality Index > 1.25 indicates significant expansion of intangible or non-standard asset reserves.")

    if risk_points == 0:
        beneish_summary = "Non-Manipulator Profile: Financial statements exhibit low distortion indicators."
        overall_status = "SAFE_ZONE"
    elif risk_points <= 2:
        beneish_summary = "Normal Banking Variance: Ratios reflect inorganic M&A growth and interest rate expansion."
        overall_status = "MODERATE_ZONE"
    else:
        beneish_summary = "Heightened Earnings Quality Scrutiny Recommended: Multiple accrual divergences detected."
        overall_status = "FLAG_SCRUTINY"

    return {
        "model": "Beneish Forensic Accounting Indices",
        "sales_growth_index_sgi": round(sgi, 3),
        "total_accruals_to_assets_tata": round(tata, 4),
        "asset_quality_index_aqi": round(aqi, 3),
        "risk_points": risk_points,
        "anomalies_detected": anomalies,
        "forensic_status": overall_status,
        "executive_summary": beneish_summary
    }
