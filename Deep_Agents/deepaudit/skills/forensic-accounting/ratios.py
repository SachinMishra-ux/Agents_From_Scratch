"""Quantitative ratio calculation engine for forensic accounting."""

from typing import Dict, Any


def compute_altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    revenue: float,
    total_assets: float,
    total_liabilities: float
) -> Dict[str, Any]:
    """Calculate Altman Z-Score with component breakdown and zone classification.
    
    Returns:
        dict containing z_score, zone classification, and individual X1-X5 components.
    """
    if total_assets <= 0 or total_liabilities <= 0:
        return {"error": "Total assets and total liabilities must be positive numbers."}

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = revenue / total_assets

    z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (0.999 * x5)

    if z_score > 2.99:
        zone = "Safe Zone"
        interpretation = "Negligible risk of insolvency within a 2-year forecast horizon."
    elif z_score >= 1.81:
        zone = "Grey Zone"
        interpretation = "Moderate risk; company shows borderline liquidity or leverage constraints."
    else:
        zone = "Distress Zone"
        interpretation = "High risk of default or restructuring within 24 months."

    return {
        "z_score": round(z_score, 2),
        "zone": zone,
        "interpretation": interpretation,
        "components": {
            "x1_liquidity": round(x1, 4),
            "x2_retained_earnings_ratio": round(x2, 4),
            "x3_asset_productivity": round(x3, 4),
            "x4_leverage_ratio": round(x4, 4),
            "x5_asset_turnover": round(x5, 4)
        }
    }


def compute_earnings_quality_ratio(operating_cash_flow: float, net_income: float) -> Dict[str, Any]:
    """Evaluate cash earnings conversion quality."""
    if net_income == 0:
        return {"ratio": 0.0, "status": "Neutral (Zero Net Income)"}

    ratio = operating_cash_flow / net_income

    if ratio > 1.0:
        status = "High Quality (Cash flow exceeds reported GAAP net earnings)"
        risk = "LOW"
    elif ratio >= 0.75:
        status = "Acceptable (Reasonable accrual variance)"
        risk = "MODERATE"
    elif ratio > 0:
        status = "Poor Quality (Earnings largely driven by non-cash accruals)"
        risk = "HIGH"
    else:
        status = "Critical Divergence (Positive Net Income alongside Negative Cash Flow)"
        risk = "CRITICAL"

    return {
        "cfo_to_ni_ratio": round(ratio, 2),
        "status": status,
        "risk_level": risk
    }
