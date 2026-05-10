import datetime as dt
from typing import Dict, List, Optional

import streamlit as st
from engines.probability import bias_probabilities


def _sign(label: str) -> int:
    if label == "Bullish":
        return 1
    if label == "Bearish":
        return -1
    return 0


def _normalize_conf(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def blend_bias_components(components: List[Dict[str, object]]) -> Dict[str, object]:
    weighted = 0.0
    total = 0.0
    for row in components:
        label = str(row.get("bias", "Neutral"))
        sign = _sign(label)
        conf = _normalize_conf(row.get("confidence", 0.0))
        if sign == 0 or conf <= 0:
            continue
        weighted += sign * conf
        total += conf

    if total <= 0:
        return {
            "bias": "Neutral",
            "confidence": 0.0,
            "score": 0.0,
            "label": "Balanced",
        }

    score = weighted / total
    abs_score = abs(score)

    if score >= 0.60:
        bias = "Bullish"
        label = "Strong bullish"
    elif score >= 0.25:
        bias = "Bullish"
        label = "Constructive bullish"
    elif score >= 0.08:
        bias = "Bullish"
        label = "Slight bullish tilt"
    elif score <= -0.60:
        bias = "Bearish"
        label = "Strong bearish"
    elif score <= -0.25:
        bias = "Bearish"
        label = "Defensive bearish"
    elif score <= -0.08:
        bias = "Bearish"
        label = "Slight bearish tilt"
    else:
        bias = "Neutral"
        label = "Balanced"

    return {
        "bias": bias,
        "confidence": abs_score,
        "score": score,
        "label": label,
    }


def render_combined_bias_panel(
    panel_title: str,
    panel_key: str,
    components: List[Dict[str, object]],
    finalized_note: Optional[str] = None,
) -> Dict[str, object]:
    result = blend_bias_components(components)
    with st.expander(panel_title, expanded=False):
        bull_p, bear_p = bias_probabilities(
            str(result.get("bias", "Neutral")),
            float(result.get("confidence", 0.0) or 0.0),
            float(result.get("score", 0.0) or 0.0),
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Combined Bias", str(result.get("bias", "Neutral")))
        c2.metric("Confidence", f"{float(result.get('confidence', 0.0)):.0%}")
        c3.metric("Blend Score", f"{float(result.get('score', 0.0)):+.2f}")
        st.caption(f"Probability: Bullish {bull_p:.0%} | Bearish {bear_p:.0%}")
        st.caption(f"Tone: {result.get('label', 'Balanced')}")

        rows = []
        for row in components:
            finalized = row.get("finalized")
            if finalized is True:
                status = "Finalized"
            elif finalized is False:
                status = "Not Finalized"
            else:
                status = "n/a"
            finalized_at = str(row.get("finalized_at", "n/a"))
            rows.append(
                {
                    "Component": row.get("name", "n/a"),
                    "Bias": row.get("bias", "Neutral"),
                    "Confidence": f"{_normalize_conf(row.get('confidence', 0.0)):.0%}",
                    "Status": status,
                    "Finalized At": finalized_at,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if finalized_note:
            st.caption(finalized_note)
    return result


def render_unified_bias_panel(
    panel_title: str,
    panel_key: str,
    unified_payload: Dict[str, object],
) -> Dict[str, object]:
    daily = unified_payload.get("daily", {}) if isinstance(unified_payload, dict) else {}
    ny_open = unified_payload.get("ny_open", {}) if isinstance(unified_payload, dict) else {}
    components_daily = unified_payload.get("components_daily", []) if isinstance(unified_payload, dict) else []
    components_ny_open = unified_payload.get("components_ny_open", []) if isinstance(unified_payload, dict) else []
    updated_at = str(unified_payload.get("updated_at", "n/a")) if isinstance(unified_payload, dict) else "n/a"

    with st.expander(panel_title, expanded=False):
        st.caption(f"Unified model update: {updated_at}")

        d_bull, d_bear = bias_probabilities(
            str(daily.get("bias", "Neutral")),
            float(daily.get("confidence", 0.0) or 0.0),
            float(daily.get("score", 0.0) or 0.0),
        )
        o_bull, o_bear = bias_probabilities(
            str(ny_open.get("bias", "Neutral")),
            float(ny_open.get("confidence", 0.0) or 0.0),
            float(ny_open.get("score", 0.0) or 0.0),
        )

        d1, d2, d3 = st.columns(3)
        d1.metric("Daily Bias", str(daily.get("bias", "Neutral")))
        d2.metric("Daily Confidence", f"{float(daily.get('confidence', 0.0) or 0.0):.0%}")
        d3.metric("Daily Score", f"{float(daily.get('score', 0.0) or 0.0):+.2f}")
        st.caption(f"Daily Probability: Bullish {d_bull:.0%} | Bearish {d_bear:.0%}")
        d_status = "Finalized" if bool(daily.get("finalized", False)) else "Not Finalized"
        st.caption(f"Daily Status: {d_status} | Finalized At: {daily.get('finalized_at', '10:45 ET')}")
        if daily.get("close_threshold_desc") is not None and daily.get("close_position_pct") is not None:
            st.caption(
                f"Daily Threshold-Close Context: {daily.get('close_threshold_desc')} "
                f"at {float(daily.get('close_position_pct', 0.0)):.1f}% of range."
            )
        st.write(f"Daily Expected Behavior: {daily.get('expected', 'n/a')}")
        st.write(f"Daily Deep Reasoning: {daily.get('reasoning', 'n/a')}")

        o1, o2, o3 = st.columns(3)
        o1.metric("NY Open Bias", str(ny_open.get("bias", "Neutral")))
        o2.metric("NY Open Confidence", f"{float(ny_open.get('confidence', 0.0) or 0.0):.0%}")
        o3.metric("NY Open Score", f"{float(ny_open.get('score', 0.0) or 0.0):+.2f}")
        st.caption(f"NY Open Probability: Bullish {o_bull:.0%} | Bearish {o_bear:.0%}")
        o_status = "Finalized" if bool(ny_open.get("finalized", False)) else "Not Finalized"
        st.caption(f"NY Open Status: {o_status} | Finalized At: {ny_open.get('finalized_at', '09:15 ET')}")
        st.write(f"NY Open Expected Behavior: {ny_open.get('expected', 'n/a')}")
        st.write(f"NY Open Deep Reasoning: {ny_open.get('reasoning', 'n/a')}")

        if components_daily:
            st.markdown("**Daily Blend Components**")
            rows = []
            for row in components_daily:
                status = "Finalized" if bool(row.get("finalized", False)) else "Not Finalized"
                rows.append(
                    {
                        "Component": row.get("name", "n/a"),
                        "Bias": row.get("bias", "Neutral"),
                        "Confidence": f"{_normalize_conf(row.get('confidence', 0.0)):.0%}",
                        "Status": status,
                        "Finalized At": row.get("finalized_at", "n/a"),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        if components_ny_open:
            st.markdown("**NY Open Blend Components**")
            rows = []
            for row in components_ny_open:
                status = "Finalized" if bool(row.get("finalized", False)) else "Not Finalized"
                rows.append(
                    {
                        "Component": row.get("name", "n/a"),
                        "Bias": row.get("bias", "Neutral"),
                        "Confidence": f"{_normalize_conf(row.get('confidence', 0.0)):.0%}",
                        "Status": status,
                        "Finalized At": row.get("finalized_at", "n/a"),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

    return unified_payload
