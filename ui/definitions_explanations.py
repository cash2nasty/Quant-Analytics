import streamlit as st


def render_definitions_explanations() -> None:
    st.header("Definitions/Explanatinos")
    st.caption(
        "Reference guide for indicators, patterns, liquidity zones, and bias logic used in this app."
    )

    st.markdown("## Liquidity")
    st.markdown(
        "Liquidity is the pool of resting orders that can be accessed when price reaches a level. "
        "Buy-side liquidity sits above swing highs (buy stops and breakout orders). "
        "Sell-side liquidity sits below swing lows (sell stops and breakdown orders). "
        "Price often moves to liquidity, runs it, then reacts based on whether new orders absorb or continue."
    )

    st.markdown("### Liquidity Zones")
    st.markdown(
        "Zones are identified on 4H, 1H, 30m, and 15m data and include FVGs, order blocks, and breakers. "
        "Confluence increases when multiple zones overlap near the current price."
    )
    st.markdown(
        "- Order block: a displacement move where a prior candle is re-used as supply/demand. "
        "Detected when a 4-bar displacement exceeds 1.2 x ATR, and the first candle's range becomes the block."
    )
    st.markdown(
        "- FVG (fair value gap): a 3-bar imbalance. Bullish if candle1.high < candle3.low; "
        "bearish if candle1.low > candle3.high."
    )
    st.markdown(
        "- Breaker block: an order block that is later broken by a close beyond its range; it flips to the opposite side."
    )
    st.markdown(
        "- Liquidity lines: swing highs/lows detected with a 3-bar window; clusters suggest thicker liquidity."
    )
    st.markdown(
        "- Zone confluence: zones near the current price are weighted by timeframe (4H 1.5, 1H 1.2, 30m 1.0, 15m 0.8)."
    )
    st.markdown(
        "- Zone setup score: weighted mix of proximity, zone size, liquidity density, volume intensity, and freshness."
    )

    st.markdown("### Premium and Discount")
    st.markdown(
        "Premium is the upper half of a reference range (above the midpoint). "
        "Discount is the lower half (below the midpoint). The midpoint is the 50 percent level of that range."
    )
    st.markdown(
        "When price is in premium, sellers are more likely to defend and mean-revert risk is higher. "
        "When price is in discount, buyers are more likely to defend and mean-revert risk is higher. "
        "On trend days, premium/discount can also mark pullback zones before continuation."
    )

    st.markdown("## Indicators and Calculations")
    st.markdown("### VWAP")
    st.markdown(
        "Daily and weekly VWAP use typical price and volume. Anchored VWAP starts at a chosen timestamp."
    )
    st.markdown(
        "$$VWAP = \\frac{\\sum (TypicalPrice \\cdot Volume)}{\\sum Volume}$$\n"
        "$$TypicalPrice = \\frac{High + Low + Close}{3}$$"
    )
    st.markdown(
        "VWAP posture compares last price vs daily and weekly VWAP to classify bullish or bearish posture."
    )
    st.markdown(
        "Anchored VWAP uses prior day high/low, overnight high/low, and US open 30m range anchors. "
        "Signals are Bullish if price is above the anchored VWAP and within 0.8 x ATR, Bearish if below, "
        "and Far if distance exceeds 1.6 x ATR."
    )

    st.markdown("### Moving Averages")
    st.markdown(
        "SMA 20 and SMA 50 are simple moving averages of close."
    )
    st.markdown(
        "$$SMA_n = \\frac{1}{n} \\sum_{i=0}^{n-1} Close_{t-i}$$"
    )

    st.markdown("### Momentum and Trend")
    st.markdown(
        "- ROC (rate of change): percent change over N bars."
    )
    st.markdown(
        "$$ROC_n = \\frac{Close_t - Close_{t-n}}{Close_{t-n}}$$"
    )
    st.markdown(
        "- Trend strength: slope of a linear regression fit to the last N closes."
    )
    st.markdown(
        "- Kalman trend: a constant-velocity Kalman filter estimates slope; the sign vs a small threshold "
        "labels bullish or bearish."
    )

    st.markdown("### Volatility and Range")
    st.markdown(
        "- Rolling volatility: standard deviation of close over N bars."
    )
    st.markdown(
        "$$Volatility_n = STD(Close, n)$$"
    )
    st.markdown(
        "- ATR-like: rolling mean of High - Low over N bars."
    )
    st.markdown(
        "$$ATRLike_n = SMA(High - Low, n)$$"
    )
    st.markdown(
        "- Volatility regime: expanded if above 70th percentile, compressed if below 30th percentile."
    )
    st.markdown(
        "- Trend classifier: higher highs and higher lows = uptrend; lower highs and lower lows = downtrend."
    )

    st.markdown("### Statistics and Volume")
    st.markdown(
        "- Z-score: distance from rolling mean in standard deviations."
    )
    st.markdown(
        "$$Z = \\frac{Close_t - Mean_n}{STD_n}$$"
    )
    st.markdown(
        "- RVOL: volume divided by rolling average volume."
    )
    st.markdown(
        "$$RVOL = \\frac{Volume_t}{SMA(Volume, n)}$$"
    )

    st.markdown("## Market Structure")
    st.markdown(
        "- BOS (break of structure): a close that breaks a prior swing high (bullish) or swing low (bearish)."
    )
    st.markdown(
        "- CHOCH (change of character): the first structural break opposite the prior trend, "
        "often signaling a potential reversal."
    )
    st.markdown(
        "The app detects BOS and CHOCH on the 1m chart using swing points and also reports 15m trend alignment."
    )

    st.markdown("## Regime")
    st.markdown(
        "Regime describes the market's tendency to trend or mean-revert. The app uses two approaches:"
    )
    st.markdown(
        "- Volatility regime: expanded or compressed based on rolling volatility percentiles."
    )
    st.markdown(
        "- HMM regime: a 2-state Gaussian model over returns, labeled Trend or MeanRevert by state bias."
    )

    st.markdown("## Session Sweeps and Liquidity Runs")
    st.markdown(
        "A session sweep happens when a later session trades beyond the prior session's high or low, "
        "then closes back inside that prior range."
    )
    st.markdown(
        "- General look: a spike through a session high/low, then a close back inside the range."
    )
    st.markdown(
        "- 1m chart look: a fast push through the level with a rejection wick and several closes back inside."
    )

    st.markdown("## Opening Range (OR) and Failed OR")
    st.markdown(
        "- 30m OR: high/low from 09:30 to 10:00 ET."
    )
    st.markdown(
        "- 60m OR: high/low from 09:30 to 10:30 ET."
    )
    st.markdown(
        "- OR break: a close beyond the OR high/low by 10 percent of OR range."
    )
    st.markdown(
        "- Failed OR: price breaks the OR and then closes back inside the range within the next 30 minutes."
    )
    st.markdown(
        "**What a non-failed OR looks like (general):** a clean break with follow-through and no immediate close "
        "back inside the OR.\n"
        "**What it looks like on the 1m chart:** multiple closes beyond the OR edge, optional retest that holds." 
    )
    st.markdown(
        "**What a failed OR looks like (general):** a breakout that is rejected and returns inside the OR.\n"
        "**What it looks like on the 1m chart:** a spike beyond the OR edge, then quick closes back inside "
        "and continuation toward the opposite side."
    )

    st.markdown("## Patterns")
    st.markdown(
        "- London breakout: London trades above Asia high and below Asia low in the same session."
    )
    st.markdown(
        "- Asia range hold: London stays inside the Asia range."
    )
    st.markdown(
        "- Asia range sweep: London sweeps Asia high or low and closes back inside the Asia range."
    )
    st.markdown(
        "- London continuation: after London breaks Asia, US extends beyond London in the same direction."
    )
    st.markdown(
        "- US open gap fill: a gap >= 20 percent of the prior day range is filled by 10:30 ET."
    )
    st.markdown(
        "- ORB 30m/60m: first decisive break beyond the OR range after the OR window completes."
    )
    st.markdown(
        "- Failed ORB 30m/60m: a break of OR that closes back inside within 30 minutes."
    )
    st.markdown(
        "- Power hour trend: 14:00-16:00 ET move exceeds 30 percent of the day range."
    )
    st.markdown(
        "- VWAP reclaim/reject: price crosses VWAP and then holds for 4 consecutive closes on the new side."
    )
    st.markdown(
        "  Reclaim means price was below VWAP, reclaims it, and holds above; after reclaim, "
        "look for continuation toward prior highs, anchored VWAP targets, or the next liquidity zone."
    )
    st.markdown(
        "  Reject means price was above VWAP, loses it, and holds below; after reject, "
        "look for continuation toward prior lows, anchored VWAP targets, or the next liquidity zone."
    )
    st.markdown(
        "- Whipsaw: London and US sessions move in opposite directions."
    )

    st.markdown("### US Open Reclaim Pattern (Watch Setup)")
    st.markdown(
        "This is a **heads-up pattern** for the US open that aims to catch the real move early. "
        "It is treated as a secondary setup and does not replace the core playbook decision."
    )
    st.markdown(
        "- Core structure: the 09:30 candle shows two-sided rejection (long upper and lower wick), "
        "and the 09:31 candle closes green."
    )
    st.markdown(
        "- Confirmation: price then reclaims Daily VWAP (close back above VWAP) before 10:30 ET."
    )
    st.markdown(
        "- Anticipation mode (before 09:30 ET): the app scores pre-open structure, sweep behavior, "
        "VWAP posture, and nearby liquidity confluences to issue a watch alert with provisional entry/stop/target rules."
    )
    st.markdown(
        "- Why it matters: this can signal a failed early sell impulse and rotation back toward upside liquidity."
    )
    st.markdown(
        "- Typical plan after confirmation: entry on reclaim close, stop at setup low, "
        "targets at pre-open high and extension levels."
    )
    st.markdown(
        "- Invalidation: no VWAP reclaim by 10:30 ET, or post-trigger failure below setup low."
    )

    st.markdown("## New Playbook Analytics")
    st.markdown(
        "### Continuous Trigger Monitoring"
    )
    st.markdown(
        "The playbook now keeps monitoring for additional confirmations after the first trigger fires. "
        "Overnight confirmations can start context, while OR triggers (30m and 60m) can still upgrade or downgrade execution confidence."
    )
    st.markdown(
        "ORB timing rule: ORB 30m only after 10:00 ET and ORB 60m only after 10:30 ET, using current-day windows only."
    )

    st.markdown("### Risk Engine")
    st.markdown(
        "Risk engine combines directional bias, confluence zone context, volatility, volume participation, momentum, and VWAP probabilities. "
        "It outputs side (long/short/wait), entry-stop-target, risk-reward, size guidance, and a risk-quality status."
    )
    st.markdown(
        "General interpretation: higher risk-quality score is better. Low score means stand down or require stronger confirmation."
    )

    st.markdown("### Entry Confidence With Reasoning")
    st.markdown(
        "Each confluence entry suggestion includes an Entry Confidence score (0 to 100), a tier, and explicit reasoning."
    )
    st.markdown(
        "The confidence score blends: risk quality, confluence reaction strength, momentum alignment, volume participation, "
        "VWAP expansion/reversion context, trigger alignment, and penalties (whipsaw, invalidation, wait/no-trade state)."
    )
    st.markdown(
        "Guide: below 55 is low confidence, 55-74 is moderate, 75+ is high confidence."
    )

    st.markdown("### Drawdown, EV, and Sharpe")
    st.markdown(
        "- Drawdown: peak-to-trough decline of the return curve. Closer to 0 percent is better."
    )
    st.markdown(
        "- Expected value (EV): average edge per return observation. Positive is good, near zero is weak, negative is bad."
    )
    st.markdown(
        "- Raw Sharpe: return mean divided by return standard deviation at native frequency."
    )
    st.markdown(
        "Practical Sharpe guide: around 0.5 is modest, around 1.0 is good, above 1.5 is strong, below 0 is weak."
    )

    st.markdown("### NY VWAP Mean Reversion vs Expansion Probability")
    st.markdown(
        "This model estimates two probabilities during NY session:"
    )
    st.markdown(
        "- Mean reversion probability: odds price rotates back toward daily VWAP."
    )
    st.markdown(
        "- Expansion probability: odds price continues away from VWAP on the current side."
    )
    st.markdown(
        "Inputs include price location vs VWAP, time above/below VWAP, RVOL, momentum tilt, and stretch."
    )

    st.markdown("### VWAP Strength After Dip Score")
    st.markdown(
        "Dip-strength score is reported out of 100 and qualifies reclaim quality after price dips below VWAP and reclaims it."
    )
    st.markdown(
        "Guide: below 45 is weak, 45-69 is moderate, 70+ is strong reclaim quality."
    )
    st.markdown(
        "The model also provides expected entry timing and retest behavior guidance."
    )
    st.markdown(
        "It now also includes a brief 'What to expect' summary describing likely rest-of-day behavior based on current dip-quality data."
    )

    st.markdown("### Volatility Regime, Volume Detector, and Momentum Prediction")
    st.markdown(
        "- Volatility regime: compressed, normal, or expanded based on rolling STDV percentiles."
    )
    st.markdown(
        "Compressed often means tighter ranges and breakout potential, expanded means larger swings and higher invalidation risk."
    )
    st.markdown(
        "- Volume detector uses RVOL and participation scoring. RVOL below 0.85 is weak, 0.85-1.25 is normal, above 1.25 is strong."
    )
    st.markdown(
        "- Momentum prediction provides bullish vs bearish probability split and confidence. "
        "Confidence below 15 is weak, 15-35 is moderate, above 35 is strong."
    )
    st.markdown(
        "Each block includes a brief 'What to expect' summary for practical rest-of-day expectations."
    )

    st.markdown("### Drawdown, EV, and Sharpe Practical Outlook")
    st.markdown(
        "The app now displays a short 'What to expect' line translating current drawdown, EV, and Sharpe into an execution stance for the rest of the session."
    )

    st.markdown("## How Data Determines Price Action")
    st.markdown(
        "- VWAP posture guides mean-reversion vs continuation around fair value."
    )
    st.markdown(
        "- OR and OR failures define early trend vs fade behavior."
    )
    st.markdown(
        "- Liquidity zones identify where price is likely to react or accelerate."
    )
    st.markdown(
        "- Regime (trend vs mean-revert) shifts expectation for continuation or snap-back."
    )
    st.markdown(
        "- Patterns (sweeps, gap fills, continuation) indicate whether liquidity was taken and whether follow-through is likely."
    )

    st.markdown("## Daily Bias Section: Calculations and Meaning")
    st.markdown(
        "**Daily Bias** is a session-weighted vote using previous day, Asia, and London. "
        "Weights are proportional to each session's range."
    )
    st.markdown(
        "**Overnight Range Position** compares US open to the Asia/London range: above, below, or inside."
    )
    st.markdown(
        "**Gap Label** compares the US open to the prior close; size is small/medium/large based on prior range."
    )
    st.markdown(
        "**Day Type** is Trend, Chop, or Normal based on overnight range size vs prior range and ROC magnitude."
    )
    st.markdown(
        "**Daily Confidence** starts at 0.50 and is adjusted by alignment factors: session agreement, "
        "anchored VWAP alignment, overnight range position, gap direction, HTF slope, VWAP posture, "
        "volatility regime, momentum, RVOL, regime type, day type (trend vs chop), and event risk."
    )
    st.markdown(
        "**US Open Bias 30m/60m** comes from a premarket signal (18:00 to 09:10 ET) that blends:"
    )
    st.markdown(
        "- Premarket trend direction (open to close)\n"
        "- Gap direction vs the 09:30 open\n"
        "- Premarket VWAP posture and open vs VWAP\n"
        "- SMA 20/50 on premarket data\n"
        "- ROC and trend strength\n"
        "- Volatility regime and RVOL\n"
        "- Z-score and standard deviation expansion\n"
        "- HTF slope bias and zone confluence bias\n"
        "- Overnight range position and AMD sweep bias\n"
        "- Pattern bias (Asia sweep or London continuation)\n"
    )
    st.markdown(
        "**VWAP Posture** compares last price to daily and weekly VWAP and labels bullish/bearish posture."
    )
    st.markdown(
        "**News Effect** uses the news tone (risk_on or risk_off) as a confidence modifier; it does not flip direction."
    )
    st.markdown(
        "**AMD Summary** uses Asia sweep and London close position to infer accumulation/manipulation/distribution bias."
    )
    st.markdown(
        "**Final Bias** is refined by anchored VWAP signals, volatility regime, momentum, RVOL, "
        "day type (trend vs chop), gaps, event risk, and pattern alignment. It can be overridden "
        "by Bayesian or ensemble scores when confidence is high."
    )
    st.markdown(
        "**Explanation** is a generated narrative that lists inputs, regime, gaps, event risk, and impact notes "
        "so you can see why the bias and confidence were set."
    )
    st.markdown(
        "**Finalization**: daily bias is confirmed after 10:45 ET using the 60m opening range signal."
    )
