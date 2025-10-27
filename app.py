from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Alignment Index", page_icon="📊", layout="wide")

PILLARS = ["T", "X", "F", "S"]
PILLAR_NAMES = {"T": "Trade", "X": "Technology", "F": "Finance/Payments", "S": "Strategic"}

@dataclass
class Forecast:
    id: str
    label: str
    p: float
    B: Dict[str, int]  # membership 0/1 per pillar
    s: Dict[str, int]  # sign -1/0/+1 per pillar
    def shares(self) -> Dict[str, float]:
        d = sum(self.B.get(k, 0) for k in PILLARS)
        if d == 0:
            return {k: 0.0 for k in PILLARS}
        return {k: (self.B.get(k, 0) / d) for k in PILLARS}

DEFAULTS: List[Forecast] = [
    Forecast("F1", "India–EU FTA (text by 2026)", 0.68, {"T":1}, {"T":+1}),
    Forecast("F2", "India joins RCEP (by 2028)", 0.22, {"T":1}, {"T":-1}),
    Forecast("F3", "Indian firm in global Top-10 fabless (2028)", 0.09, {"X":1}, {"X":0}),
    Forecast("F4", "e₹ ↔ FedNow live interoperability (by 2028)", 0.08, {"X":1, "F":1}, {"X":+1, "F":+1}),
    Forecast("F5", "India’s cumulative NDB approvals > $12B (by 2027)", 0.66, {"F":1}, {"F":-1}),
    Forecast("F6", "New formal restrictions on Chinese tech imports (by 2027)", 0.64, {"T":1, "X":1}, {"T":+1, "X":+1}),
    Forecast("F7", "Russia >40% of India’s major-arms imports in any 2025–27 year", 0.45, {"S":1}, {"S":-1}),
    Forecast("F8", "U.S.–India goods trade 2025 > 2024", 0.95, {"T":1}, {"T":+1}),
    Forecast("F9", "Lowy Economic Relationships rank ≤ 9 (2026)", 0.46, {"T":1}, {"T":-1}),
    Forecast("F10", "India attends >85% Quad Leaders/FMs meetings (2025–27)", 0.88, {"S":1}, {"S":+1}),
]

# ------------------------------
# Core math
# ------------------------------

def pillar_counts(forecasts: List[Forecast], active: List[str]) -> Dict[str, float]:
    Nk = {k: 0.0 for k in PILLARS}
    for f in forecasts:
        C = f.shares()
        for k in active:
            Nk[k] += C.get(k, 0.0)
    return Nk


def compute_scores(forecasts: List[Forecast], active: List[str], weights: Dict[str, float]) -> Tuple[Dict[str,float], float, float]:
    """Return pillar_scores in [-1,1], S, AI.
    Weights are normalized over active pillars before computing S.
    """
    Nk = pillar_counts(forecasts, active)
    numer = {k: 0.0 for k in PILLARS}
    for f in forecasts:
        C = f.shares()
        for k in active:
            numer[k] += C.get(k,0.0) * f.s.get(k,0) * f.p
    ps = {}
    for k in PILLARS:
        ps[k] = 0.0 if (k not in active or Nk[k]==0) else numer[k]/Nk[k]
    # normalize weights across active pillars
    active_sum = sum(weights.get(k, 0.0) for k in active)
    if len(active) == 0 or active_sum == 0:
        S = 0.0
    else:
        w_norm = {k: (weights.get(k,0.0)/active_sum if k in active else 0.0) for k in PILLARS}
        S = sum(w_norm[k]*ps[k] for k in active)
    AI = 50.0 * (1.0 + S)
    return ps, S, AI


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def oat_sensitivity_simple(forecasts: List[Forecast], active: List[str]) -> pd.DataFrame:
    # Baseline
    base_ps, base_S, base_AI = compute_scores(forecasts, active)

    rows = []
    for f in forecasts:
        orig = f.p
        # +10pp
        f.p = clamp01(orig + 0.10)
        _, _, AI_up = compute_scores(forecasts, active)
        # -10pp
        f.p = clamp01(orig - 0.10)
        _, _, AI_dn = compute_scores(forecasts, active)
        # revert
        f.p = orig
        rows.append({
            "ID": f.id,
            "Forecast": f.label,
            "p": orig,
            "ΔAI↑ (+10pp)": AI_up - base_AI,
            "ΔAI↓ (−10pp)": AI_dn - base_AI,
        })
    df = pd.DataFrame(rows)
    df["|ΔAI|max"] = df[["ΔAI↑ (+10pp)", "ΔAI↓ (−10pp)"]].abs().max(axis=1)
    return df.sort_values("|ΔAI|max", ascending=False).reset_index(drop=True)

# ------------------------------
# UI — Minimal
# ------------------------------

st.title("📊 Alignment Index — Interactive Dashboard")
st.caption("Equal weights across active pillars. Adjust probabilities and pillar set; see per-pillar scores and overall index.")
with st.sidebar:
    st.header("Weights (sum to 100%) — 3 sliders + 1 auto")
    st.caption("Set Trade, Technology, and Finance/Payments. The Strategic weight is computed automatically so the total is 100%.")

    # Persist previous values for smoother UX
    st.session_state.setdefault("w_T", 25)
    st.session_state.setdefault("w_X", 25)
    st.session_state.setdefault("w_F", 25)

    remaining = 100
    w_T = st.slider("Trade (%)", 0, remaining, int(st.session_state["w_T"]), 1)
    remaining -= w_T
    w_X = st.slider("Technology (%)", 0, remaining, int(min(st.session_state["w_X"], remaining)), 1)
    remaining -= w_X
    w_F = st.slider("Finance/Payments (%)", 0, remaining, int(min(st.session_state["w_F"], remaining)), 1)
    remaining -= w_F
    w_S = remaining  # auto-computed so the sum is exactly 100

    # Save back to session
    st.session_state["w_T"], st.session_state["w_X"], st.session_state["w_F"] = w_T, w_X, w_F

    st.markdown(f"**Strategic (%)**: `{w_S}` (auto)")
    st.progress((w_T + w_X + w_F + w_S) / 100.0, text="Total = 100%")

    # Convert to fractions for the model (and show a small summary table)
    WEIGHTS = {"T": w_T/100.0, "X": w_X/100.0, "F": w_F/100.0, "S": w_S/100.0}
    w_df = pd.DataFrame({
        "Pillar": [PILLAR_NAMES[k] for k in PILLARS],
        "%": [w_T, w_X, w_F, w_S],
        "fraction": [WEIGHTS[k] for k in PILLARS],
    })
    st.dataframe(w_df, use_container_width=True)

# All pillars are considered active in this simplified build
ACTIVE = PILLARS

st.header("Forecast Probabilities")

# Paper default probabilities (used only for initial values)
_PAPER_P_DEFAULTS = {
    "F1": 0.68, "F2": 0.22, "F3": 0.09, "F4": 0.08, "F5": 0.66,
    "F6": 0.64, "F7": 0.45, "F8": 0.95, "F9": 0.46, "F10": 0.88,
}

# Initialize session state for probabilities (persist across reruns)
for f in DEFAULTS:
    st.session_state.setdefault(f"p_{f.id}", _PAPER_P_DEFAULTS[f.id])

st.caption("Set the current probabilities for each forecast. (Defaults match the paper; adjust as needed.)")

cols = st.columns(2)
for i, f in enumerate(DEFAULTS):
    with cols[i % 2]:
        st.session_state[f"p_{f.id}"] = st.slider(
            f"{f.id}: {f.label}", 0.0, 1.0, float(st.session_state[f"p_{f.id}"]), 0.01
        )
        f.p = float(st.session_state[f"p_{f.id}"])

# Compute
ps, S, AI = compute_scores(DEFAULTS, ACTIVE, WEIGHTS)

# KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Alignment Index (AI)", f"{AI:0.3f}")
with k2:
    st.metric("Composite S (−1↔+1)", f"{S:0.3f}")
with k3:
    st.metric("Active pillars", len(ACTIVE))
with k4:
    st.metric("Events (N)", len(DEFAULTS))

#st.subheader("Per-Pillar Scores")
ps_df = pd.DataFrame({
    "Pillar": [PILLAR_NAMES[k] for k in PILLARS],
    "Score (−1↔+1)": [ps[k] for k in PILLARS],
}).set_index("Pillar")
st.subheader("📊 Pillar Scores Data")
st.dataframe(ps_df, use_container_width=True)
st.markdown("---")
st.subheader("📈 Pillar Scores Chart")
st.bar_chart(ps_df)

st.markdown("---")
st.latex(r"""
\textbf{Definitions:}\\
\mathrm{PillarScore}_k = \frac{1}{N_k} \sum_i C_{ik} s_{ik} p_i, \quad
S = \sum_k w_k^{\mathrm{norm}} \cdot \mathrm{PillarScore}_k, \quad
\mathrm{AI} = 50(1 + S), \quad
C_{ik} = \frac{B_{ik}}{d_i}.
""")