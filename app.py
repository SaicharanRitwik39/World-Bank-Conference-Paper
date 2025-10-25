# streamlit run alignment_index_streamlit_app.py
# Alignment Index Dashboard for Ten India-Focused Policy Forecasts
# Author: Saicharan Ritwik (with assistant help)
# Date: 2025-10-25
# ---------------------------------------------------------------
# Features
# - Sliders to adjust each forecast's probability (default = paper values)
# - Toggle pillars (Trade=T, Tech=X, Finance=F, Strategic=S); weights re-normalize
# - Per-pillar scores and overall Alignment Index (AI = 50*(1+S))
# - One-at-a-time (OAT) sensitivity to ±10pp changes in a single probability
# - Pillar-weight sensitivity (analytical \u2202AI/\u2202w_k) and quick reallocation calculator
# - Download/Upload current configuration as JSON
# ---------------------------------------------------------------

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Alignment Index Dashboard",
    page_icon="📊",
    layout="wide",
)

# -------------------------------
# Data model
# -------------------------------
PILLARS = ["T", "X", "F", "S"]  # Trade, Tech, Finance/Payments, Strategic
PILLAR_NAMES = {"T": "Trade", "X": "Technology", "F": "Finance/Payments", "S": "Strategic"}

@dataclass
class Forecast:
    id: str
    label: str
    p: float  # probability in [0,1]
    # membership per pillar (1 if belongs, else 0). Equal-split shares are computed from this.
    B: Dict[str, int]
    # signs per pillar: +1 West, -1 East, 0 Neutral
    s: Dict[str, int]

    def shares(self) -> Dict[str, float]:
        d_i = sum(self.B.get(k, 0) for k in PILLARS)
        if d_i == 0:
            return {k: 0.0 for k in PILLARS}
        return {k: (self.B.get(k, 0) / d_i) for k in PILLARS}

# ----------------------------------------------
# Paper defaults (from the provided PDF's codebook)
# ----------------------------------------------
# Notes on mapping and signs (derived from the paper's Table 2 contributions):
# F1: T +1; F2: T -1; F3: Neutral s=0; F4: X +1, F +1 (membership on both, equal split);
# F5: F -1; F6: T +1, X +1; F7: S -1; F8: T +1; F9: T -1; F10: S +1.
# Default probabilities p_i: F1 0.68, F2 0.22, F3 0.09, F4 0.08, F5 0.66,
# F6 0.64, F7 0.45, F8 0.95, F9 0.46, F10 0.88.

DEFAULT_FORECASTS: List[Forecast] = [
    Forecast("F1", "India–EU FTA (text by 2026)", 0.68, {"T": 1}, {"T": +1}),
    Forecast("F2", "India joins RCEP (by 2028)", 0.22, {"T": 1}, {"T": -1}),
    Forecast("F3", "Indian firm in global Top-10 fabless (2028)", 0.09, {"X": 1}, {"X": 0}),
    Forecast("F4", "e₹ ↔ FedNow live interoperability (by 2028)", 0.08, {"X": 1, "F": 1}, {"X": +1, "F": +1}),
    Forecast("F5", "India’s cumulative NDB approvals > $12B (by 2027)", 0.66, {"F": 1}, {"F": -1}),
    Forecast("F6", "New formal restrictions on Chinese tech imports (by 2027)", 0.64, {"T": 1, "X": 1}, {"T": +1, "X": +1}),
    Forecast("F7", "Russia >40% of India’s major-arms imports in any 2025–27 year", 0.45, {"S": 1}, {"S": -1}),
    Forecast("F8", "U.S.–India goods trade 2025 > 2024", 0.95, {"T": 1}, {"T": +1}),
    Forecast("F9", "Lowy Economic Relationships rank ≤ 9 (2026)", 0.46, {"T": 1}, {"T": -1}),
    Forecast("F10", "India attends >85% Quad Leaders/FMs meetings (2025–27)", 0.88, {"S": 1}, {"S": +1}),
]

# ----------------------------------------------
# Helper functions
# ----------------------------------------------

def compute_pillar_counts(forecasts: List[Forecast], active_pillars: List[str]) -> Dict[str, float]:
    Nk = {k: 0.0 for k in PILLARS}
    for f in forecasts:
        C = f.shares()
        for k in active_pillars:
            Nk[k] += C.get(k, 0.0)
    return Nk


def compute_pillar_scores(
    forecasts: List[Forecast],
    active_pillars: List[str],
    weights: Dict[str, float],
) -> Tuple[Dict[str, float], float, float, pd.DataFrame]:
    """Returns per-pillar score dict, S in [-1,1], AI in [0,100], and a contributions table."""
    Nk = compute_pillar_counts(forecasts, active_pillars)
    numerators = {k: 0.0 for k in PILLARS}

    rows = []
    for f in forecasts:
        C = f.shares()
        row = {
            "ID": f.id,
            "Forecast": f.label,
            "p": f.p,
        }
        for k in PILLARS:
            contrib = C.get(k, 0.0) * f.s.get(k, 0) * f.p
            numerators[k] += contrib if k in active_pillars else 0.0
            row[f"{PILLAR_NAMES[k]} contrib"] = contrib
        rows.append(row)
    contrib_df = pd.DataFrame(rows)

    pillar_scores = {}
    for k in PILLARS:
        if k not in active_pillars or Nk[k] == 0:
            pillar_scores[k] = 0.0
        else:
            pillar_scores[k] = numerators[k] / Nk[k]

    # normalized weights among active pillars
    active_weight_sum = sum(weights[k] for k in active_pillars)
    if active_weight_sum == 0:
        wk_norm = {k: 0.0 for k in PILLARS}
    else:
        wk_norm = {k: (weights[k] / active_weight_sum if k in active_pillars else 0.0) for k in PILLARS}

    S = sum(wk_norm[k] * pillar_scores[k] for k in PILLARS)
    AI = 50.0 * (1.0 + S)
    return pillar_scores, S, AI, contrib_df


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def oat_sensitivity(
    forecasts: List[Forecast], active_pillars: List[str], weights: Dict[str, float]
) -> pd.DataFrame:
    base_scores, base_S, base_AI, _ = compute_pillar_scores(forecasts, active_pillars, weights)
    base_AI = float(base_AI)

    rows = []
    for idx, f in enumerate(forecasts):
        # Up/down 10pp perturbations
        p_up = clamp01(f.p + 0.10)
        p_down = clamp01(f.p - 0.10)

        # mutate, compute, revert
        orig = f.p
        f.p = p_up
        _, _, AI_up, _ = compute_pillar_scores(forecasts, active_pillars, weights)
        f.p = p_down
        _, _, AI_down, _ = compute_pillar_scores(forecasts, active_pillars, weights)
        f.p = orig

        d_up = float(AI_up) - base_AI
        d_down = float(AI_down) - base_AI
        max_abs = float(max(abs(d_up), abs(d_down)))
        slope = (d_up - d_down) / 0.20 if (p_up != p_down) else 0.0

        rows.append({
            "ID": f.id,
            "Forecast": f.label,
            "p": orig,
            "∆AI↑ (+10pp)": d_up,
            "∆AI↓ (−10pp)": d_down,
            "max |∆AI|": max_abs,
            "\u005Eslope dAI/dp": slope,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="max |∆AI|", ascending=False).reset_index(drop=True)
    return df


def analytical_weight_sensitivity(pillar_scores: Dict[str, float]) -> pd.DataFrame:
    # dAI/dw_k = 50 * PillarScore_k
    rows = []
    for k in PILLARS:
        rows.append({
            "Pillar": PILLAR_NAMES[k],
            "Score (in [-1,1])": pillar_scores.get(k, 0.0),
            "∂AI/∂w_k": 50.0 * pillar_scores.get(k, 0.0),
        })
    return pd.DataFrame(rows)


def apply_weight_reallocation(
    pillar_scores: Dict[str, float], from_k: str, to_k: str, delta: float
) -> float:
    """Closed-form change in AI when moving 'delta' weight from from_k to to_k (others unchanged).
       ∆AI = 50 * δ * (Score_to - Score_from)
    """
    return 50.0 * delta * (pillar_scores.get(to_k, 0.0) - pillar_scores.get(from_k, 0.0))


# ----------------------------------------------
# Sidebar: configuration & IO
# ----------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    st.markdown("**Active pillars**")
    active_pillars = st.multiselect(
        "Toggle pillars",
        options=[PILLAR_NAMES[k] for k in PILLARS],
        default=[PILLAR_NAMES[k] for k in PILLARS],
    )
    # map back to keys
    active_keys = [k for k in PILLARS if PILLAR_NAMES[k] in active_pillars]
    if not active_keys:
        st.warning("Select at least one pillar to compute the index.")

    st.markdown("**Pillar weights** (re-normalized over active pillars)")
    weight_inputs = {}
    for k in PILLARS:
        default = 0.25
        weight_inputs[k] = st.slider(
            f"{PILLAR_NAMES[k]} weight",
            min_value=0.0,
            max_value=1.0,
            value=default,
            step=0.01,
            help="Set to 0 to ignore this pillar; remaining active weights are re-normalized automatically.",
        )

    st.caption("Weights shown are raw inputs; the dashboard normalizes across active pillars before computing S and AI.")

    st.divider()
    st.markdown("**Import/Export**")
    if st.button("Reset probabilities to paper defaults"):
        st.session_state["prob_reset_flag"] = True
    else:
        st.session_state.setdefault("prob_reset_flag", False)

    uploaded = st.file_uploader("Upload settings JSON (optional)", type=["json"])
    if uploaded is not None:
        try:
            cfg = json.load(uploaded)
            # Set session-level overrides
            st.session_state["uploaded_cfg"] = cfg
            st.success("Configuration loaded. Scroll to probabilities and weights to see applied values.")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

# Apply uploaded weights if present
if "uploaded_cfg" in st.session_state:
    cfg = st.session_state["uploaded_cfg"]
    # weights
    if "weights" in cfg and isinstance(cfg["weights"], dict):
        for k in PILLARS:
            if k in cfg["weights"]:
                weight_inputs[k] = float(cfg["weights"][k])
    # probabilities
    if "probabilities" in cfg and isinstance(cfg["probabilities"], dict):
        for f in DEFAULT_FORECASTS:
            if f.id in cfg["probabilities"]:
                try:
                    f.p = clamp01(float(cfg["probabilities"][f.id]))
                except Exception:
                    pass

# ----------------------------------------------
# Main layout
# ----------------------------------------------
st.title("📊 Alignment Index — Interactive Dashboard")

st.markdown(
    "This dashboard implements the paper's **Alignment Index** with ten India-focused forecasts. "
    "Adjust probabilities and pillar weights, toggle pillars, and inspect sensitivity analyses."
)

# Probabilities editor
st.header("1) Forecast Probabilities (p_i)")

cols = st.columns(2)

# probabilities grid
prob_overrides = {}
for i, f in enumerate(DEFAULT_FORECASTS):
    col = cols[i % 2]
    with col:
        default_val = f.p if not st.session_state.get("prob_reset_flag", False) else next(ff.p for ff in DEFAULT_FORECASTS if ff.id == f.id)
        prob_overrides[f.id] = st.slider(
            f"{f.id}: {f.label}",
            min_value=0.0,
            max_value=1.0,
            value=float(default_val),
            step=0.01,
        )

# assign back
for f in DEFAULT_FORECASTS:
    f.p = prob_overrides[f.id]

# Active weights dict
weights = {k: float(weight_inputs[k]) for k in PILLARS}

# Compute scores
pillar_scores, S, AI, contrib_df = compute_pillar_scores(DEFAULT_FORECASTS, active_keys, weights)

# KPI row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Alignment Index (AI)", f"{AI:0.3f}")
with kpi2:
    st.metric("Composite S (−1↔+1)", f"{S:0.3f}")
with kpi3:
    st.metric("Active pillars", len(active_keys))
with kpi4:
    st.metric("Events (N)", len(DEFAULT_FORECASTS))

st.subheader("Per-Pillar Scores")
ps_df = pd.DataFrame({
    "Pillar": [PILLAR_NAMES[k] for k in PILLARS],
    "Score (−1↔+1)": [pillar_scores[k] for k in PILLARS],
}).set_index("Pillar")
st.dataframe(ps_df, use_container_width=True)
st.bar_chart(ps_df)

st.subheader("Event Contributions by Pillar")
st.dataframe(contrib_df, use_container_width=True)

st.divider()

# ----------------------------------------------
# Sensitivity analyses
# ----------------------------------------------

st.header("2) Sensitivity Analysis")

# 2a) One-at-a-time (OAT) ±10pp
st.subheader("2a. One-at-a-time (±10pp) on probabilities")
oat_df = oat_sensitivity(DEFAULT_FORECASTS, active_keys, weights)
st.dataframe(oat_df, use_container_width=True)

st.markdown("**Most influential events** by |∆AI| (higher bars = bigger impact):")
st.bar_chart(oat_df.set_index("ID")["max |∆AI|"])

# 2b) Pillar-weight sensitivity (analytical)
st.subheader("2b. Pillar-weight sensitivity (analytical)")
ws_df = analytical_weight_sensitivity(pillar_scores)
st.dataframe(ws_df, use_container_width=True)

with st.expander("Quick reallocation calculator (10–20 seconds to explore)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        from_k = st.selectbox("From pillar", [PILLAR_NAMES[k] for k in PILLARS])
    with c2:
        to_k = st.selectbox("To pillar", [PILLAR_NAMES[k] for k in PILLARS])
    with c3:
        delta = st.slider("Reallocate weight δ (0 to 0.50)", 0.0, 0.50, 0.10, 0.01)

    # map back to keys
    rev_map = {v: k for k, v in PILLAR_NAMES.items()}
    if from_k == to_k:
        st.info("Choose different pillars for a meaningful reallocation.")
    else:
        dAI = apply_weight_reallocation(pillar_scores, rev_map[from_k], rev_map[to_k], delta)
        st.write(f"**Closed-form ∆AI** for moving {delta:0.02f} from **{from_k}** to **{to_k}**: **{dAI:+0.3f}** points")
        st.caption("Formula: ∆AI = 50 · δ · (Score_to − Score_from)")

st.divider()

# ----------------------------------------------
# Download current configuration
# ----------------------------------------------

st.header("3) Save / Load Configuration")
cfg = {
    "weights": weights,
    "active_pillars": active_keys,
    "probabilities": {f.id: f.p for f in DEFAULT_FORECASTS},
}
st.download_button(
    label="Download current settings as JSON",
    data=json.dumps(cfg, indent=2),
    file_name="alignment_index_settings.json",
    mime="application/json",
)

st.caption(
    "Definitions: PillarScore_k = (1/N_k) · Σ_i C_ik · s_ik · p_i; S = Σ_k w_k · PillarScore_k; AI = 50·(1+S).\n"
    "Here C_ik = B_ik / d_i shares a multi-pillar event equally; signs s_ik: +1 West, −1 East, 0 Neutral."
)