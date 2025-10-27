# 🧭 Alignment Index for Policy-Relevant Forecasts  
*A West–East Tilt Meter with Per-Pillar Decomposition*  

**Author:** [Saicharan Ritwik Chinni](https://github.com/SaicharanRitwik39)  
**Affiliation:** Independent Researcher (New Delhi, India)  
**Date:** October 27, 2025  

---

## 📘 Overview  

This repository contains the replication package and code for the paper:  

> **An Alignment Index for Policy-Relevant Forecasts: A West–East Tilt Meter with Per-Pillar Decomposition**  
> *(World Bank–ECB–Bank of Italy Conference: “Trade, Value Chains, and Financial Linkages in the Global Economy”, Rome, Dec 2025)*  

The **Alignment Index (AI)** is a transparent, axiomatic framework that translates a panel of **policy-salient probabilistic forecasts** into a single, interpretable measure of a country’s **strategic and economic orientation** across four pillars:  

| Pillar | Description | Example Dimension |
|:-------:|--------------|-------------------|
| **T** | Trade | FTAs, RCEP accession, export dynamics |
| **X** | Technology | Tech-import restrictions, semiconductor ecosystem |
| **F** | Finance/Payments | Cross-border payments, MDB borrowing |
| **S** | Strategic | Defence sourcing, multilateral alignment |

The paper demonstrates the framework using **ten India-focused forecasts (2025–2028)** and provides per-pillar decomposition, sensitivity analysis, and policy implications.

---

## 🌐 Interactive Dashboard  

🔗 **Live app:** [alignment-index.streamlit.app](https://alignment-index.streamlit.app/)  

The accompanying **Streamlit dashboard** allows policymakers and analysts to:  
- Adjust forecast probabilities interactively via sliders  
- Change pillar weights (Trade, Tech, Finance, Strategic)  
- View the resulting Alignment Index (`AI ∈ [0,100]`) and per-pillar scores  
- Explore *One-at-a-Time (OAT)* sensitivities to identify high-impact forecasts  

Each update reflects how incremental changes in forecast probabilities translate into **directional shifts in alignment** — West-leaning (>50) or East-leaning (<50).

---

## 🧮 Core Formula Interpretation:
- `AI = 50` → Neutral alignment  
- `AI > 50` → West-lean  
- `AI < 50` → East-lean  

---

## 📈 Example (India, 2025 Baseline)

| Pillar | Score | Interpretation |
|:-------:|:------:|:----------------|
| Trade | +0.282 | West-lean (FTA progress, robust US trade) |
| Technology | +0.180 | Moderate West-lean (tech-import restrictions) |
| Finance | −0.413 | East-lean (NDB borrowing, BRICS finance) |
| Strategic | +0.215 | Balanced-West (Quad participation, Russia exposure) |

**Composite Index:** `AI = 53.3` → *Moderate West-lean with deliberate hedging.*

---

## 🔍 Sensitivity Analysis  

**Most influential forecasts:**  
| ID | Forecast | Sensitivity (±10pp) | Pillar(s) |
|----|-----------|----------------------|-----------|
| F5 | India’s borrowing from BRICS NDB > $12B | ±0.83 | Finance |
| F4 | e-Rupee ↔ FedNow interoperability | ±0.73 | Finance, Tech |
| F10 | India’s QUAD participation | ±0.63 | Strategic |
| F6 | New formal restrictions on Chinese tech | ±0.45 | Trade, Tech |
| F1 | EU–India FTA progress | ±0.28 | Trade |

---

## 🧩 Features  

- 📊 **Per-pillar decomposition** for interpretability  
- 🔁 **Quarterly & event-driven updating** (2025–2028)  
- 🧠 **Probabilistic forecasting integration** (base rates + signals)  
- 🔍 **OAT & robustness checks** to identify leverage points  
- 📉 **Brier scoring** for forecast accuracy evaluation  

---

## 🧠 Methodology Summary  

Forecast probabilities were estimated using:  
- Historical **base rates** (precedents, frequency analysis)  
- Updated **signals** (policy events, trade data, institutional actions)  
- Calibrated **interpretation scale** (consistent 0–100 mapping)  
- **Cromwell’s Rule** adherence (no extreme certainty)  

---
