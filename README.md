<div align="center">

```
███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗███╗   ███╗ █████╗ ██╗  ██╗
██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║████╗ ████║██╔══██╗╚██╗██╔╝
███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║██╔████╔██║███████║ ╚███╔╝ 
╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║ ██╔██╗ 
███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║██╔╝ ██╗
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
```

**Predicting viewer disengagement before they even know they're leaving.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.791+-success?style=flat-square)](/)
[![Models](https://img.shields.io/badge/Ensemble-CatBoost%20%2B%20XGB%20%2B%20LGB-orange?style=flat-square)](/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

</div>

---

## What This Is

Every streaming platform hemorrhages subscribers quietly. Not in dramatic cancellations, but in the slow drift — sessions getting shorter, genres narrowing, the app opening less and less until one day the subscription just... lapses.

By then, it's too late. The user is mentally gone.

This project builds a system that catches that drift *early* — predicting "engagement fatigue" 30 days before it becomes churn. We call these users **fatigued**, not churned, because that distinction matters: intervening at the fatigue stage is exponentially cheaper and more effective than re-acquiring a lost subscriber.

The model was built on behavioral data from 10,000 users of a streaming platform — watch velocity, binge patterns, genre breadth, session decay — and achieves **AUC-ROC > 0.79** using a weighted ensemble of gradient boosting models with domain-informed monotone constraints.

It works on any subscription streaming service. The behavioral patterns of disengagement are the same whether you're watching Bollywood blockbusters on Hotstar or prestige drama on any western platform.

---

## The Problem, Precisely

```
Standard churn detection:
  User stops paying → Platform notices → Platform tries to win them back
  
  Cost to re-acquire: ₹800–2000 per user
  Success rate: ~15%

This system:
  User's watch time drops 30% week-over-week → Model flags them → Intervention triggers
  
  Cost to retain: ₹80–200 per user  
  Success rate: ~60%
```

The math is obvious. The hard part is knowing *who* to target and *when*. That's what this solves.

---

## Architecture

```
Raw Behavioral Data (13 features)
         │
         ▼
┌─────────────────────────────────┐
│      Feature Engineering        │
│  • Watch Velocity (7d vs 30d)   │
│  • Session Decay Index          │
│  • Binge Exhaustion Score       │
│  • Genre Entropy                │
│  • Content-Engagement Ratio     │
│  • Quality×Quantity composite   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  Stratified 5-Fold CV                    │
│                                                          │
│   CatBoost (60%)  ──┐                                   │
│   XGBoost  (30%)  ──┼──► Weighted Blend ──► Probabilities│
│   LightGBM (10%)  ──┘                                   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Pseudo-Label Augmentation   │
│  High-confidence test samples   │
│  fed back for CatBoost refit    │
└─────────────────────────────────┘
         │
         ▼
    Final Ensemble Predictions
    AUC-ROC: 0.791+
```

---

## Key Features Engineered

| Feature | What it captures |
|---|---|
| `watch_velocity` | Rate of decline: 7-day avg vs 30-day avg, normalized |
| `session_velocity` | Session frequency drop vs expected baseline |
| `binge_exhaustion_index` | High completion + falling watch time = content exhaustion |
| `engagement_score` | Composite of recency, frequency, depth |
| `quality_quantity_ratio` | Minutes × completion rate — quality-adjusted watch time |
| `tier_target_enc` | Out-of-fold target encoding of subscription tier |

The model enforces **monotone constraints** on features where directionality is theoretically guaranteed — more days since last session always increases fatigue probability, for example. This prevents spurious reversals that would undermine trust in production.

---

## Results

```
LightGBM  OOF AUC : 0.76xxx  (weight = 0.10)
XGBoost   OOF AUC : 0.78xxx  (weight = 0.30)
CatBoost  OOF AUC : 0.79xxx  (weight = 0.60)
──────────────────────────────────
Final Blend  AUC  : 0.791+
```

Of 2,000 held-out users, approximately 1,000 were flagged as at-risk. The model's probability outputs span from 0.06 (deeply engaged) to 0.94 (severely fatigued), making it suitable for tiered intervention strategies rather than binary targeting.

---

## Repository Structure

```
streammax-fatigue-prediction/
│
├── Gyanvardhak_Analysis.ipynb    # Full pipeline: EDA → features → ensemble → predictions
├── Gyanvardhak_Predictions.csv   # Final predictions (2000 users, probabilities)
│
├── data/
│   ├── ott_train.csv             # 8,000 users with fatigue_label
│   └── ott_test.csv              # 2,000 users, no label (prediction target)
│
├── simulation/
│   └── index.html                # Interactive StreamMax-style dashboard
│
└── README.md
```

---

## Running It

**Colab** (recommended — everything runs in-browser):

```
File → Open notebook → GitHub → paste this repo URL
```

**Local**:

```bash
git clone https://github.com/tanx1509/streammax-fatigue-prediction
cd streammax-fatigue-prediction
pip install pandas numpy scikit-learn lightgbm catboost xgboost
jupyter notebook Gyanvardhak_Analysis.ipynb
```

The notebook is self-contained. Run cells top to bottom. Final predictions export automatically to `Gyanvardhak_Predictions.csv`.

---

## Business Interpretation

Three user archetypes emerged from the data:

**Ghost Users** — High tenure, near-zero recent activity. Days since last session: 20+. These users have already mentally churned. Retention cost is high; prioritize understanding *why*.

**Decelerating Bingers** — Recent high engagement that's dropped sharply. High binge history, falling 7-day numbers. These are the highest-value intervention targets. They loved the platform; something specific broke the habit.

**Narrow Consumers** — Active but locked into 1-2 genres. Low genre diversity score. Not immediately at risk, but vulnerable to a content gap. Recommendation diversification is the play here.

---

## What Streaming Platforms Can Do With This

1. **Tiered alerts** — Different interventions for 0.5–0.65 (nudge), 0.65–0.8 (offer), 0.8+ (direct outreach)
2. **Content gap detection** — If high-fatigue users cluster around specific genre exhaustion, that's a commissioning signal
3. **Recommendation re-weighting** — For flagged users, temporarily bias recommendations toward genres they haven't explored
4. **Re-engagement timing** — Model output tells you the 30-day window. Intervene at day 10-15, not day 28.

---

## Tech Stack

- **Python 3.10+**
- **CatBoost 1.2** — Primary model, handles categoricals natively, most stable OOF performance
- **XGBoost** — Second model, strong on interaction features
- **LightGBM** — Fast baseline, useful for feature selection
- **scikit-learn** — Cross-validation, permutation importance
- **pandas / numpy** — Data manipulation
- **Google Colab** — Training environment

---

<div align="center">

*Built by Team Gyanvardhak, MAIT Delhi*

</div>
