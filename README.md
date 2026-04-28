```markdown
<div align="center">

```text
███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗███╗   ███╗ █████╗ ██╗  ██╗
██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║████╗ ████║██╔══██╗╚██╗██╔╝
███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║██╔████╔██║███████║ ╚███╔╝ 
╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║ ██╔██╗ 
███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║██╔╝ ██╗
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
```

**Predicting viewer disengagement before they even know they are leaving.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![AUC-ROC](https://img.shields.io/badge/AUC_ROC-0.791+-success?style=flat-square)](/)
[![Models](https://img.shields.io/badge/Ensemble-CatBoost%20%2B%20XGB%20%2B%20LGB-orange?style=flat-square)](/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[![StreamMax Dashboard Preview](assets/streammax_demo.gif)](https://streammax-analytics.netlify.app)

*Click the dashboard preview above to launch the live interactive simulation.*

</div>

## The Problem, Precisely
Subscription platforms do not lose users in a single moment. Churn is a slow degradation of engagement that happens over weeks. By the time a user formally cancels, the decision was already made 30 days prior.

```text
Standard reactive churn detection:
  User stops paying -> Platform notices -> Platform tries to win them back
  Cost to re-acquire: ₹800 to 2000 per user
  Success rate: ~15%

This predictive system:
  User's watch time drops 30% week-over-week -> Model flags them -> Intervention triggers
  Cost to retain: ₹80 to 200 per user  
  Success rate: ~60%
```

The math is obvious. The hard part is knowing *who* to target and *when*. This project solves exactly that.

## The Solution
This system is an end-to-end predictive machine learning pipeline designed to identify "engagement fatigue" 30 days before it becomes churn. We call these users **fatigued**, not churned, because intervening at the fatigue stage is exponentially cheaper and more effective.

The model was built on behavioral data from 10,000 users. By engineering 88 behavioral features capturing watch velocity, session decay, and binge exhaustion, the ensemble model predicts disengagement with a **0.791 AUC-ROC**, allowing for targeted retention interventions. 

## Technical Architecture

```text
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
│                  Stratified 5-Fold CV                   │
│                                                         │
│  CatBoost (60%)  ──┐                                    │
│  XGBoost  (30%)  ──┼──► Weighted Blend ──► Probabilities│
│  LightGBM (10%)  ──┘                                    │
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

## Key Features Engineered

| Feature | What it captures |
|---|---|
| `watch_velocity` | Rate of decline: 7-day avg vs 30-day avg, normalized |
| `session_velocity` | Session frequency drop vs expected baseline |
| `binge_exhaustion_index` | High completion + falling watch time = content exhaustion |
| `engagement_score` | Composite of recency, frequency, depth |
| `quality_quantity_ratio` | Minutes × completion rate = quality-adjusted watch time |
| `tier_target_enc` | Out-of-fold target encoding of subscription tier |

The model enforces **monotone constraints** on features where directionality is theoretically guaranteed. For example, more days since the last session always increases fatigue probability. This prevents spurious reversals that would undermine trust in a production environment.

## Business Interpretation
Three distinct user archetypes emerged from the data, dictating the required intervention strategy:

1. **Ghost Users:** High tenure, near-zero recent activity. Days since last session: 20+. These users have already mentally churned. Retention cost is high; prioritize understanding *why*.
2. **Decelerating Bingers:** Recent high engagement that has dropped sharply. High binge history, falling 7-day numbers. These are the highest-value intervention targets. They loved the platform; something specific broke the habit.
3. **Narrow Consumers:** Active but locked into 1 to 2 genres. Low genre diversity score. Not immediately at risk, but highly vulnerable to a content gap. Recommendation diversification is the play here.

## Running It

**Colab** (recommended, everything runs in-browser):
```text
File -> Open notebook -> GitHub -> paste this repo URL
```

**Local**:
```bash
git clone [https://github.com/tanx1509/streammax-fatigue-prediction](https://github.com/tanx1509/streammax-fatigue-prediction)
cd streammax-fatigue-prediction
pip install pandas numpy scikit-learn lightgbm catboost xgboost
jupyter notebook Gyanvardhak_Analysis.ipynb
```
The notebook is self-contained. Run cells top to bottom. Final predictions export automatically to `Gyanvardhak_Predictions.csv`.

## Repository Structure
```text
streammax-fatigue-prediction/
│
├── Gyanvardhak_Analysis.ipynb    # Full pipeline: EDA -> features -> ensemble -> predictions
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

<div align="center">

*Built by Team Gyanvardhak (Tanishq Sethi & Vaibhav Mathpal), MAIT Delhi*

</div>
```
