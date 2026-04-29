```text
███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗███╗   ███╗ █████╗ ██╗  ██╗
██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║████╗ ████║██╔══██╗╚██╗██╔╝
███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║██╔████╔██║███████║ ╚███╔╝ 
╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║ ██╔██╗ 
███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║██╔╝ ██╗
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
```
```text
 ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗██╗   ██╗ █████╗ ██████╗ ██████╗ ██╗  ██╗ █████╗ ██╗  ██╗
██╔════╝ ╚██╗ ██╔╝██╔══██╗████╗  ██║██║   ██║██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔══██╗██║ ██╔╝
██║  ███╗ ╚████╔╝ ███████║██╔██╗ ██║██║   ██║███████║██████╔╝██║  ██║███████║███████║█████╔╝ 
██║   ██║  ╚██╔╝  ██╔══██║██║╚██╗██║╚██╗ ██╔╝██╔══██║██╔══██╗██║  ██║██╔══██║██╔══██║██╔═██╗ 
╚██████╔╝   ██║   ██║  ██║██║ ╚████║ ╚████╔╝ ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
```

**OTT Subscriber Churn Prediction — Top 5 of 1,000+ Teams | XLRI StrategiX 2.0**

*Predicting engagement fatigue 30 days before it becomes churn.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![AUC-ROC](https://img.shields.io/badge/Final_AUC-0.7917-success?style=flat-square)](/)
[![Improvement](https://img.shields.io/badge/Improvement-+0.0717_over_baseline-blue?style=flat-square)](/)
[![Models](https://img.shields.io/badge/Ensemble-CatBoost_0.6_×_XGB_0.3_×_LGB_0.1-orange?style=flat-square)](/)
[![Competition](https://img.shields.io/badge/XLRI_StrategiX_2.0-Top_5_of_1000%2B-gold?style=flat-square)](/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

---

## The Problem, Precisely

StreamMax was not losing subscribers in a single moment. It was losing them in **slow motion — over 30 days** — while continuing to charge them.

```text
Standard reactive churn detection:
  User stops paying → Platform notices → Platform tries to win them back
  Cost to re-acquire : ₹800–2,000 per user
  Success rate       : ~15%

This system:
  Watch time drops 30% week-over-week → Model flags user → Intervention triggers
  Cost to retain     : ₹80–200 per user
  Success rate       : ~60%
```

The math is obvious. The hard part is knowing *who* to target and *when*. This project solves exactly that.

---

## Headline Numbers

| Metric | Value |
|---|---|
| Final OOF ROC-AUC | **0.7917** |
| Baseline AUC (Logistic Regression) | 0.7200 |
| Total improvement | **+0.0717** |
| Features engineered | **88** (from 13 raw, +576%) |
| Models trained across all stages | **12** |
| CV strategy | Stratified 10-Fold throughout |
| Pseudo-label samples added | **194** (high-confidence augmentation) |
| Users at fatigue risk | **35.4%** of 8,000-user train set |
| AUC ceiling (from max correlation 0.30) | ~0.81 — **0.7917 is near-optimal** |

---

## Technical Architecture

```text
Raw Behavioral Data (13 features)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Nuclear Feature Engineering                │
│  • watch_velocity      (7d vs 30d normalized decline)  │
│  • session_velocity    (recent vs expected baseline)    │
│  • binge_exhaustion_index (completion × falling watch)  │
│  • recency_log / recency_squared (monotone penalty)     │
│  • engagement_score    (composite: time+sessions+compl) │
│  • recommendation_failure (algo failing the user)       │
│  • premium_disengagement (tier × (1 - engagement))      │
│  • genre_diversity_pct / genre_x_completion             │
│  ... 88 total                                           │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│         Permutation Importance Feature Selection        │
│   Kill features contributing < 0.001 AUC               │
│   Uses GBM selector, 10 repeats — no noise survives     │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│          Power-Weighted Ensemble (5-Fold Strat CV)      │
│                                                         │
│  CatBoost v2  (60%) ──┐                                 │
│  XGBoost  v2  (30%) ──┼──► Weighted blend ──► OOF probs│
│  LightGBM v2  (10%) ──┘                                 │
│                                                         │
│  Monotonic constraints enforced on 12 features:         │
│  days_since_last_session → always increases fatigue     │
│  engagement_score        → always decreases fatigue     │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│        Meta-Stacking (LR on OOF + top 20 raw feats)    │
│   Learns which specialist is right when they disagree   │
│   Meta-stack OOF AUC: 0.7911                            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│          Semi-Supervised Pseudo-Labeling                │
│   194 test samples with prob > 0.88 or < 0.12           │
│   Added to training set → CatBoost + LGB retrained      │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│           Rank Calibration (Sigmoid on Ranks)           │
│   rank_calibrate(probs, k=5.5)                          │
│   Converts raw probabilities to well-calibrated scores  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
    Final Predictions — AUC 0.7917
    Gyanvardhak_Predictions.csv (2,000 users)
```

---

## Complete Model Progression

Every stage logged. Every AUC documented. No cherry-picking.

| Stage | Model | OOF AUC | Status |
|---|---|---|---|
| 1 | Logistic Regression | 0.7200 | REJECTED — baseline only |
| 2 | Random Forest | 0.7810 | Baseline |
| 3 | Extra Trees | 0.7791 | Baseline |
| 4 | XGBoost v1 | 0.7889 | Layer 1 |
| 5 | XGBoost v2 | 0.7860 | Layer 1 |
| 6 | LightGBM v1 | 0.7871 | Layer 1 |
| 7 | LightGBM v2 | 0.7812 | Layer 1 |
| 8 | CatBoost v1 | 0.7902 | Layer 1 |
| 9 | **CatBoost v2** | **0.7904** | Layer 1 champion |
| 10 | Meta-Stack (LR on OOF) | 0.7911 | Champion v1 |
| 11 | + Pseudo-Labeling | 0.7917 | Champion v2 |
| 12 | **Final (10-Fold + SWA)** | **0.7917** | **FINAL SUBMISSION** |

Total gain over baseline: **+0.0717 AUC.**

---

## Key Features Engineered

| Feature | Formula / Logic | Why it works |
|---|---|---|
| `watch_velocity` | `(7d_avg − 30d_avg) / 30d_avg` | Rate of decline, normalized |
| `session_velocity` | `(actual_7d − expected_7d) / expected_7d` | Acceleration of disengagement |
| `binge_exhaustion_index` | `completion_rate × (1 − 7d/30d ratio)` | High completion + falling time = content exhausted |
| `recency_log` | `log1p(days_since_last_session)` | Log-transform amplifies small gaps; monotone constraint enforced |
| `engagement_score` | `0.35×time + 0.25×sessions + 0.20×completion + 0.20×rec_click` | Composite recency-frequency-depth score |
| `recommendation_failure` | `(1 − rec_click_rate) × (1 − 7d_minutes/300)` | Algorithm failing user = churn accelerator |
| `premium_disengagement` | `tier_encoded × (1 − engagement_score)` | Premium + low engagement = highest churn cost |
| `genre_diversity_pct` | `unique_genres / 15.0` | Narrow consumer = content gap vulnerability |

**Monotonic constraints** enforced on 12 features — prevents spurious reversals that would undermine trust in production. `days_since_last_session` must always increase fatigue probability. No exceptions.

---

## Three User Archetypes

```text
┌─────────────────────────────────────────────────────────────────┐
│  GHOST USERS          (fatigue prob > 0.80)                     │
│  Tenure: high. Recent activity: near-zero. Days since login 20+ │
│  These users have already mentally churned.                     │
│  Action: survey why — do not waste high-value retention offers  │
├─────────────────────────────────────────────────────────────────┤
│  DECELERATING BINGERS (fatigue prob 0.50–0.80)  ← GOLD MINE    │
│  Recent high engagement that has dropped sharply.               │
│  They loved the platform. Something specific broke the habit.   │
│  Action: personalized content digest + recommendation override  │
├─────────────────────────────────────────────────────────────────┤
│  NARROW CONSUMERS     (fatigue prob 0.35–0.50)                  │
│  Active but locked into 1–2 genres. Low diversity score.        │
│  Not immediately at risk. Vulnerable to content gaps.           │
│  Action: recommendation diversification — proactive, not rescue │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Recommendation Algorithm Death Spiral

This is the systemic insight the model uncovered. When `rec_click_rate` drops, the algorithm interprets it as "user dislikes content" and serves increasingly niche material — which accelerates disengagement further:

```text
Low engagement
    → Algorithm shows niche content
        → Even lower engagement
            → User mentally cancels
                → Cancellation button is a formality
```

**Fix:** Override the recommendation algorithm for users whose `rec_click_rate` has dropped >30% — force proven popular content until engagement stabilises. The model flags exactly who needs this override.

---

## Prediction Output

`Gyanvardhak_Predictions.csv` — calibrated fatigue probabilities for all 2,000 test users.

```
user_id, predicted_fatigue_probability
U000400, 0.094   ← Low risk   (< 0.30)
U006407, 0.657   ← Medium risk (0.30–0.70)
U007248, 0.853   ← High risk  (> 0.70)
```

Final distribution (2,000 users):
- High risk > 0.70: **~530 users**
- Medium risk 0.30–0.70: **~940 users**
- Low risk < 0.30: **~530 users**

---

## Running It

**Colab** (zero setup):
```text
File → Open notebook → GitHub → paste this repo URL
Open Gyanvardhak_Analysis.ipynb → Runtime → Run all
```

**Local:**
```bash
git clone https://github.com/tanx1509/gyanvardhak-streammax
cd gyanvardhak-streammax
pip install pandas numpy scikit-learn lightgbm catboost xgboost matplotlib seaborn
jupyter notebook Gyanvardhak_Analysis.ipynb
```

Run cells top to bottom. Final predictions export automatically to `Gyanvardhak_Predictions.csv`.

---

## Repository Structure

```text
gyanvardhak-streammax/
│
├── Gyanvardhak_Analysis.ipynb      # Full pipeline: EDA → features → ensemble → predictions
├── Gyanvardhak_Predictions.csv     # Final calibrated predictions (2,000 users)
│
├── data/
│   ├── ott_train.csv               # 8,000 users with fatigue_label
│   └── ott_test.csv                # 2,000 users, no label
│
├── assets/
│   ├── eda_distributions.png       # Engaged vs Fatigued feature distributions
│   ├── feature_importance.png      # Nuclear feature ranking with AUC threshold
│   └── viz_summary_progression.png # Full model progression chart
│
└── README.md
```

---

## Stack

Python 3.10 · Pandas · NumPy · Scikit-learn · LightGBM · CatBoost · XGBoost · Matplotlib · Seaborn

---

*Team Gyanvardhak — Tanishq Sethi & Vaibhav Mathpal*
*Maharaja Agrasen Institute of Technology, Delhi*
*XLRI StrategiX 2.0 | Top 5 of 1,000+ Teams*
