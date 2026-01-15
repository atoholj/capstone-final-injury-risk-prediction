# Injury Risk Prediction from Athlete Monitoring Signals (Berkeley ML/AI Capstone)

## Goal
The goal of this project is to predict **short-term musculoskeletal injury risk** using athlete monitoring data such as training load, movement/biomechanics, and recovery signals. A strong model can support an **early warning system** that helps coaches and athletes adjust training to reduce time-loss injuries.

---

## Data Problem
This is a **binary classification** task:

- **Target:** `injury_risk`
  - `0` = no injury risk
  - `1` = injury risk
- Injury events are **rare (~5%)**, so **accuracy can be misleading**.  
For that reason, we prioritize **PR-AUC (Average Precision)** and **recall**, which better measure performance on rare injury-risk cases.

---

## Dataset
**Source (Kaggle):** Multimodal Sports Injury Prediction Dataset  
https://www.kaggle.com/datasets/anjalibhegam/multimodal-sports-injury-prediction-dataset

**Size:** 5,430 rows × 31 columns  
**Type:** Multimodal athlete monitoring signals, including:

- **Physiology / recovery:** heart rate, SpO2, blood pressure, respiratory rate, skin temperature, GSR  
- **Movement / biomechanics:** ground reaction force, impact force, gait symmetry, cadence, range of motion  
- **Training context:** training duration, workload intensity, rest period, repetition count  
- **Label:** `injury_risk`

---

## Repository Structure
This project is organized into three notebooks:

- `01_EDA_and_Data_Checks.ipynb`  
  Data loading, missing values, duplicates, summary statistics, and EDA visualizations.

- `02_Modeling_and_Model_Selection.ipynb`  
  Baseline models, cross-validation, GridSearchCV tuning, and model selection using PR-AUC.

- `03_Final_Model_Interpretation.ipynb`  
  Final model evaluation, feature importance, interpretation, and recommendations for a real-world early warning workflow.

---

## Exploratory Data Analysis (EDA)
Key EDA observations:

- The dataset has **no missing values** and **no duplicate rows**, making it clean and ready for modeling.
- Injury-risk is **class-imbalanced (~5%)**, so we focus on metrics that reflect performance on rare injury cases.
- Correlation analysis showed that the dataset contains many weak-to-moderate relationships, suggesting injury risk is likely driven by **nonlinear interactions** across multiple signals.

---

## Modeling Approach
### Baseline
A Logistic Regression baseline model was created to establish a simple reference point.

### Model Selection
We trained and tuned multiple models and selected a final model using:

- **Stratified train/test split**
- **Stratified K-Fold Cross-Validation**
- **GridSearchCV** hyperparameter tuning
- **PR-AUC (Average Precision)** as the primary selection metric

---

## Final Model Results
The best performing model was a **Random Forest Classifier** tuned using GridSearchCV.

### Held-Out Test Set Performance
- **Test PR-AUC (Average Precision): `0.9859`**
- **Test ROC-AUC: `0.9990`**

### Why this matters
- **PR-AUC** is the most important metric here because injury cases are rare.  
A PR-AUC of **0.9859** indicates the model is extremely strong at identifying injury-risk cases while controlling false alarms.
- **ROC-AUC** of **0.9990** shows the model separates injury-risk vs non-risk classes almost perfectly across thresholds.

 **Conclusion:** The final model demonstrates excellent predictive power and would be highly effective as an injury early warning tool.

---

## Model Interpretation (Feature Importance)
Random Forest feature importance highlights which signals most influence injury-risk predictions.  
The most important predictors generally fall into these categories:

- **Biomechanical load** (forces, acceleration, impact)
- **Physiological stress** (heart rate, respiration, recovery-related signals)
- **Training load and fatigue** (duration, intensity, fatigue index)
- **Previous injury history** (risk indicator)

This matches real-world expectations: injury risk tends to rise when high workload combines with fatigue and increased mechanical stress.

---

## Actionable Takeaways (Nontechnical)
This model can support coaches by flagging elevated injury risk early, enabling proactive adjustments such as:

- reducing intensity for 24–48 hours (“deload”)
- replacing impact-heavy sessions with mobility or low-impact recovery
- increasing rest periods and recovery time
- monitoring repeated high-risk alerts across multiple sessions

The model is designed to support **prevention**, not diagnosis.

---

## Recommendations / Next Steps
To make this system more realistic and deployable:

1. Add time-series features (rolling load trends, acute-to-chronic workload ratios)
2. Tune classification thresholds depending on coaching goals:
   - higher recall = fewer missed injury cases
   - higher precision = fewer false alarms
3. Try gradient boosting models (XGBoost / LightGBM) for comparison
4. Use SHAP for more detailed interpretability
5. Validate on real athlete tracking data (per-athlete longitudinal records)

---

## Requirements
Main libraries used:
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`

---
