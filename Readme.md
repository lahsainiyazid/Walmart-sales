***

```markdown
# 🛒 Walmart Sales Forecasting: Impact of Weather, Economy, and Holidays

## 📊 Project Overview
This project tackles the classic Kaggle **"Walmart Recruiting - Sales in Stormy Weather"** challenge. The objective is to predict the weekly sales of 45 Walmart stores located in different regions, taking into account the effects of holidays, weather conditions, fuel prices, and macroeconomic indicators (CPI and Unemployment).

Instead of just throwing data into a model, this project focuses heavily on **domain-driven feature engineering**, **strict time-series validation**, and **preventing data leakage** to build a highly reliable baseline model.

## 🏆 Key Results
* **Model:** Random Forest Regressor (Baseline)
* **Mean Absolute Error (MAE):** $40,771
* **Error Rate:** ~3.7% (Relative to an average weekly sales baseline of ~$1.1M per store)
* **Status:** Highly competitive baseline that outperforms standard naive models, providing a solid foundation for advanced ensemble tuning.

---

## 🧠 Feature Engineering Strategy
The core of this project's success lies in transforming raw data into predictive signals. Features were engineered across four distinct domains:

### 1. Temporal & Seasonality (Capturing the Calendar)
Retail sales are highly cyclical. Raw dates were decomposed to capture weekly, monthly, and yearly patterns.
* `Week_of_year`, `Month`, `Quarter`, `Day_Of_Month`
* **Payday Effects:** `Is_Start_Of_Month` (Days < 10) and `Is_End_Of_Month` (Days > 20) to capture the "halo effect" of consumer paychecks.

### 2. Specific Holiday Flags
A generic holiday flag is insufficient because different holidays drive completely different consumer behaviors.
* `Is_Super_Bowl`, `Is_Labor_Day`, `Is_Thanksgiving`, `Is_Christmas`

### 3. Macro-Economic Momentum
Raw economic numbers (CPI, Fuel Price) only show the *state* of the economy. To capture consumer *behavioral shifts*, we calculated the week-over-week momentum.
* `cpi_inflation_rate`: Percentage change in CPI.
* `fuel_inflation_rate`: Percentage change in Fuel Price (captures "shock" at the pump).
* `Consumer_Squeeze`: Interaction feature (`Unemployment * Fuel_Price`) to measure overall financial stress on the consumer.

### 4. Store Momentum & Baselines
* `Sales_Lag_1`: Previous week's sales (captures short-term momentum).
* `Rolling_window_4_weeks`: 4-week moving average (smooths out weekly noise to show the true baseline trend).

---

## 🛡️ Methodology & Preventing Data Leakage
A major focus of this project was ensuring the model's evaluation was 100% reflective of real-world performance.

### 1. Chronological Train/Test Split
**Rule:** Time-series data cannot be split randomly. 
The dataset was sorted by `Date` and split chronologically (80% Train / 20% Test). This ensures the model is trained on the past (2010-2011) and tested on the future (2012), completely preventing future data from leaking into the past.

### 2. The Target Leakage Trap (Caught & Fixed)
During initial feature engineering, a feature named `Wow_Sales_Growth` (Week-over-Week Growth) was created:
```python
# THE TRAP:
df['Wow_Sales_Growth'] = df['Weekly_Sales'] - df['Sales_Lag_1']
```
This resulted in a suspiciously low MAE of ~$9,500. Upon investigation, it was discovered that this feature mathematically contained the target variable (`Weekly_Sales`). 
* **The Fix:** The feature was dropped. To calculate true growth without leakage, one must use strictly historical data (e.g., `Sales_Lag_1 - Sales_Lag_2`).

### 3. Tree-Based Model Optimization
Because the baseline model is a **Random Forest**, unnecessary preprocessing steps were intentionally skipped:
* **No Standard Scaling:** Tree-based models split on feature thresholds (e.g., `Temperature > 50`), not mathematical distances. Scaling continuous variables adds computational overhead without improving accuracy.
* **No One-Hot Encoding for `Store`:** The `Store` ID was kept as a raw integer. Tree models can easily partition integer IDs without creating the massive sparse matrices associated with One-Hot Encoding.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (RandomForestRegressor)
* **Environment:** Linux (Fedora)

---

## 🚀 Future Improvements
To push the MAE below the $30,000 threshold, the following advanced techniques are planned:
1. **Algorithm Upgrade:** Transition from Random Forest to **XGBoost / LightGBM** to capture complex, non-linear interactions between economic and weather features.
2. **Target Encoding:** Implement leakage-free Target Encoding for the `Store` column (mapping Store IDs to their historical training mean) to give the model an immediate baseline context.
3. **Weighted MAE (WMAE):** Implement the official Kaggle evaluation metric, which penalizes errors more heavily on high-volume stores and during holiday weeks.
4. **Hyperparameter Tuning:** Use `RandomizedSearchCV` to optimize tree depth and split criteria.

---

## 📂 How to Run
1. Clone the repository and navigate to the directory.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Place the Kaggle dataset (`train.csv`, `test.csv`, `features.csv`, `stores.csv`) in the `/data` directory.
4. Run the main notebook or script:
   ```bash
   jupyter notebook walmart_sales_forecasting.ipynb
   ```

---
*Built with a focus on clean data pipelines, rigorous leakage prevention, and business-logic feature engineering.*
```
