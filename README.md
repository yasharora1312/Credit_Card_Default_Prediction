# 🧠 Building a Behaviour Score: Credit Card Default Prediction

This project predicts **credit card default** using machine learning techniques and explores **behavioral risk modeling**. The goal is to assist financial institutions in **early identification of potential defaulters**, improving risk management strategies.

> 🔍 *Project from Finance Club Open Project — Summer 2025*

## 👨‍💻 Author
**Yash Arora**  
Roll No: 22112117

---

## 📌 Problem Statement

To build a classification model that predicts whether a customer will default on their credit card payment next month using historical and behavioral data.

---

## 📊 Project Workflow

### 1. 🔎 Exploratory Data Analysis (EDA)
- Identified right-skewed distributions in bill amounts, payments, and credit limits.
- Key insights from correlation heatmap:
  - High correlation between `pay_X`, `bill_amtX`, and `pay_amtX` columns → PCA applied.
  - `pay_0` is the strongest predictor of default.
  - `LIMIT_BAL` and `Age` show slight negative correlation with default.

### 2. 🏗 Feature Engineering
- Created features like average bill amount.
- Scaled numerical features using **StandardScaler** and **MinMaxScaler**.

### 3. ⚖️ Handling Class Imbalance
- Used **SMOTE** to oversample the minority class (defaulters).
- Increased **recall** while reducing bias toward majority class.

### 4. 🔧 Dimensionality Reduction
- Applied **PCA** to handle multicollinearity and optimize model performance.

---

## 🤖 Models Trained & Evaluated

| Model              | F1 Score (Defaulters) | F2 Score | Recall |
|--------------------|-----------------------|----------|--------|
| Logistic Regression| 0.43                  | 0.538    | 0.64   |
| Decision Tree      | 0.38                  | 0.546    | 0.457  |
| KNN                | 0.40                  | 0.520    | 0.59   |
| Random Forest      | 0.506                 | 0.531    | 0.42   |
| XGBoost            | 0.52                  | 0.550    | 0.596  |
| **AdaBoost**       | **0.52**              | **0.550**| **0.596** |

✅ **Best Model: AdaBoost** (based on highest F1 & F2 score)

---

## 🛠 Optimization & Tuning

- **GridSearchCV** used for hyperparameter tuning.
- Optimized:
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
  - XGBoost: `learning_rate`, `scale_pos_weight`
- Tuned classification threshold based on **precision-recall** trade-off.

---

## 📈 Metrics

- **Accuracy** (not prioritized due to class imbalance)
- **Recall** (important to catch all defaulters)
- **F1 Score** (balance between precision & recall)
- **F2 Score** (favoring recall — crucial in financial risk)

---

## 🏦 Business Implications

- 🛡 **Risk Management**: Early detection of defaulters minimizes losses.
- 🧠 **Actionable Insights**: Payment patterns and overdue status guide credit decisions.
- 🎯 **Targeted Strategies**: Personalized monitoring or intervention plans for high-risk customers.

---

## 📁 Project Structure

```bash
Main.ipynb       # Notebook with EDA, models, tuning, and evaluation
Report.pdf       # Formal write-up with charts, results, and model rationale
