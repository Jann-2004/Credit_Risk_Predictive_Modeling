# eda.py  (Logistic Regression baseline with imputation)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_excel(r"D:\project 2\final.xlsx")

# 2. Select features and target
top_features = [
    "Income",
    "Credit_Score",
    "Credit_Utilization",
    "Missed_Payments",
    "Debt_to_Income_Ratio",
]
X = df[top_features]
y = df["Delinquent_Account"]

# 3. Train‑test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Build pipeline: Imputer ➜ Scaler ➜ Logistic Regression
pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),  # fills NaNs
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
            ),
        ),
    ]
)

# 5. Fit model
pipe.fit(X_train, y_train)

# 6. Evaluate
y_pred  = pipe.predict(X_test)
y_prob  = pipe.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("AUC‑ROC  :", round(roc_auc_score(y_test, y_prob), 3))

# 7. Plot ROC (optional)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("roc_logreg.png")
plt.show()
print("\nROC curve saved as roc_logreg.png")
