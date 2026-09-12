import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Load training and testing data
# --------------------------------------------------

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()


print("\n======================================")
print("LOGISTIC REGRESSION")
print("======================================")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])


# --------------------------------------------------
# 2. Scale numerical values
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 3. Create Logistic Regression model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)


# --------------------------------------------------
# 4. Train
# --------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training complete!")


# --------------------------------------------------
# 5. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# --------------------------------------------------
# 6. Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\n======================================")
print("MODEL RESULTS")
print("======================================")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# --------------------------------------------------
# 8. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 9. Detailed report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Overrun",
            "Cost Overrun"
        ]
    )
)


print("\n======================================")
print("BASELINE MODEL COMPLETE")
print("======================================")