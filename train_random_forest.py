import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=" * 50)
print("RANDOM FOREST")
print("=" * 50)

# Load datasets
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")
print(f"Features        : {X_train.shape[1]}")

# Create Random Forest
print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete!")

# Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Overrun",
            "Cost Overrun"
        ],
        zero_division=0
    )
)

# Feature importance
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 50)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 50)

print(
    importance.head(20).to_string(index=False)
)

# Save feature importance
importance.to_csv(
    "random_forest_feature_importance.csv",
    index=False
)

# Save predictions
predictions = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "risk_probability": y_probability
})

predictions.to_csv(
    "random_forest_predictions.csv",
    index=False
)

print("\nFiles created:")
print("  random_forest_feature_importance.csv")
print("  random_forest_predictions.csv")

print("\n" + "=" * 50)
print("DONE")
print("=" * 50)