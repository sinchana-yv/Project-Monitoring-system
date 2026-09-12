import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()


# --------------------------------------------------
# 2. Scale data
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 3. Train model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 4. Get feature coefficients
# --------------------------------------------------

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": coefficients,
    "absolute_importance": np.abs(coefficients)
})


# --------------------------------------------------
# 5. Sort by importance
# --------------------------------------------------

feature_importance = feature_importance.sort_values(
    "absolute_importance",
    ascending=False
)


# --------------------------------------------------
# 6. Display top 30
# --------------------------------------------------

print("\n======================================")
print("TOP FEATURE IMPORTANCE")
print("======================================")

print(
    feature_importance[
        [
            "feature",
            "coefficient",
            "absolute_importance"
        ]
    ].head(30).to_string(index=False)
)


# --------------------------------------------------
# 7. Separate positive and negative influences
# --------------------------------------------------

print("\n======================================")
print("FEATURES ASSOCIATED WITH HIGHER RISK")
print("======================================")

positive = feature_importance[
    feature_importance["coefficient"] > 0
].head(15)

print(
    positive[
        ["feature", "coefficient"]
    ].to_string(index=False)
)


print("\n======================================")
print("FEATURES ASSOCIATED WITH LOWER RISK")
print("======================================")

negative = feature_importance[
    feature_importance["coefficient"] < 0
].sort_values(
    "coefficient"
).head(15)

print(
    negative[
        ["feature", "coefficient"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 8. Save
# --------------------------------------------------

feature_importance.to_csv(
    "feature_importance_logistic.csv",
    index=False
)

print("\nSaved:")
print("feature_importance_logistic.csv")

print("\n======================================")
print("DONE")
print("======================================")