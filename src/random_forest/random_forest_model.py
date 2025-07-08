import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load data
train_data = np.load("data/train_pixelwise.npz")
val_data = np.load("data/val_pixelwise.npz")
test_data = np.load("data/test_pixelwise.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val = val_data["X"], val_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

# Hyperparameter grid
param_grid = [
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 50,  "max_depth": 10},
    {"n_estimators": 150, "max_depth": 5},
    {"n_estimators": 100, "max_depth": None},
]

best_acc = 0
best_params = None

print("Hyperparameter search on validation set:")
for params in param_grid:
    clf = XGBClassifier(
        tree_method='gpu_hist',  # Use GPU
        predictor='gpu_predictor',
        use_label_encoder=False,
        eval_metric='mlogloss',
        **params
    )
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    print(f"Params {params} -> Validation Accuracy: {acc_val:.4f}")

    if acc_val > best_acc:
        best_acc = acc_val
        best_params = params

# Retrain final model on train + val
X_train_val = np.concatenate([X_train, X_val], axis=0)
y_train_val = np.concatenate([y_train, y_val], axis=0)
final_model = XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    use_label_encoder=False,
    eval_metric='mlogloss',
    **best_params
)
final_model.fit(X_train_val, y_train_val)

# Final test
y_test_pred = final_model.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)

print("\nBest Hyperparameters:", best_params)
print(f"Test Accuracy with Best Model: {acc_test:.4f}")
