import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.data import get_dataset_splits_from_h5
from sklearn.model_selection import KFold

# 1. Load data
data = np.load("data/tabular_pixelwise_data.npz")
X = data["X"]
y = data["y"]

# 2. Split into train_val and test
train_val_dataset, test_dataset = get_dataset_splits_from_h5("data/Dataset.h5", test_ratio=0.2)
train_val_idx = np.array(train_val_dataset.indices)
test_idx = np.array(test_dataset.indices)

X_train_val = X[train_val_idx]
y_train_val = y[train_val_idx]
X_test = X[test_idx]
y_test = y[test_idx]

# 3. Split train_val into train and validation
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_train_val, y_train_val, train_val_idx, test_size=0.2, random_state=42, stratify=y_train_val
)

# 4. HPO: test different parameter combinations
param_grid = [
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 50, "max_depth": 10},
    {"n_estimators": 150, "max_depth": 5},
    {"n_estimators": 100, "max_depth": None},
]

best_acc = 0
best_model = None
best_params = None

print("Hyperparameter search on validation set:")
for params in param_grid:
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    print(f"Params {params} -> Validation Accuracy: {acc_val:.4f}")

    if acc_val > best_acc:
        best_acc = acc_val
        best_model = clf
        best_params = params

# 5. Retrain on full train+val set with best parameters
X_train_full = X[train_val_idx]
y_train_full = y[train_val_idx]
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train_full, y_train_full)

# 6. Final evaluation on test set
y_test_pred = final_model.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)

print("\nBest Hyperparameters:", best_params)
print(f"Test Accuracy with Best Model: {acc_test:.4f}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for train_idx, val_idx in kf.split(train_val_idx):
    X_fold_train = X[train_val_idx[train_idx]]
    y_fold_train = y[train_val_idx[train_idx]]
    X_fold_val = X[train_val_idx[val_idx]]
    y_fold_val = y[train_val_idx[val_idx]]

    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_fold_train, y_fold_train)
    y_pred = clf.predict(X_fold_val)
    acc = accuracy_score(y_fold_val, y_pred)
    fold_accuracies.append(acc)

print(f"\nCross-Validation (5-fold) Accuracy: {np.mean(fold_accuracies):.4f}")