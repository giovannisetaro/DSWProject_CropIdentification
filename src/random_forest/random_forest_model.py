import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from src.eval import evaluate
import xgboost as xgb

# Load datasets
train_data = np.load("data/train_pixelwise.npz")
val_data = np.load("data/val_pixelwise.npz")
test_data = np.load("data/test_pixelwise.npz")

X_train, y_train = train_data["X"], train_data["y"].astype(int)
X_val, y_val = val_data["X"], val_data["y"].astype(int)
X_test, y_test = test_data["X"], test_data["y"].astype(int)

# Combine train and val for cross-validation
X_train_val = np.concatenate([X_train, X_val], axis=0)
y_train_val = np.concatenate([y_train, y_val], axis=0)

print("Train+Val shape:", X_train_val.shape, y_train_val.shape)

# Remap class labels to consecutive integers
all_classes = np.unique(y_train_val)
class_map = {old_label: new_label for new_label, old_label in enumerate(all_classes)}

def map_labels(y, mapping):
    y_mapped = np.copy(y)
    for old_label, new_label in mapping.items():
        y_mapped[y == old_label] = new_label
    return y_mapped

y_train_val = map_labels(y_train_val, class_map)
y_test = map_labels(y_test, class_map)

num_classes = len(all_classes)
print(f"Number of classes: {num_classes}")

# Normalize features
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

# Hyperparameter grid
param_grid = {
    "n_estimators": [100,200],
    "max_depth": [10,6],
    "learning_rate": [0.01,0.1]
}

print("Starting hyperparameter optimization...")

clf = XGBClassifier(
    tree_method='hist',
    device='cuda',
    num_class=num_classes,
    objective="multi:softprob",
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_val, y_train_val)

print("\nBest hyperparameters found:")
print(grid_search.best_params_)
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_

import joblib

# Save the best model
joblib.dump(best_model, "models/xgb_best_model.joblib")

# Save the scaler
joblib.dump(scaler, "models/scaler.joblib")

y_test_pred = best_model.predict(X_test)

class_names = [str(i) for i in range(num_classes)]

print("\n[Final Test Evaluation]")
_, metrics_test = evaluate(
    y_true=y_test,
    y_pred=y_test_pred,
    num_classes=num_classes,
    class_names=class_names,
    plot_cm=True
)
