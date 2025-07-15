import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.eval import evaluate
import joblib

# Load combined trainval dataset (already excludes label 225 pixels)
trainval_data = np.load("data/trainval_pixelwise.npz")

X_train_val = trainval_data["X"]
y_train_val = trainval_data["y"].astype(int)

print("TrainVal shape:", X_train_val.shape, y_train_val.shape)

# Remap class labels to consecutive integers
all_classes = np.unique(y_train_val)
class_map = {old_label: new_label for new_label, old_label in enumerate(all_classes)}

def map_labels(y, mapping):
    y_mapped = np.copy(y)
    for old_label, new_label in mapping.items():
        y_mapped[y == old_label] = new_label
    return y_mapped

y_train_val = map_labels(y_train_val, class_map)

num_classes = len(all_classes)
print(f"Number of classes: {num_classes}")

# Normalize features
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)

# Save the scaler
joblib.dump(scaler, "models/scaler.joblib")
print("Scaler saved to models/scaler.joblib")

# Hyperparameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10],
    "learning_rate": [0.01, 0.1]
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

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "models/xgb_best_model.joblib")
print("Best XGB model saved to models/xgb_best_model.joblib")
