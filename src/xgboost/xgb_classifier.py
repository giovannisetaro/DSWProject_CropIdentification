import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.eval import evaluate
import joblib

# Load separate datasets
train_data = np.load("data/train_pixelwise.npz")
val_data = np.load("data/val_pixelwise.npz")

X_train, y_train = train_data["X"], train_data["y"].astype(int)
X_val, y_val = val_data["X"], val_data["y"].astype(int)

# Remap class labels to consecutive integers based on train set only
all_classes = np.unique(y_train)
class_map = {old_label: new_label for new_label, old_label in enumerate(all_classes)}

def map_labels(y, mapping):
    y_mapped = np.copy(y)
    for old_label, new_label in mapping.items():
        y_mapped[y == old_label] = new_label
    return y_mapped

y_train = map_labels(y_train, class_map)
y_val = map_labels(y_val, class_map)

num_classes = len(all_classes)
print(f"Number of classes (train): {num_classes}")

# Normalize features using train set scaler, then apply to val 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
# Save the scaler
joblib.dump(scaler, "models/scaler.joblib")
print("Scaler saved to models/scaler.joblib")

# Combine train and val for training and hyperparameter tuning
X_train_val = np.concatenate([X_train, X_val], axis=0)
y_train_val = np.concatenate([y_train, y_val], axis=0)

max_depth = 6  # Set max depth for XGBoost trees

param_grid = {
    "n_estimators": [100, 200],
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
