import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load datasets
train_data = np.load("data/train_pixelwise.npz")
val_data = np.load("data/val_pixelwise.npz")
test_data = np.load("data/test_pixelwise.npz")

X_train, y_train = train_data["X"], train_data["y"].astype(int)
X_val, y_val = val_data["X"], val_data["y"].astype(int)
X_test, y_test = test_data["X"], test_data["y"].astype(int)

# Print unique class labels before remapping
print("Unique classes before mapping:")
print("Train:", np.unique(y_train))
print("Val:", np.unique(y_val))
print("Test:", np.unique(y_test))

# Get all unique classes across train, val, and test
all_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
print("All classes found:", all_classes)

# Create mapping from original labels to consecutive integers starting at 0
class_map = {old_label: new_label for new_label, old_label in enumerate(all_classes)}
print("Class mapping:", class_map)

# Function to apply the mapping to label arrays
def map_labels(y, mapping):
    y_mapped = np.copy(y)
    for old_label, new_label in mapping.items():
        y_mapped[y == old_label] = new_label
    return y_mapped

# Apply the label mapping
y_train = map_labels(y_train, class_map)
y_val = map_labels(y_val, class_map)
y_test = map_labels(y_test, class_map)

# Confirm classes after mapping
print("Unique classes after mapping:")
print("Train:", np.unique(y_train))
print("Val:", np.unique(y_val))
print("Test:", np.unique(y_test))

# Define hyperparameter sets to try
param_grid = [
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15},
    {"n_estimators": 50,  "max_depth": 10},
    {"n_estimators": 150, "max_depth": 5},
    {"n_estimators": 100, "max_depth": None},
]

best_acc = 0
best_params = None

print("Starting hyperparameter search on validation set:")
num_classes = len(all_classes)  # number of unique classes

for params in param_grid:
    # Initialize model with GPU support and current params
    clf = XGBClassifier(
        tree_method='hist',           # recommended method with GPU support
        device='cuda',                # enable GPU usage
        eval_metric='mlogloss',       # multi-class log loss metric
        num_class=num_classes,
        **params
    )

    # Train on training data
    clf.fit(X_train, y_train)

    # Predict on validation set and compute accuracy
    y_val_pred = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)
    print(f"Params {params} -> Validation Accuracy: {acc_val:.4f}")

    # Update best parameters if improved
    if acc_val > best_acc:
        best_acc = acc_val
        best_params = params

# Combine train and validation sets for final training
X_train_val = np.concatenate([X_train, X_val], axis=0)
y_train_val = np.concatenate([y_train, y_val], axis=0)

# Train final model using best parameters on combined data
final_model = XGBClassifier(
    tree_method='hist',
    device='cuda',
    eval_metric='mlogloss',
    num_class=num_classes,
    **best_params
)
final_model.fit(X_train_val, y_train_val)

# Evaluate on the test set
y_test_pred = final_model.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {acc_test:.4f}")
