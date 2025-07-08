import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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

# === REMAP CLASSES TO CONSECUTIVE INTEGERS ===
all_classes = np.unique(y_train_val)
print("Original classes:", all_classes)

# Create mapping original_label -> consecutive_label
class_map = {old_label: new_label for new_label, old_label in enumerate(all_classes)}

def map_labels(y, mapping):
    y_mapped = np.copy(y)
    for old_label, new_label in mapping.items():
        y_mapped[y == old_label] = new_label
    return y_mapped

# Apply remapping to train+val and test labels
y_train_val = map_labels(y_train_val, class_map)
y_test = map_labels(y_test, class_map)

num_classes = len(all_classes)
print(f"Remapped classes: {np.unique(y_train_val)}")
print(f"Number of classes: {num_classes}")

# Normalize data (fit scaler only on train+val)
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_accuracies = []

print("Starting KFold cross-validation with XGBoost...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
    # Split fold data
    X_tr, X_va = X_train_val[train_idx], X_train_val[val_idx]
    y_tr, y_va = y_train_val[train_idx], y_train_val[val_idx]

    # Create XGBoost model with fixed params
    clf = XGBClassifier(
        device ='cuda',
        tree_method='hist',
        eval_metric='mlogloss',
        num_class=num_classes,
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    clf.fit(X_tr, y_tr)

    # Validate and compute accuracy
    y_va_pred = clf.predict(X_va)
    acc = accuracy_score(y_va, y_va_pred)
    fold_accuracies.append(acc)

    print(f"Fold {fold+1} accuracy: {acc:.4f}")

# Print average accuracy across folds
print(f"Mean CV accuracy: {np.mean(fold_accuracies):.4f}")

# Train final model on full train+val set
final_clf = XGBClassifier(
    device ='cuda',
    tree_method='hist',
    eval_metric='mlogloss',
    num_class=num_classes,
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
final_clf.fit(X_train_val, y_train_val)

# Test final model on test set
y_test_pred = final_clf.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {acc_test:.4f}")
