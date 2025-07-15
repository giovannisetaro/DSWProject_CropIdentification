import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from src.eval import evaluate

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
        tree_method='gpu_hist',
        eval_metric='mlogloss',
        num_class=num_classes,
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    clf.fit(X_tr, y_tr)

     # classes et probabilities prediction
    y_va_pred  = clf.predict(X_va)
    y_va_proba = clf.predict_proba(X_va)

    # compute log-loss mean and total_loss
    ll = log_loss(y_va, y_va_proba, labels=list(range(num_classes)))
    total_loss = ll * len(y_va)

    # evaluate
    print(f"\n[Fold {fold+1}] Evaluation:")
    cm_fold, metrics = evaluate(
        y_true=y_va,
        y_pred=y_va_pred,
        num_classes=num_classes,
        total_loss=total_loss,
        data_length=len(y_va),
        plot_cm=False
    )
    fold_accuracies.append(metrics["accuracy"])


# Print average accuracy across folds
print(f"Mean CV accuracy: {np.mean(fold_accuracies):.4f}")

# Train final model on full train+val set
final_clf = XGBClassifier(
    device ='cuda',
    tree_method='gpu_hist',
    eval_metric='mlogloss',
    num_class=num_classes,
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
final_clf.fit(X_train_val, y_train_val)

y_test_pred  = final_clf.predict(X_test)
y_test_proba = final_clf.predict_proba(X_test)

# log‐loss on test
ll_test    = log_loss(y_test, y_test_proba, labels=list(range(num_classes)))
total_loss = ll_test * len(y_test)

print("\n[Final Test Evaluation]")
cm_test, metrics_test = evaluate(
    y_true=y_test,
    y_pred=y_test_pred,
    num_classes=num_classes,
    total_loss=total_loss,
    data_length=len(y_test),
    plot_cm=True
)