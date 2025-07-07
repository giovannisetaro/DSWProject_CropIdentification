import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from src.data import get_dataset_splits_from_h5

# Load tabular features and labels extracted earlier
data = np.load("data/tabular_pixelwise_data.npz")
X_tabular = data["X"]  # shape: [num_samples, features_dim]
y_tabular = data["y"]  # shape: [num_samples]

# Use your existing function to split dataset into train+val and test subsets
train_val_dataset, test_dataset = get_dataset_splits_from_h5("data/Dataset.h5", test_ratio=0.2)

# Get indices corresponding to train+val and test sets
train_val_indices = train_val_dataset.indices
test_indices = test_dataset.indices

# Initialize K-Fold cross-validation on train+val set indices
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
    print(f"Fold {fold + 1}")

    # Map fold indices to global dataset indices
    train_global_idx = np.array(train_val_indices)[train_idx]
    val_global_idx = np.array(train_val_indices)[val_idx]

    # Extract feature vectors and labels for train and validation fold subsets
    X_train_fold = X_tabular[train_global_idx]
    y_train_fold = y_tabular[train_global_idx]
    X_val_fold = X_tabular[val_global_idx]
    y_val_fold = y_tabular[val_global_idx]

    # Initialize and train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_fold, y_train_fold)

    # Predict on validation fold and compute accuracy
    y_val_pred = clf.predict(X_val_fold)
    acc_val = accuracy_score(y_val_fold, y_val_pred)
    print(f"Validation Accuracy Fold {fold + 1}: {acc_val:.4f}")

# After cross-validation, evaluate on the test set
X_test = X_tabular[test_indices]
y_test = y_tabular[test_indices]
y_test_pred = clf.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {acc_test:.4f}")
