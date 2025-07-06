import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load tabular data from file
data = np.load("data/tabular_data.npz")
X_tabular = data["X"]
y_tabular = data["y"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tabular, y_tabular, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Random Forest Accuracy: {acc:.4f}")
