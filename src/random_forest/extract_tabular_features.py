import h5py
import torch
import numpy as np
from scipy.stats import mode
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features_list = []
labels_list = []

with h5py.File("data/Dataset.h5", 'r') as hf:
    X = torch.tensor(hf['data'][:])        # [N, T, C, H, W]
    Y = torch.tensor(hf['labels'][:])      # [N, H, W]

for x, y in tqdm(zip(X, Y), total=len(X)):
    x_avg = x.mean(dim=[2, 3])  # [T, C]
    features = x_avg.flatten().numpy()  # [T*C]
    label = mode(y.flatten().numpy(), keepdims=False).mode
    features_list.append(features)
    labels_list.append(label)

X_tabular = np.stack(features_list)
y_tabular = np.array(labels_list)
