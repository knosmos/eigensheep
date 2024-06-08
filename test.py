import numpy as np
import cv2
import os
import sklearn.metrics
from train import Dataset

# load eigenvectors
vecs = np.load('data/eigensheep.npy')
print("eigensheep loaded")

# load dataset
dataset = Dataset('data/processed')
dataset.load()

# reconstruction error of one sample
def error(img, vecs):
    img = img.flatten() / 255.0
    img = img - dataset.mean

    proj = []
    for i in range(150):
        proj.append(np.dot(img, vecs[:, i]) / np.dot(vecs[:, i], vecs[:, i]))
    recon = np.zeros_like(img, dtype=np.float32)
    for i in range(150):
        recon += proj[i] * vecs[:, i]
    return np.linalg.norm(img - recon.flatten())

# run positive class
pos_score = []
for f in os.listdir('data/test/pos'):
    if f.endswith('.png'):
        img = cv2.imread('data/test/pos/' + f, cv2.IMREAD_GRAYSCALE)
        pos_score.append(error(img, vecs))
print("positive class error:", np.mean(pos_score))

# run negative class
neg_score = []
for f in os.listdir('data/test/neg'):
    if f.endswith('.png'):
        img = cv2.imread('data/test/neg/' + f, cv2.IMREAD_GRAYSCALE)
        neg_score.append(error(img, vecs))
print("negative class error:", np.mean(neg_score))

# compute ROC AUC
y_true = [0] * len(pos_score) + [1] * len(neg_score)
y_score = pos_score + neg_score
roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)
print("ROC AUC:", roc_auc)