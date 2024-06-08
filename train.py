'''
Generate eigensheep from processed dataset
'''

import numpy as np
import cv2
import os

# dataset class
class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = []
        self.mean = None

    def load(self):
        for f in os.listdir('data/processed'):
            img = cv2.imread('data/processed/' + f, cv2.IMREAD_GRAYSCALE)
            img = img.flatten() / 255.0
            self.data.append(img)
        self.data = np.array(self.data)
        self.mean = np.mean(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# generate covariance matrix
def covariance(dataset):
    return np.dot(
        dataset.data - dataset.mean,
        (dataset.data - dataset.mean).T
    )

# generate eigensheep
def eigensheep(cov, dataset):
    # compute eigenvectors and eigenvalues of covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    vecs = np.dot(dataset.data.T, vecs)

    # sort eigenvectors by eigenvalues
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs

dataset = Dataset('data/processed')
dataset.load()

cov = covariance(dataset)
vals, vecs = eigensheep(cov, dataset)

print("eigenvectors:", vecs)

# save eigenvectors to npy
np.save('data/eigensheep.npy', vecs)

# generate sample eigensheep images
for i in range(30):
    sheep = vecs[:, i].reshape(128, 128)
    sheep += dataset.mean.reshape(128, 128)
    # normalize to [0, 255]
    sheep = (sheep - np.min(sheep)) / (np.max(sheep) - np.min(sheep)) * 255
    cv2.imwrite(f'data/eigen/{i}.png', sheep)