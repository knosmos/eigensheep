import numpy as np
import cv2
import os

# dataset class
class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = []

    def load(self):
        for f in os.listdir('data/processed'):
            img = cv2.imread('data/processed/' + f, cv2.IMREAD_GRAYSCALE)
            img = img.flatten()
            self.data.append(img)
        self.data = np.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# generate covariance matrix
def covariance(dataset):
    mean = np.mean(dataset.data, axis=0)
    data = dataset.data - mean
    cov = np.matmul(data, data.T) / len(dataset)
    return cov

# generate eigensheep
def eigensheep(cov, dataset):
    # compute eigenvectors and eigenvalues of covariance matrix
    vals, vecs = np.linalg.eig(cov)
    vecs = np.real(vecs)
    vals = np.real(vals)
    vecs = np.matmul(dataset.data.T, vecs)
    # sort eigenvectors by eigenvalues
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals, vecs

dataset = Dataset('data/processed')
dataset.load()

cov = covariance(dataset)
vals, vecs = eigensheep(cov, dataset)
#print(vals)

# generate eigensheep images
for i in range(30):
    face = vecs[:, i].reshape(128, 128)
    cv2.imwrite(f'data/eigen/{i}.png', face)