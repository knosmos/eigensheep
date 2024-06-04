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
            img = img.flatten()
            self.data.append(img)
        self.data = np.array(self.data)
        self.mean = np.mean(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# generate covariance matrix
def covariance(dataset):
    data = dataset.data - dataset.mean
    #cov = np.matmul(data, data.T) / len(dataset)
    cov = np.cov(data)
    return cov

# generate eigensheep
def eigensheep(cov, dataset):
    # compute eigenvectors and eigenvalues of covariance matrix
    vals, vecs = np.linalg.eig(cov)
    vecs = np.real(vecs)
    vals = np.real(vals)

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
#print(vals)

print(vecs)

# example projection
img = cv2.imread('data/processed/2.png', cv2.IMREAD_GRAYSCALE)
img = img.flatten()
img = img - dataset.mean

print(img)

# vector projections
proj = []
for i in range(50):
    proj.append(np.dot(img, vecs[:, i]) / np.dot(vecs[:, i], vecs[:, i]))

print(proj)
recon = np.zeros_like(img, dtype=np.float32)
for i in range(50):
    recon += proj[i] * vecs[:, i]
recon = recon + dataset.mean
recon = recon.reshape(128, 128)
cv2.imwrite('data/recon.png', recon)

# generate eigensheep images
for i in range(30):
    sheep = vecs[:, i].reshape(128, 128)
    cv2.imwrite(f'data/eigen/{i}.png', sheep + dataset.mean.reshape(128, 128))