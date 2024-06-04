import numpy as np
import cv2
from train import Dataset

dataset = Dataset('data/processed')
dataset.load()
mean = np.mean(dataset.data, axis=0)

# load eigensheep
eigen = []
for i in range(30):
    img = cv2.imread(f'data/eigen/{i}.png', cv2.IMREAD_GRAYSCALE)
    eigen.append(img.flatten())

# example projection
img = cv2.imread('data/eigen/2.png', cv2.IMREAD_GRAYSCALE)
img = img.flatten()
img = img - dataset.mean

print(img)

# vector projections
proj = []
for i in range(30):
    proj.append(np.dot(img, eigen[i]) / np.dot(eigen[i], eigen[i]))

print(proj)
recon = np.zeros_like(img, dtype=np.float32)
for i in range(30):
    recon += proj[i] * eigen[i]
recon = recon + dataset.mean
recon = recon.reshape(128, 128)
cv2.imwrite('data/recon.png', recon)