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
    data = dataset.data - dataset.mean
    #cov = np.matmul(data, data.T) / len(dataset)
    cov = np.cov(data)
    return cov

# generate eigensheep
def eigensheep(cov, dataset):
    # compute eigenvectors and eigenvalues of covariance matrix
    vals, vecs = np.linalg.eigh(np.dot(dataset.data, dataset.data.T))
    #vecs = np.real(vecs)
    #vals = np.real(vals)

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

print("eigenvectors:", vecs)

# find orthogonality
'''
for i in range(200):
    for j in range(i + 1, 200):
        dot = np.dot(vecs[:, i], vecs[:, j])
        print(f"dot product of {i} and {j}:", dot)
'''

# example projection
import sys
img = cv2.imread('data/' + sys.argv[1], cv2.IMREAD_GRAYSCALE)
img = img.flatten() / 255.0
img = img - dataset.mean

print("image loaded:", img)

# vector projections
proj = []
for i in range(150):
    proj.append(np.dot(img, vecs[:, i]) / np.dot(vecs[:, i], vecs[:, i]))

# build video
video = cv2.VideoWriter('data/projections2.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))

print("projections:", proj)
recon = np.zeros_like(img, dtype=np.float32)
for i in range(150):
    recon += proj[i] * vecs[:, i]
    k = ((recon + dataset.mean).reshape(128, 128) * 255).astype(np.uint8)
    # clamp to [0, 255]
    #cv2.imshow('Reconstruction', k)
    #cv2.waitKey()
    print(k)
    video.write(cv2.cvtColor(k, cv2.COLOR_GRAY2BGR))
for i in range(60):
    video.write(cv2.cvtColor(k, cv2.COLOR_GRAY2BGR))
video.release()
print("reconstruction:", recon)

# reconstruction error
err = np.linalg.norm(img - recon.flatten())
print("error:", err)

recon = recon + dataset.mean
recon = recon.reshape(128, 128)

# save reconstructed image
cv2.imwrite('data/recon2.png', recon * 255)

# generate eigensheep images
for i in range(30):
    sheep = vecs[:, i].reshape(128, 128)
    sheep += dataset.mean.reshape(128, 128)
    # normalize to [0, 255]
    sheep = (sheep - np.min(sheep)) / (np.max(sheep) - np.min(sheep)) * 255
    cv2.imwrite(f'data/eigen/{i}.png', sheep)