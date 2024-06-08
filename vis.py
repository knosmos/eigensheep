import cv2
import numpy as np
import os
import sys

# load eigenvectors
vecs = np.load('data/eigensheep.npy')
print("eigensheep loaded")

# load image
img = cv2.imread('data/' + sys.argv[1], cv2.IMREAD_GRAYSCALE)
img = img.flatten() / 255.0
img = img - dataset.mean

print("image loaded")

proj = []
for i in range(150):
    proj.append(np.dot(img, vecs[:, i]) / np.dot(vecs[:, i], vecs[:, i]))

# build video
video = cv2.VideoWriter('data/projections.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))

recon = np.zeros_like(img, dtype=np.float32)
for i in range(150):
    recon += proj[i] * vecs[:, i]
    k = ((recon + dataset.mean).reshape(128, 128) * 255).astype(np.uint8)
    video.write(cv2.cvtColor(k, cv2.COLOR_GRAY2BGR))
for i in range(60):
    video.write(cv2.cvtColor(k, cv2.COLOR_GRAY2BGR))
video.release()
print("reconstruction complete, video saved")

# reconstruction error
err = np.linalg.norm(img - recon.flatten())
print("error:", err)

recon = recon + dataset.mean
recon = recon.reshape(128, 128)

# save reconstructed image
cv2.imwrite('data/recon.png', recon * 255)