'''
Standardize dataset by resizing all images to 128x128 and converting to grayscale
'''

import os
import numpy as np
import cv2

avg = np.zeros((128, 128))

for i, f in enumerate(os.listdir('data/raw')):
    img = cv2.imread('data/raw/' + f, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    cv2.imwrite(f'data/processed/{i}.png', img)
    avg += img

avg /= len(os.listdir('data/raw'))
print(avg)
cv2.imwrite('data/avg.png', avg)