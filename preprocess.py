'''
Standardize dataset by resizing all images to 128x128 and converting to grayscale
'''

import os
import numpy as np
import cv2
import sys

def preprocess(f):
    img = cv2.imread('data/' + f, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    return img

if len(sys.argv) == 1:
    avg = np.zeros((128, 128))

    for i, f in enumerate(os.listdir('data/raw')):
        img = preprocess("raw/" + f)
        cv2.imwrite(f'data/processed/{i}.png', img)
        avg += img

    avg /= len(os.listdir('data/raw'))
    print(avg)
    cv2.imwrite('data/avg.png', avg)
else:
    img = preprocess(sys.argv[1])
    cv2.imwrite('data/processed_' + sys.argv[1].split(".")[0] + ".png", img)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()