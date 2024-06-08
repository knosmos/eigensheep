# eigensheep
## Image-based Ovine Detection with Principal Component Analysis
![image](https://github.com/knosmos/eigensheep/assets/30610197/af402972-c668-4652-a4ba-386087707e3d)

> “What do you call a baby eigensheep? A lamb, duh.” ― joke you’ve probably heard a thousand times

This project, contrived entirely to make the poor joke above, aims to differentiate images of sheep from images that do not contain sheep. To accomplish this, we use a method inspired by the machine learning technique of autoencoder anomaly detection: using a dataset of many sheep images, the program tries to learn to reduce a (high-dimensional) image of a sheep into a vector in a low-dimensional latent space. The goal is to accomplish this reduction in such a way that the latent space vector, despite being low-dimensional, still encodes sufficient information to reconstruct the original image. Then, the program will be very good at efficiently encoding images of sheep (leading to low reconstruction error), but should theoretically be very poor at efficiently encoding images of anything else (leading to a high reconstruction error). This difference in reconstruction error can be used to determine whether the image contains a sheep.

In modern machine learning, we would train a neural network to convert input data into latent space vectors. In this project, we instead use Principal Component Analysis (PCA) to accomplish this dimension reduction step. PCA, in essence, uses a coordinate transform to convert a large number of variables (in this case, the values of each pixel) into a smaller number of variables, while still preserving as much information as possible. It involves finding “principal components,” which are eigenvectors (eigensheep, I suppose) of the data’s covariance matrix. By sorting these eigensheep by their eigenvalues, we can determine the “importance” of each eigensheep, and keep the most significant eigensheep. We can then project the data along the coordinate system formed by the eigensheep basis to obtain our approximations, which can be compared to the original data to calculate the reconstruction error.

### Requirements
* `opencv-python`
* `numpy`
* `scikit-learn`

### Files
* `preprocess.py` standardizes images in `data/raw`.
* `train.py` generates eigensheep from images in `data/processed` and stores them in `data/eigensheep.npy`.
* `vis.py` generates reconstruction visualization videos.
* `test.py` uses samples in `data/test` to calculate classification performance.
