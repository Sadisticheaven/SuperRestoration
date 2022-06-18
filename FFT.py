import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image
from imresize import imresize
import utils
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


img = Image.open('./datasets/test/t1.png').convert('RGB')
img = np.array(img).astype(np.float32)
img = utils.rgb2ycbcr(img)[..., 0]
# img = cv2.imread('./datasets/test/t1.png', 0)
bic = imresize(img, 0.5, 'bicubic')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

dft2 = cv2.dft(np.float32(bic), flags=cv2.DFT_COMPLEX_OUTPUT)
dft2 = np.fft.fftshift(dft2)
dft2 = 20 * np.log(cv2.magnitude(dft2[:, :, 0], dft2[:, :, 1]))

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(bic, cmap='gray')
plt.title('bicubic'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(dft2, cmap='gray')
plt.title('bicubic_fft'), plt.xticks([]), plt.yticks([])
plt.show()
