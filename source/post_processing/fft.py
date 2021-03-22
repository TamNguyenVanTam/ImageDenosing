import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
import time

import argparse


def seperate(img):
	w, h = img.shape[0], img.shape[1]
	h = h / 3 
	jpeg = img[:, :h]
	restore = img[:, h:2*h]
	clean   = img[:,2*h:3*h]
	return jpeg, restore, clean


def fft(img):
	"""
		Image format gray
	"""
	img = img.astype(np.float32)
	fft_img = np.fft.fft2(img)
	fft_shift = np.fft.fftshift(fft_img)
	return fft_shift


def ifft(fft):
	f_ishift = np.fft.ifftshift(fft)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)
	img_back = np.clip(img_back, 0, 255).astype('uint8')
	return img_back


def get_area(image_shape, r = 100):
	rows, cols = image_shape[0], image_shape[1]
	crow, ccol = int(rows / 2), int(cols / 2)
	center = [crow, ccol]
	x, y = np.ogrid[:rows, :cols]
	mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 < r*r
	return mask_area

def post_one_channel(jpeg, dncnn, r = 100):
	"""
		jpeg and dncnn format are grayscale
	"""
	mask_area = get_area(jpeg.shape, r)
	fft_jpeg  = fft(jpeg)
	fft_dncnn = fft(dncnn)
	fft_dncnn[mask_area] = fft_jpeg[mask_area]
	restore = ifft(fft_dncnn)
	return restore

def post(test, r = 100):
	jpeg, dncnn, ground = seperate(test)
	result = np.zeros(jpeg.shape)
	result[:, :, 0] = post_one_channel(jpeg[:, :, 0], dncnn[:, :, 0])
	result[:, :, 1] = post_one_channel(jpeg[:, :, 1], dncnn[:, :, 1])
	result[:, :, 2] = post_one_channel(jpeg[:, :, 2], dncnn[:, :, 2])
	return np.concatenate((jpeg, dncnn ,result, ground), axis = 1)

import matplotlib.pyplot as plt


A = np.zeros((200, 200), np.uint8) 
A[:, 50:53] = 255

fft_img = fft(A)
fft_img = np.abs(fft_img)

plt.imshow(fft_img, "gray")
plt.show()


# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--dncnn_dir', dest='dncnn_dir', default = "../../results/DnCNN_results/Test_set_2", help="DnCNN's result")
# parser.add_argument('--fft_dir', dest='fft_dir', default= "../../results/result_high_frequency_filter", help= "a directory for saving result")
# args = parser.parse_args()

# if __name__ == "__main__":
# 	dncnn_dir =	args.dncnn_dir
# 	fft_dir = args.fft_dir

# 	jpeg_quals = os.listdir(dncnn_dir)
# 	t1 = time.time()
# 	for jpeg_qual in jpeg_quals:
# 		jpeg_qual_dir = os.path.join(dncnn_dir, str(jpeg_qual))
# 		files = os.listdir(jpeg_qual_dir)
# 		fft_dir_qual = os.path.join(fft_dir, str(jpeg_qual))
# 		if not os.path.exists(fft_dir_qual):
# 			os.mkdir(fft_dir_qual)
# 		for file in files:
# 			print("Qual {} file {}".format(jpeg_qual, file))
# 			img = cv2.imread(os.path.join(jpeg_qual_dir, file))
# 			out = post(img)
# 			cv2.imwrite(os.path.join(fft_dir_qual, file), out)
# 	print("Time consume {}".format(time.time() - t1))
	


