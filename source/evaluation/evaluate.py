import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def mean_square_error(im1, im2):
	mse = ((im1.astype(np.float) - im2.astype(np.float))) 
	mse = mse ** 2

	return mse.mean()

def seperate(img):
	w, h, _ = img.shape
	h = h / 3
	noise = img[:, 0:h]
	denoise = img[:, h:2*h]
	clean = img[:, 2*h:3*h]
	return clean, denoise, noise


def seperate_frequence_result(img):
	w, h, _ = img.shape
	h = h / 4
	jpeg = img[:, 0:h]
	dncnn   = img[:, h:2*h]
	fft_filter = img[:, 2*h:3*h]
	clean = img[:, 3*h:4*h]
	return jpeg, dncnn, fft_filter, clean

def evaluate_one_qual_(datadir):
	paths = os.listdir(datadir)
	sorted(paths)
	ratio = []
	for path in paths:
		g_d_n = cv2.imread(os.path.join(datadir, path))
		jpeg, dncnn, fft_filter, clean = seperate_frequence_result(g_d_n)
		c_d = mean_square_error(fft_filter, clean)
		c_n = mean_square_error(jpeg, clean)
		ratio.append(c_n/c_d)
	print '%.3f' % np.mean(ratio)
	
def evaluate_one_qual(datadir):
	paths = os.listdir(datadir)
	sorted(paths)
	ratio = []
	for path in paths:
		g_d_n = cv2.imread(os.path.join(datadir, path))
		jpeg, dncnn, clean = seperate(g_d_n)
		c_d = mean_square_error(dncnn, clean)
		c_n = mean_square_error(jpeg, clean)
		ratio.append(c_n/c_d)
	print '%.3f' % np.mean(ratio)


if __name__== "__main__":
	datadir = "../../results/DNCNN/English"
	for qual in range(20, 100, 10):
		evaluate_one_qual(os.path.join(datadir, str(qual)))
