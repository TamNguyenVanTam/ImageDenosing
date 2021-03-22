import os 
from sklearn.utils import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint


class Dataset:
	def __init__(self, data_dir, batch_size, tmp_folder):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.files = os.listdir(self.data_dir)
		self.idx   = 0
		self.quals = list(range(20, 100, 10))
		self.tmp_folder = tmp_folder
		self.iters = int(len(self.files) * 1.0/batch_size)
		self.pat_sizes = list(range(40, 80, 10))
		if not os.path.exists(self.tmp_folder):
			os.makedirs(self.tmp_folder)

	def getJpegImg(self, img, qual):
		cv2.imwrite(os.path.join(self.tmp_folder, "jpeg_tmp.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), qual])
		jpeg = cv2.imread(os.path.join(self.tmp_folder, "jpeg_tmp.jpg"))
		return jpeg

	def getNextBatch(self):
		'''
			Return a batch has the shape : batch_size x W x H x C
		'''
		nextIdx = min(self.idx + self.batch_size, len(self.files))
		ground_truths = []
		jpeg_imgs     = []
		for file in self.files[self.idx : nextIdx]:
			path = os.path.join(self.data_dir, file)
			ground = cv2.imread(path)
			for qual in self.quals:
				jpeg = self.getJpegImg(ground, qual)
				ground_truths.append(ground)
				jpeg_imgs.append(jpeg)
		if nextIdx == len(self.files):
			# finish one epoch 
			# shufle data train
			self.files = shuffle(self.files)
			self.idx   = 0
		else:
			self.idx = nextIdx
		pat_size = self.pat_sizes[randint(0, len(self.pat_sizes) - 1)] # augument patch size
		x = randint(0, 128 - pat_size)
		y = randint(0, 128 - pat_size)
		return np.array(ground_truths)[:, x : x + pat_size, y:y+pat_size, :], np.array(jpeg_imgs)[:, x : x + pat_size, y : y+pat_size, :]