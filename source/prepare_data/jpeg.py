'''
	This file create jpeg image with quality in range 20-90 from original forder
'''
import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--clean_image_dir', dest='clean_dir', default='../../data/test/Cus/', help='clean image dir')
parser.add_argument('--jpeg_image_dir', dest='jpeg_dir', default='../../data/jpeg', help= 'dir for save jpeg image')
args = parser.parse_args()

def makeDir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def getJpegImg(img, qual):
	if not os.path.exists("./data/tmp/"):
		os.makedirs("./data/tmp/")
	tmp_path = "./data/tmp/jpeg_tmp.jpg"
	cv2.imwrite(tmp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), qual])
	jpeg = cv2.imread(tmp_path)
	return jpeg

def jpegCompress(minQuality, maxQuality, stride):
	quals = list(range(minQuality, maxQuality, stride))
	image_files = os.listdir(args.clean_dir)
	for qual in quals:
		print("Start create jpeg with quality {}".format(qual))
		path_jpeg_dir = os.path.join(args.jpeg_dir, str(qual))
		makeDir(path_jpeg_dir)
		for file in image_files:
			image_path = os.path.join(args.clean_dir, file)
			img = cv2.imread(image_path)
			jpeg = getJpegImg(img, qual)
			jpeg_path = os.path.join(path_jpeg_dir, file)
			cv2.imwrite(jpeg_path, jpeg)
		print("Done")

if __name__ == "__main__":
	jpegCompress(20, 100, 10)