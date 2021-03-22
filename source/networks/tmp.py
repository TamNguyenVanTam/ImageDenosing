import cv2
import os

origDir = "/projects/FormDemo/dataset/fill/"
jpegDir = "/projects/FormProcessing/jpeg/"

if __name__ == "__main__":
	files = os.listdir(origDir)
	for file in files:
		name = file.split(".")[0]
		path = os.path.join(origDir, file)

		img  = cv2.imread(path)
		save = os.path.join(jpegDir, "{}.jpg".format(name))
		
		cv2.imwrite(save, img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])





python main.py --test_dir "../../jpeg" --save_dir "../../restored"
