from model import *
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import random

def cal_psnr(im1, im2):
	# assert pixel value range is 0-255 and type is uint8
	mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
	psnr = 10 * np.log10(255 ** 2 / mse)
	return psnr

def getJpegImg(img, qual, tmp_folder):
	cv2.imwrite(os.path.join(tmp_folder, "jpeg_tmp.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), qual])
	jpeg = cv2.imread(os.path.join(tmp_folder, "jpeg_tmp.jpg"))
	return jpeg

def save_images(filepath, ground_truth, clean_image=None, noisy_image=None, crop = False):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if crop:
    	tmp = np.zeros((ground_truth.shape[0], 5, 3))
    if not clean_image.any():
        cat_image = ground_truth
    else:
    	if crop:
        	cat_image = np.concatenate([noisy_image,tmp, clean_image,tmp, ground_truth], axis=1)
        else:
        	cat_image = np.concatenate([noisy_image, clean_image, ground_truth], axis=1)
    cat_image = cat_image.astype('uint8')
    cv2.imwrite(filepath, cat_image)


class Denoiser(object):
	def __init__(self, sess, input_c_dim = 3, batch_size = 32, model_name = 'dncnn'):
		self.sess = sess
		self.input_c_dim = input_c_dim
		self.quals = list(range(20, 100, 10))
		self.model_name = model_name
		self.batch_size = batch_size
		self.stateModel()

	def stateModel(self):
		'''
			if self.model_name is dncnn using DnCNN model
	
		'''
		self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name = 'clean_image')
		self.X  = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name = 'jpeg_image')	
		self.is_training = tf.placeholder(tf.bool, name = 'is_training')
		if self.model_name == 'dncnn':
			print("======================selected model DNCNN===================")
			self.Y = dncnn(self.X, is_training = self.is_training)
		if self.model_name == 'cbd':
			print("======================selected model CBD=====================")
			self.Y = CBDNet(self.X)
		self.loss = (1.0 / (self.batch_size * len(self.quals))) * tf.nn.l2_loss(self.Y_ - self.Y)
		
		self.lr   = tf.placeholder(tf.float32, name = 'learning_rate')
		
		optimizer = tf.train.AdamOptimizer(self.lr, name = "AdamOptimizer")
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		
		with tf.control_dependencies(update_ops):
			self.train_op = optimizer.minimize(self.loss)
		
		init = tf.global_variables_initializer()
		self.sess.run(init)
		print("[*] Initialize model successfully...")

	def load(self, ckpt_dir):
		if not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)
			return False
		if len(os.listdir(ckpt_dir)) < 4:
			return False
		saver = tf.train.Saver()
		saver.restore(self.sess, os.path.join(ckpt_dir, "best_model.ckpt"))
		return True

	def evaluate(self, valid_data):
		total_loss = []
		for patch in range(valid_data.iters + 1):
			cleans, jpegs = valid_data.getNextBatch() 
			cleans = cleans.astype(np.float32) / 255.0
			jpegs  = jpegs.astype(np.float32) / 255.0
			l_p  = self.sess.run([self.loss], 
								feed_dict={self.Y_: cleans, self.is_training: False, self.X: jpegs})
			total_loss.append(l_p)
		return np.mean(total_loss)

	def save(self, ckpt_dir):
		saver = tf.train.Saver()
		if not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)
		print("[*] Saving model...")
		saver.save(self.sess,
				   os.path.join(ckpt_dir, "best_model.ckpt"))

	

	def fit(self, train_data, valid_data, ckpt_dir, n_epoches, lr):
		'''
			train_data is instance of Dataset Class that has a method : getNextBatch() 
			property iters		'''
		
		load_model_status = self.load(ckpt_dir)

		if load_model_status:
			print("[*] Model restore success!")
		else:
			print("[*] Not find pretrained model!")
		start_time = time.time()
		iter_num = 0
		valid_loss  = self.evaluate(valid_data) # have to modify for return loss in valid_data
		print("Loss initalizer {}".format(valid_loss))

		loss_hist_train = []
		loss_hist_valid = []
		best_loss = valid_loss

		for epoch in range(n_epoches):
			print("Epoch {}".format(epoch))

			l_e = []
			for batch in range(train_data.iters + 1):
				(cleans, jpegs) = train_data.getNextBatch() 
				cleans = cleans.astype(np.float32) / 255.0
				jpegs  = jpegs.astype(np.float32) / 255.0
				_, l = self.sess.run([self.train_op, self.loss], feed_dict={self.Y_: cleans, self.lr: lr[epoch], self.is_training: True, self.X: jpegs})
				
				l_e.append(l)
				print("Epoch: {} batch {}: loss: {}".format(epoch, batch, l))	
			
			valid_loss  = self.evaluate(valid_data)

			loss_hist_train.append(np.mean(l_e))
			loss_hist_valid.append(valid_loss)
			
			print("Loss on train set {}".format(np.mean(l_e)))
			print("loss on valid set {}".format(valid_loss))
			
			if valid_loss < best_loss:
				self.save(ckpt_dir)
				best_loss = valid_loss
				print("Update save best model")

		plt.plot(loss_hist_train)
		plt.plot(loss_hist_valid)
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'valid'], loc='upper left')
		plt.savefig("log_train_model_{}.png".format(self.model_name))
		print("Training successing! {}".format(time.time()))


	def predict2(self, test_dir, ckpt_dir, save_dir, crop = False):
		'''
			Run with exist jpeg image. Dont need to compression
			
		'''
		test_files = os.listdir(test_dir)
		print(len(test_files))
		assert len(test_files) != 0, "No testing data"
		load_model_status = self.load(ckpt_dir)
		assert load_model_status == True, "[!] Load weights FAILED..."
		print("[*] Load weights SUCCESS...")
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		t1 = time.time()
		for idx in xrange(len(test_files)):
			print(test_files[idx])
			jpeg = cv2.imread(os.path.join(test_dir, test_files[idx]))
			w, h, c = jpeg.shape
			if crop:
				x = random.randint(0, w - 512)
				y = random.randint(0, h - 512)
				jpeg = jpeg[x:x + 512, y:y+512, :]
				w, h, c = jpeg.shape
			jpeg = jpeg / 255.0
			jpeg = np.reshape(jpeg, (1, w, h, c))
			output_clean_image = self.sess.run(self.Y, feed_dict={self.X: jpeg, self.is_training: False})
			outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
			outputimage = np.reshape(outputimage, (w, h, c))
			cv2.imwrite(os.path.join(save_dir, test_files[idx]), outputimage)
		print("Time comsume {}".format((time.time() - t1) / len(test_files)) )


	def predict1(self, test_dir, ckpt_dir, save_dir, tmp_folder, crop):
		'''
			Test_dir: a directory consists a clean image
			Create jpeg quality image for each jpeg qualiy level
			
		'''
		test_files = os.listdir(test_dir)
		print(len(test_files))
		assert len(test_files) != 0, "No testing data"
		load_model_status = self.load(ckpt_dir)
		assert load_model_status == True, "[!] Load weights FAILED..."
		print("[*] Load weights SUCCESS...")
		for qual in range(20,100, 10):
			psnr_sum = 0
			k = 0
			print("test with quality {}".format(qual))
			save_dir_qual = os.path.join(save_dir, str(qual))
			if not os.path.exists(save_dir_qual):
				os.makedirs(save_dir_qual)
			for idx in xrange(len(test_files)):
				clean_image = cv2.imread(os.path.join(test_dir, test_files[idx]))
				w, h, c = clean_image.shape
				
				jpeg = getJpegImg(clean_image, qual, tmp_folder )
				if crop:
					x = random.randint(0, w - 512)
					y = random.randint(0, h - 512)
					jpeg = jpeg[x:x + 512, y:y+512, :]
					clean_image = clean_image[x:x + 512, y:y+512, :]
					w, h, c = jpeg.shape
				jpeg = jpeg / 255.0
				jpeg = jpeg.reshape(1, w, h, c)
				clean_image = clean_image.reshape(1, w, h, c)

				t =time.time()
				output_clean_image = self.sess.run(self.Y,
																feed_dict={self.Y_: clean_image / 255.0, self.X: jpeg, self.is_training: False})
				outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
				noisyimage  = np.clip(255 * jpeg, 0, 255).astype('uint8')
				groundtruth = clean_image
			
				psnr = cal_psnr(groundtruth, outputimage)
				print("img%d PSNR: %.2f" % (idx, psnr))
				psnr_sum += psnr
				save_images(os.path.join(save_dir_qual, 'test_%d_%d.png' % (k+1,qual)), groundtruth, outputimage, noisyimage, crop = crop)
				k+=1
			avg_psnr = psnr_sum / k
			print("--- Average PSNR %.2f ---" % avg_psnr)