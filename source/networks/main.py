import argparse
from glob import glob
from dataset import *
from denoiser import *

import os 
parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default = 100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default= 16, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')
parser.add_argument('--model', dest='model', default='dncnn', help='dncnn or cbd')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint/DNCNN', help='models are saved here')
parser.add_argument('--train_dir', dest='train_dir', default='../../data/train', help='dataset for training')
parser.add_argument('--valid_dir', dest='valid_dir', default="../../data/valid", help='dataset for eval in training')
parser.add_argument('--test_dir', dest='test_dir', default= "/projects/FormProcessing/data/test/tu/", help='dataset for eval in training')
parser.add_argument('--save_dir', dest='save_dir', default="../../results/DNCNN/Tu", help='dataset for eval in training')
parser.add_argument('--tmp_dir', dest='tmp_dir', default="../../data/tmp", help='save temporary result')
parser.add_argument('--jpeg_flag', dest='jpeg_flag', type = int , default="1", help='jpeg forder 0 or 1')
parser.add_argument('--gpu', dest='gpu', default = "0", help='gpu id')
args = parser.parse_args()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main():
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	
	lr = args.lr * np.ones([args.epoch])
	lr[30:] = lr[0] / 10.0
	if args.phase == 'train':
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 1.0)
	elif args.phase == 'test':
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.5)
		
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		denoiser = Denoiser(sess, batch_size = args.batch_size, model_name = args.model)
		if args.phase == 'train':
			print("Start training")
			train_data = Dataset(data_dir = args.train_dir, batch_size = args.batch_size, tmp_folder = args.tmp_dir)
			valid_data = Dataset(data_dir = args.valid_dir, batch_size = args.batch_size, tmp_folder = args.tmp_dir)
			denoiser.fit(train_data, valid_data, args.ckpt_dir, args.epoch, lr)
		elif args.phase == 'test':
			print("Start test")
			if args.jpeg_flag == 0:
				denoiser.predict1(args.test_dir, args.ckpt_dir, args.save_dir,args.tmp_dir, False)
			elif args.jpeg_flag == 1:
				denoiser.predict2(args.test_dir, args.ckpt_dir, args.save_dir, False)
			else:
				print("Unknown")
		else:
			print('[!]Unknown phase')
			exit(0)
	
if __name__ == "__main__":
	main()