import sys
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse


WORD_PATCH_SIZE = 128

'''
    the purpose of this file is crop regions which have in one form   

'''
def pre_process(gray_image):
    '''
        Using opening for merge nearly conected coomponents 
    '''
    mask = np.ones((5, 4), np.int8)
    if np.mean(gray_image) < 127:
        gray_image = 255 - gray_image
    ret2,th2 = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   
    img_label_file_open = cv2.morphologyEx(th2, cv2.MORPH_OPEN, mask)
   
    if np.mean(img_label_file_open) > 127:
        img_label_file_open = 255 - img_label_file_open
    return img_label_file_open


def find_connected_component(img_label_file):
    w, h = img_label_file.shape
    cords = []
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_label_file, connectivity  = 4)
    for stat in stats:
        if stat[4] < w*h/4:
            cord_x =  int(stat[0])
            cord_y = int(stat[1])
            width = int(stat[2])
            height = int(stat[3])
            if width < 300 and height < 300 and width > 10 and height > 10:
                cords.append((cord_y, cord_x, height, width))
    return cords

def show_conneted_component(img_label_file, cords):
    img_label_file = cv2.cvtColor(img_label_file, cv2.COLOR_GRAY2RGB)
    for cord in cords:
        x, y, width, height = cord
        cv2.rectangle(img_label_file, (y,x), (y+height, x+width), (255, 0, 0), 5)
    plt.imshow(img_label_file)
    plt.show()



def get_word(img, flag = 1):
    img_label_file = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)
    preprocess_ = pre_process(img_label_file)
    cords = find_connected_component(preprocess_)
    # show_conneted_component(img_label_file, cords)
    new_cords = []
    for (x, y, w, h) in cords:
        x_center = x + int(w/2)
        y_center = y + int(h/2)
        new_x = x_center - WORD_PATCH_SIZE / 2
        new_y = y_center - WORD_PATCH_SIZE / 2
        if new_x >= 0 and new_y >= 0:
            new_cords.append((int(new_x), int(new_y)))
        
    words = []
    for (x, y) in new_cords:
        tmp = img[x:x+WORD_PATCH_SIZE, y:y+WORD_PATCH_SIZE, :]
        words.append(tmp)
    return words

def show_words(words, n_words = 2):
    for i in range(n_words):
        word = words[i]
        plt.imshow(word)
        plt.show()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', dest='img_dir', default = "../../data/Forms", help="the directory consists forms")
parser.add_argument('--train_dir', dest='train_dir', default= "../../data/train1", help= "training dir")
parser.add_argument('--valid_dir', dest='valid_dir', default= "../../data/valid1", help= "valid dir")
args = parser.parse_args()

if __name__ == "__main__":
    k = 0
    files = os.listdir(args.img_dir)
    n_file_training = int(len(files) * 0.8)

    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.exists(args.valid_dir):
        os.makedirs(args.valid_dir)
    print("Start")
    for idx,file in enumerate(files):
        if idx % 50 == 0 and idx > 0:
            print("Conpletely {} images".format(idx)) 
        img = cv2.imread(os.path.join(args.img_dir, file))
        img = cv2.resize(img,(1653, 2339),  interpolation = cv2.INTER_CUBIC)
        words = get_word(img)
        name = file.split('.')[0]
        if idx < n_file_training:
            path = args.train_dir
        else:
            path = args.valid_dir
        n = min(200, len(words))
        for w in words[0:n]:
            cv2.imwrite(os.path.join(path, "{}_{}.png".format(name, k)), w)
            k += 1
    print("Total batch {} ".format(k))