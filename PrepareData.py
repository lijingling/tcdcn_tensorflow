import os,sys
import scipy.misc
import json
import numpy as np
from glob import glob
from PIL import Image
import random

data_path = 'dataprocess2/outputdel'
box_path = 'dataprocess2/boxfile.txt'
landmark_num = 68

def get_boxpos(box_path):
	boxobj = open(box_path)
	boxlist = {}
	while 1:
		fileline = boxobj.readline()
		if not fileline:
			break
		imgname = fileline.split(" ")[0]
		boxlist[imgname] = fileline
	return boxlist

def get_img_landmark(img_path, boxlist, resize_h):
	# get image
	resize_w = resize_h
	img = Image.open(img_path)
	grayimg = img.convert('L') 
	h, w = grayimg.size 

	imgname = img_path.split(".")[0]
	boxname = imgname.split("/")[2]
	box = boxlist[boxname].split(" ")
	x0 = float(box[1])
	w0 = float(box[2]) - float(box[1])
	y0 = float(box[3])
	h0 = float(box[4]) - float(box[3])

	margin = 0.1
	x1 = int(round(max(0, x0 - w0 * margin)))
	y1 = int(round(max(0, y0 - h0 * margin)))
	w1 = int(round(min(w, x0 + w0 * (1+margin)) - x0))
	h1 = int(round(min(h, y0 + h0 * (1+margin)) - y0))

	bounds = (x1, y1, (x1+w1), (y1+h1))
	crop_img = grayimg.crop(bounds)
	resizeimg = crop_img.resize((resize_h, resize_w))
	#resizeimg.save("c.png")
	nlcrop_img = np.array(resizeimg)/127.5 - 1

	# get landmark
	landname = imgname + ".txt"
	fileobj = open(landname)
	landlist = []

	while 1:
		fileline = fileobj.readline()
		if not fileline:
			break
		xpos = (float(fileline.split(" ")[0]) - x1) / w1
		ypos = (float(fileline.split(" ")[1]) - y1) / h1
		landlist.append(xpos)
		landlist.append(ypos)
	landmark = np.array(landlist)
	return nlcrop_img, landmark, bounds

def get_next_batch(img_path, imagebox, resize_h, batch_size):
	imgdata = glob(os.path.join(img_path, '*.jpg'))
	imgnum = len(imgdata)
	trainnum = int(imgnum*0.8)
	lst = [i for i in range(0, trainnum)]
	shuffle_list = random.sample(lst, batch_size)

	batch_imgs = []
	landmark = []
	for i in range(batch_size):
		img_index = shuffle_list[i]
		tmpimg, tmpland, bounds = get_img_landmark(imgdata[img_index], imagebox, resize_h)
		batch_imgs.append(tmpimg)
		landmark.append(tmpland)

	batchimgarr = np.array(batch_imgs)
	landmarkarr = np.array(landmark)
	return batchimgarr, landmarkarr

def get_batch(batch_size):
	imagebox = get_boxpos(box_path)
	resize_h = 40
	img, landmark = get_next_batch(data_path, imagebox, resize_h, batch_size)
	return img, landmark

def get_testset(testbatch):
	imagebox = get_boxpos(box_path)
	resize_h = 40
	imgdata = glob(os.path.join(data_path, '*.jpg'))
	imgnum = len(imgdata)
	testnum = int(imgnum * 0.2)
	lst = [i for i in range(imgnum-testnum, imgnum)]
	shuffle_list = random.sample(lst, testbatch)
	batch_imgs = []
	landmark = []
	index = []
	for i in range(testbatch):
		img_index = shuffle_list[i]
		tmpimg, tmpland, bounds = get_img_landmark(imgdata[img_index], imagebox, resize_h)
		batch_imgs.append(tmpimg)
		landmark.append(tmpland)
		index.append(img_index)
	testimg = np.array(batch_imgs)
	landmarkarr = np.array(landmark)	
	testindex = np.array(index)
	return testimg, landmarkarr, testindex

if __name__ == '__main__':
	batch_size = 64
	testbatch = 5
	img, landmark, testindex = get_testset(testbatch)
	print testindex
