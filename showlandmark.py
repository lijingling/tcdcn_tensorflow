import os,sys
import numpy as np 
from PIL import Image, ImageDraw
from PrepareData import get_img_landmark, get_boxpos
from glob import glob
data_path = 'dataprocess2/outputdel'

def getBox(img_path):
	txtfile = img_path.split(".")[0] + ".txt"
	fileobj = open(txtfile)
	xmin = 512
	xmax = 0
	ymin = 512
	ymax = 0

	while 1:
		fileline = fileobj.readline()
		if not fileline:
			break
		pos = fileline.split(" ")
		xpos = float(pos[0])
		ypos = float(pos[1])
		if xpos < xmin:
			xmin = xpos
		if xpos > xmax:
			xmax = xpos;
		if ypos < ymin:
			ymin = ypos
		if ypos >ymax:
			ymax = ypos

	x0 = xmin
	w0 = xmax - xmin
	y0 = ymin
	h0 = ymax - ymin

	margin = 0.1
	x1 = int(round(max(0, x0 - w0 * margin)))
	y1 = int(round(max(0, y0 - h0 * margin)))
	w1 = int(round(min(512, x0 + w0 * (1+margin)) - x0))
	h1 = int(round(min(512, y0 + h0 * (1+margin)) - y0))	
	return x1, y1, w1, h1


imgindex = np.loadtxt("out_index.txt")
landmark = np.loadtxt("out_landmark.txt")
testimgnum = len(imgindex)
landmarknum = 68

imgdata = glob(os.path.join(data_path, '*.jpg'))
for i in range(testimgnum):
#for i in range(1):
	index = int(imgindex[i])
	img_path = imgdata[index]
	print img_path
	img = Image.open(img_path)
	x1, y1, w1, h1 = getBox(img_path)
	print (x1, y1, w1, h1)

	draw = ImageDraw.Draw(img)
	
	#draw groudtruth
	txtfile = img_path.split(".")[0] + ".txt"
	fileobj = open(txtfile)
	while 1:
		fileline = fileobj.readline()
		if not fileline:
			break
		pos = fileline.split(" ")
		xpos = float(pos[0])
		ypos = float(pos[1])
		gbbox = (xpos, ypos, xpos+5, ypos+5)
		draw.ellipse(gbbox, fill = (255, 0, 0))
	
	
	for j in range(landmarknum):
		x = landmark[i, j*2] * w1 + x1
		y = landmark[i, j*2+1] * h1 + y1
		bbox = (x, y, x+5, y+5)
		draw.ellipse(bbox, fill = (0, 255, 0))
	
	del draw
	img.save("outputimg/"+str(i)+"out.png")
