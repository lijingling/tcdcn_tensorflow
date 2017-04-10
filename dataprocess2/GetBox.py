import os,shutil
import json
boxfile = open('boxfile.txt','w')
path = 'output'
dirs = os.listdir(path)
cnt = 0


for files in dirs:
    filetype = files.split(".")[1]
    if filetype == "txt":
    	print files
    	fileobj = open(path+"/"+files)
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
    			lx = xpos
    			ly = ypos
    		if xpos > xmax:
    			xmax = xpos;
    			rx = xpos
    			ry = ypos
    		if ypos < ymin:
    			ymin = ypos
    			ux = xpos
    			uy = ypos
    		if ypos >ymax:
    			ymax = ypos
    			dx = xpos
    			dy = ypos
    	boxfile.write(files.split(".")[0]+ " " + str(xmin) + " " + str(xmax) + " " + str(ymin) + " " + str(ymax) + "\n")
    	#boxfile.write(files.split(".")[0]+ " " + str(lx) + " " + str(ly) + " " + str(rx) + " " + str(ry) + " " + str(ux) + " " + str(uy) + " " + str(dx) + " " + str(dy) + "\n")
    cnt+=1
print cnt
boxfile.close() 