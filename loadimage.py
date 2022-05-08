import os
from PIL import Image
import numpy as np

def getimg(path,imagesize):
	img=Image.open(path)
	grayimg=img.convert('L')
	arrimg=np.array(grayimg.resize(imagesize,Image.ANTIALIAS))
	#img=Image.imresize(img,imagesize[0],imagesize[1],1)
	return arrimg

def loadimg(path,imagesize):
	imagefiles=os.listdir(path)
	X=[]
	for f in imagefiles:
		X.append(getimg(os.path.join(path,f),imagesize))
	#X=1-np.array(X).astype('float32')/255
	return X
	


