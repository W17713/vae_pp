import tensorflow as tf
import numpy as np
import os
from PIL import Image


class img:
	def __init__(self):
        	pass

	def load_data(self,path,bs,ims):
		'''imgs=tf.keras.utils.image_dataset_from_directory(path,
		labels=None,
		label_mode=None,
		color_mode='grayscale',
		batch_size=bs,
		image_size=ims,interpolation='bilinear',
		shuffle=True,
		seed=None,
		validation_split=None,
		subset=None,
		follow_links=False,
		crop_to_aspect_ratio=False,)
		return imgs'''
		imgs = np.empty((0,ims[0],ims[1], 1)) #empty dummy array, we will append to this array all the images
		for filename in os.listdir(path):
			if filename.endswith(".jpeg"):
				img = Image.open(os.path.join(path,filename)).convert('L')
				imgs = np.append(imgs, np.array(img).reshape((1,ims[0],ims[1], 1)), axis=0)
		return imgs

	def preprocess(self,path,dest):
		#images = images.reshape((images.shape[0], 840, 840, 1)) / 255. #28, 28
		#return np.where(images > .5, 1.0, 0.0).astype('float32')
		imagenames=os.listdir(path)
		for i,j in enumerate(imagenames):
			if i<len(imagenames)-1:
				imgone=Image.open(os.path.join(path,j))
				imgtwo=Image.open(os.path.join(path,imagenames[i+1]))
				size1=imgone.size
				size2=imgtwo.size
				if size1[0]>size2[0]:
					minh=size2[0]
				else:
					minh=size1[0]
				if size1[1]>size2[1]:
					minw=size2[1]
				else:
					minw=size1[1]
				imgone.close()
				imgtwo.close()
		for name in imagenames:
			img=Image.open(os.path.join(path,name))
			img.resize((minh,minw))
			img.save(os.path.join(dest,name),'JPEG')
			img.close()
			
		print('minh: '+str(minh))
		print('minw: '+str(minw)) 
