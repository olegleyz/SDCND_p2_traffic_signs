import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import tensorflow as tf
from random import randint
import cv2


class DataSet():
	def __init__(self):
		self.X_train, self.y_train, self.X_test, self.y_test = [None]*4
		self.n_test, self.image_shape, self.image_shape, self.n_classes = [None]*4
		self.load_data()
		self.signnames = pd.read_csv('signnames.csv')
		self.signs_dic_train = self.get_signs_dic(self.y_train)
		self.signs_dic_test = self.get_signs_dic(self.y_test)
		self.signs_freq_list  = np.bincount(self.y_train)
		self.new_signs_count = max(self.signs_freq_list)# maximum of single class frequencies

	def get_class_name(self,index):
		"""
		Function takes class of the image and returns it's name
		"""
		return self.signnames[self.signnames.ClassId == index].SignName.tolist()[0]		

	def load_data(self):
		"""
		Function loads given training and testing dataset and returns 
		X_train, y_train, X_test, y_test
		"""
		# Load pickled data
		training_file = '../dataset/train.p'
		testing_file = '../dataset/test.p'

		with open(training_file, mode='rb') as f:
		    train = pickle.load(f)
		with open(testing_file, mode='rb') as f:
		    test = pickle.load(f)
		    
		self.X_train, self.y_train = train['features'], train['labels']
		self.X_test, self.y_test = test['features'], test['labels']

		self.n_train = len(self.X_train)
		# Number of testing examples
		self.n_test = len(self.y_train)
		# Shape of an traffic sign image
		self.image_shape = self.X_train.shape[1:3]
		# Number of unique classes/labels there are in the dataset.
		self.n_classes = len(np.unique(self.y_train))

	def get_summary(self):
		# Number of training examples
		n_train = len(self.X_train)
		# Number of testing examples
		n_test = len(self.y_train)
		# Shape of an traffic sign image
		image_shape = self.X_train.shape[1:3]
		# Number of unique classes/labels there are in the dataset.
		n_classes = len(np.unique(self.y_train))
		self.n_classes = n_classes

		print('There are {} training examples'.format(n_train))
		print('There are {} testing examples'.format(n_test))
		print("Image has size {}x{} px".format(image_shape[1],image_shape[0]))
		print("Number of classes is {}".format(n_classes))

	def get_train_density(self):
		"""
		Function visualizes histogram of traffic sign classes
		"""
		hist = sns.distplot(self.y_train, bins=43)
		plt.xlim(0, 42)

	def get_signs_dic(self,y):
	    """
	    Function creates dictionary of traffic signs class - index
	    """
	    signs_dic = {}
	    for i, var in enumerate(y):
	        if var not in signs_dic:
	            signs_dic[var] = [i]
	        else:
	            signs_dic[var].append(i)
	    
	    return signs_dic

	def show_signs(self, X,y,signs_dic):
	    f = plt.figure(figsize=(20,40))
	    
	    for i in range(self.n_classes):
	        ax = f.add_subplot(11, 4, i+1)
	        index = signs_dic[i][randint(0, len(signs_dic[i]))]
	        image = X[index].squeeze()
	        plt.grid(False)
	        plt.imshow(image, cmap="gray")
	        ax.set_title(str(index)+': '+self.get_class_name(y[index]))   


