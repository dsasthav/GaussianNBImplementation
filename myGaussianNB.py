"""
My implementation of the GaussianNB class:
This implementation will only be used for predicting
the classes, it will not be used to output predicted probabilities
because NaiveBayes isn't really good at that anyway

"""

import pandas as pd 
import numpy as np 
import math as math

class myGaussianNB():

	def __init__(self):
		self.priors = {}
		self.summaries = {}
		self.target_vals = []
		self.separated_df = {}
		self.df = []
		self.target = ""

	# Set possible target values
	def get_targets(self, y):
		self.target_vals = np.unique(y)

	def separate_df(self):
		for val in self.target_vals:
			self.separated_df[val] = self.df[self.df[self.target] == val]

	# Calculate 
	def calculate_priors(self):
		for val in self.target_vals:
			self.priors[val] = float(self.separated_df[val].shape[0]) / self.df.shape[0]
	
	def calculate_summaries(self):
		for val in self.target_vals:
			self.summaries[val] = [self.separated_df[val].apply(lambda x: np.mean(x)),self.separated_df[val].apply(lambda x: np.std(x))]


	def fit(self, df, target):
		
		self.df = df
		self.target = target

		X = self.df.ix[:,self.df.columns != self.target]
		Y = self.df[self.target]

		# get unique target values
		self.get_targets(Y)

		# set separated df's
		self.separate_df()

		# Calculate class priors
		self.calculate_priors()

		# Calculate class conditional mean and std 
		self.calculate_summaries()


	# Calculates normal density 
	def calculate_density(self, x, mean, std):
		exponential = np.exp(-math.pow(x-mean,2)/(2*math.pow(std,2)))
		return math.pow(2*math.pi*math.pow(std,2),-0.5)*exponential
	
	def calculate_class_probability(self, inputRow):
		# assume a pandas dataframe rowsc
		columns = inputRow.index.tolist()
		probabilities = {}
		for val in self.summaries.keys():
			probabilities[val] = self.priors[val]
			for col in columns:
				mean = self.summaries[val][0][col]
				std = self.summaries[val][1][col]
				probabilities[val] *= self.calculate_density(inputRow[col], mean, std)
		return probabilities

	def pickWinner(self,probabilities):
		return max(probabilities, key=probabilities.get)

	def predict(self, X):
		func = lambda x: self.pickWinner(self.calculate_class_probability(x))
		return X.apply(lambda x: func(x), axis = 1)



