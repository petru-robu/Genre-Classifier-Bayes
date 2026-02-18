"""
Gaussian Naive Bayes implementation
"""
import numpy as np

class GaussianNB:
	def fit(self, X, y):
		self.classes = np.unique(y)
		n_features = X.shape[1]

		self.priors = {}
		self.means = {}
		self.vars = {}

		for cl in self.classes:
			X_c = X[y == cl]

			self.priors[cl] = X_c.shape[0] / X.shape[0]
			self.means[cl] = X_c.mean(axis=0)
			self.vars[cl] = X_c.var(axis=0) + 1e-6 

	def log_gaussian(self, x, mean, var):
		return -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

	def predict(self, X):
		predictions = []

		for x in X:
			log_posteriors = {}

			for cl in self.classes:
				log_prior = np.log(self.priors[cl])
				log_likelihood = self.log_gaussian(x, self.means[cl], self.vars[cl]).sum()

				log_posteriors[cl] = log_prior + log_likelihood

			predictions.append(max(log_posteriors, key=log_posteriors.get))
		return np.array(predictions)