import numpy as np

class SoftmaxRegression(object):
  '''
	input is the initial matrix[m,n]
	self.x: [m,n+1]
	label: [m,k] one hot
	W: [n+1, k]
	'''
	def __init__(self, input, label, n, k):
		self.x = np.concatenate((input, np.ones((input.shape[0],1))),axis=1)
		self.y = label
		self.W = np.zeros((n+1, k))

	def softmax(self, x):
		e = np.exp(x - np.max(x))
		if e.ndim == 1:
			return e / np.sum(e, axis=0)
		else:  
			return e / np.array([np.sum(e, axis=1)]).T

	def train(self, lr=0.1, lamda=0.0):
		m = self.x.shape[0]
		tmp = self.y - self.softmax(np.dot(self.x, self.W))
		deta = -1/m*np.dot(self.x.T, tmp) + lamda*self.W
		self.W -= lr*deta

	def costFunc(self, lamda=0.0):
		m = self.x.shape[0]
		p = self.softmax(np.dot(self.x, self.W))
		return -1/m*np.sum(np.multiply(self.y, np.log(p)))+0.5*lamda*np.sum(self.W**2)
