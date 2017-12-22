import numpy as np
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
def get_fdf(data,VV,Dim):
	'''
		data(1000,784)
		w1(785,1000)
		w2(1001,500)
		w3(501,250)
		w4(251,2)
		w5(3,250)
		w6(251,500)
		w7(501,1000)
		w8(1001,784)
		l1~l9:784,1000,500,250,2,250,500,1000,784
	'''
	numcases = 1000
	l1=Dim[0];l2=Dim[1];l3=Dim[2];l4=Dim[3];l5=Dim[4];l6=Dim[5];l7=Dim[6];l8=Dim[7];l9=Dim[8]

	N = data.shape[0]
	
	w1 = VV[:(l1+1)*l2].reshape(l1+1,l2)
	xxx = (l1+1)*l2
	w2 = VV[xxx:xxx+(l2+1)*l3].reshape(l2+1,l3)
	xxx = xxx + (l2+1)*l3
	w3 = VV[xxx:xxx+(l3+1)*l4].reshape(l3+1,l4)
	xxx = xxx + (l3+1)*l4
	w4 = VV[xxx:xxx+(l4+1)*l5].reshape(l4+1,l5)
	xxx = xxx + (l4+1)*l5
	w5 = VV[xxx:xxx+(l5+1)*l6].reshape(l5+1,l6)
	xxx = xxx + (l5+1)*l6
	w6 = VV[xxx:xxx+(l6+1)*l7].reshape(l6+1,l7)
	xxx = xxx + (l6+1)*l7
	w7 = VV[xxx:xxx+(l7+1)*l8].reshape(l7+1,l8)
	xxx = xxx + (l7+1)*l8
	w8 = VV[xxx:xxx+(l8+1)*l9].reshape(l8+1,l9)


	data = np.concatenate((data, np.ones((N,1))), axis=1)
	w1probs = sigmoid(np.dot(data, w1))
	w1probs = np.concatenate((w1probs, np.ones((numcases,1))), axis=1)
	w2probs = sigmoid(np.dot(w1probs, w2))
	w2probs = np.concatenate((w2probs, np.ones((numcases,1))), axis=1)
	w3probs = sigmoid(np.dot(w2probs, w3))
	w3probs = np.concatenate((w3probs, np.ones((numcases,1))), axis=1)
	w4probs = np.dot(w3probs, w4)
	w4probs = np.concatenate((w4probs, np.ones((numcases,1))), axis=1)
	w5probs = sigmoid(np.dot(w4probs, w5))
	w5probs = np.concatenate((w5probs, np.ones((numcases,1))), axis=1)
	w6probs = sigmoid(np.dot(w5probs, w6))
	w6probs = np.concatenate((w6probs, np.ones((numcases,1))), axis=1)
	w7probs = sigmoid(np.dot(w6probs, w7))
	w7probs = np.concatenate((w7probs, np.ones((numcases,1))), axis=1)
	dataout = sigmoid(np.dot(w7probs, w8))

	f = -1/N*np.sum(np.multiply(data[:,:-1],np.log(dataout))+np.multiply((1-data[:,:-1]), np.log(1-dataout)))
	#IO = 1/N*(dataout-data[:,:-1])/(np.multiply(dataout, 1-dataout))
	IO = 1/N*(dataout-data[:,:-1])
	Ix8 = IO
	dw8 = np.dot(w7probs.T,Ix8)

	Ix7 = np.multiply(np.multiply(np.dot(Ix8,w8.T), w7probs),(1-w7probs))
	Ix7 = Ix7[:,:-1]
	dw7 = np.dot(w6probs.T, Ix7)

	Ix6 = np.multiply(np.multiply(np.dot(Ix7,w7.T), w6probs),(1-w6probs))
	Ix6 = Ix6[:,:-1]
	dw6 = np.dot(w5probs.T, Ix6)

	Ix5 = np.multiply(np.multiply(np.dot(Ix6,w6.T), w5probs),(1-w5probs))
	Ix5 = Ix5[:,:-1]
	dw5 = np.dot(w4probs.T, Ix5)

	Ix4 = np.dot(Ix5, w5.T)
	Ix4 = Ix4[:,:-1]
	dw4 = np.dot(w3probs.T, Ix4)

	Ix3 = np.multiply(np.multiply(np.dot(Ix4,w4.T), w3probs),(1-w3probs))
	Ix3 = Ix3[:,:-1]
	dw3 = np.dot(w2probs.T, Ix3)

	Ix2 = np.multiply(np.multiply(np.dot(Ix3,w3.T), w2probs),(1-w2probs))
	Ix2 = Ix2[:,:-1]
	dw2 = np.dot(w1probs.T, Ix2)

	Ix1 = np.multiply(np.multiply(np.dot(Ix2,w2.T), w1probs),(1-w1probs))
	Ix1 = Ix1[:,:-1]
	dw1 = np.dot(data.T, Ix1)

	n1 = dw1.shape[0]*dw1.shape[1]
	n2 = dw2.shape[0]*dw2.shape[1]
	n3 = dw3.shape[0]*dw3.shape[1]
	n4 = dw4.shape[0]*dw4.shape[1]
	n5 = dw5.shape[0]*dw5.shape[1]
	n6 = dw6.shape[0]*dw6.shape[1]
	n7 = dw7.shape[0]*dw7.shape[1]
	n8 = dw8.shape[0]*dw8.shape[1]

	df = np.concatenate((dw1.reshape(n1,1), dw2.reshape(n2,1), dw3.reshape(n3,1), dw4.reshape(n4,1), dw5.reshape(n5,1), dw6.reshape(n6,1), dw7.reshape(n7,1), dw8.reshape(n8,1)), axis = 0)

	return f,df
