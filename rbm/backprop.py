import numpy as np
import makebatches
import fdf
import minimize

def sigmoid(x):
	return 1. / (1 + np.exp(-x))

def fine_tuning(rbm1, rbm2, rbm3, rbm4, train_images, test_images, maxepoch=200):
	databaches,testbatches = makebatches.generate(train_images, test_images)
	print('Fine-tuning deep autoencoder by minimizing cross entropy error. ');
	print('60 batches of 1000 cases each. ');

	w1 = np.concatenate((rbm1.W,rbm1.hbias), axis=0)#785,1000
	w2 = np.concatenate((rbm2.W,rbm2.hbias), axis=0)#1001,500
	w3 = np.concatenate((rbm3.W,rbm3.hbias), axis=0)#501,250
	w4 = np.concatenate((rbm4.W,rbm4.hbias), axis=0)#251,2
	w5 = np.concatenate((rbm4.W.T,rbm4.vbias), axis=0)#3,250
	w6 = np.concatenate((rbm3.W.T,rbm3.vbias), axis=0)#251,500
	w7 = np.concatenate((rbm2.W.T,rbm2.vbias), axis=0)#501,1000
	w8 = np.concatenate((rbm1.W.T,rbm1.vbias), axis=0)#1001,784

	l1 = w1.shape[0] - 1
	l2 = w2.shape[0] - 1 
	l3 = w3.shape[0] - 1
	l4 = w4.shape[0] - 1
	l5 = w5.shape[0] - 1
	l6 = w6.shape[0] - 1
	l7 = w7.shape[0] - 1
	l8 = w8.shape[0] - 1
	l9 = l1;

	n1 = w1.shape[0]*w1.shape[1]
	n2 = w2.shape[0]*w2.shape[1]
	n3 = w3.shape[0]*w3.shape[1]
	n4 = w4.shape[0]*w4.shape[1]
	n5 = w5.shape[0]*w5.shape[1]
	n6 = w6.shape[0]*w6.shape[1]
	n7 = w7.shape[0]*w7.shape[1]
	n8 = w8.shape[0]*w8.shape[1]
	VV = np.concatenate((w1.reshape(n1,1), w2.reshape(n2,1), w3.reshape(n3,1), w4.reshape(n4,1), w5.reshape(n5,1), w6.reshape(n6,1), w7.reshape(n7,1), w8.reshape(n8,1)), axis = 0)
	Dim = np.array([l1,l2,l3,l4,l5,l6,l7,l8,l9])

	test_err=[];
	train_err=[];

	'''
		caculate the train_error and test_error
	'''
	for epoch in range(maxepoch):
		err = 0;
		numcases, numdims, numbatches = databaches.shape
		for batch in range(numbatches):
			data = databaches[:,:,batch]
			data = np.concatenate((data, np.ones((numcases,1))), axis=1)
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
			err += 1/numcases*np.sum((data[:,:-1]-dataout)**2)
		train_err.append(err/numbatches)

		err = 0;
		testnumcases, testnumdims, testnumbatches = testbatches.shape
		for batch in range(testnumbatches):
			data = testbatches[:,:,batch]
			data = np.concatenate((data, np.ones((testnumcases,1))), axis=1)
			w1probs = sigmoid(np.dot(data, w1))
			w1probs = np.concatenate((w1probs, np.ones((testnumcases,1))), axis=1)
			w2probs = sigmoid(np.dot(w1probs, w2))
			w2probs = np.concatenate((w2probs, np.ones((testnumcases,1))), axis=1)
			w3probs = sigmoid(np.dot(w2probs, w3))
			w3probs = np.concatenate((w3probs, np.ones((testnumcases,1))), axis=1)
			w4probs = np.dot(w3probs, w4)
			w4probs = np.concatenate((w4probs, np.ones((testnumcases,1))), axis=1)
			w5probs = sigmoid(np.dot(w4probs, w5))
			w5probs = np.concatenate((w5probs, np.ones((testnumcases,1))), axis=1)
			w6probs = sigmoid(np.dot(w5probs, w6))
			w6probs = np.concatenate((w6probs, np.ones((testnumcases,1))), axis=1)
			w7probs = sigmoid(np.dot(w6probs, w7))
			w7probs = np.concatenate((w7probs, np.ones((testnumcases,1))), axis=1)
			dataout = sigmoid(np.dot(w7probs, w8))
			err += 1/testnumcases*np.sum((data[:,:-1]-dataout)**2)
		test_err.append(err/testnumbatches)

		print('Before epoch{}: --> Train squared error: {}  --> Test squared error: {}'.format(epoch, train_err[epoch], test_err[epoch]))

		'''
			re-makebatches: 60 batches of 1000 cases
			then fine tuning
		'''
		t = 0
		new_numbatches = int(numbatches / 10)
		max_iter = 3
		for batch in range(new_numbatches):
			print('epoch{} --> batch{}'.format(epoch, batch))
			data = databaches[:,:,t*10]
			for i in range(1,10):
				data = np.concatenate((data, databaches[:,:,t*10+i]), axis=0)
			t += 1
			f,df = fdf.get_fdf(data, VV, Dim)
			VV = minimize.get_minimize(f, df, data, VV, Dim, max_iter)

	return VV, Dim








