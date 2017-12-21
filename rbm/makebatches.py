import numpy as np

def generate(train_images,test_images):

	total_num = train_images.shape[0]
	batches_num = int(total_num / 100)
	dims_num = train_images.shape[1]
	batchsize = 100
	batchdata = np.zeros((batchsize, dims_num, batches_num))
	shuffle = np.random.permutation(train_images)	
	for i in range(batches_num):
		batchdata[:,:,i] = shuffle[i*batchsize:(i+1)*batchsize,:]
	
	total_num = test_images.shape[0]
	batches_num = int(total_num / 100)
	dims_num = test_images.shape[1]
	batchsize = 100
	testbatches = np.zeros((batchsize, dims_num, batches_num))
	shuffle = np.random.permutation(test_images)
	for i in range(batches_num):
		testbatches[:,:,i] = shuffle[i*batchsize:(i+1)*batchsize,:]

	return batchdata, testbatches