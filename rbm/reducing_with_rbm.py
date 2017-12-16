import numpy as np
import read_mnist
import rbm
import makebatches
import backprop

'''
prepare the data
'''
train_images = read_mnist.read_images('train-images.idx3-ubyte')
train_labels = read_mnist.read_labels('train-labels.idx1-ubyte')
test_images = read_mnist.read_images('t10k-images.idx3-ubyte')
test_labels = read_mnist.read_labels('t10k-labels.idx1-ubyte')

train_images = train_images / 255
test_images = test_images / 255

img0,img1,img2,img3,img4,img5,img6,img7,img8,img9 = read_mnist.get_each(train_images,train_labels)
test0,test1,test2,test3,test4,test5,test6,test7,test8,test9 = read_mnist.get_each(test_images,test_labels)

databaches,testbatches = makebatches.generate(train_images, test_images)


'''
train 4 rbms
784->1000->500->250->2
'''

dim = databaches.shape[1]
numlay1 = 1000; numlay2 = 500; numlay3 = 250; numlay4 = 2


rbm1 = rbm.rbm(input=databaches, n_visible=dim,n_hidden=numlay1)
h1 = rbm1.contrastive_divergence(maxepoch=10, k=1)


rbm2 = rbm.rbm(input=h1, n_visible=numlay1,n_hidden=numlay2)
h2 = rbm2.contrastive_divergence(maxepoch=10, k=1)


rbm3 = rbm.rbm(input=h2, n_visible=numlay2,n_hidden=numlay3)
h3 = rbm3.contrastive_divergence(maxepoch=10, k=1)


rbm4 = rbm.rbm(input=h3, n_visible=numlay3,n_hidden=numlay4)
h4 = rbm4.rbmhiddenlinear(maxepoch=10, k=1)

backprop.fine_tuning(rbm1, rbm2, rbm3, rbm4, train_images, test_images, maxepoch=200)
