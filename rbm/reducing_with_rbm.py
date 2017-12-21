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

VV, Dim = backprop.fine_tuning(rbm1, rbm2, rbm3, rbm4, train_images, test_images, maxepoch=200)

l1=Dim[0];l2=Dim[1];l3=Dim[2];l4=Dim[3];l5=Dim[4];l6=Dim[5];l7=Dim[6];l8=Dim[7];l9=Dim[8]
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
rbm1.W = w1[l1,:]
rbm1.hbias = w1[l1,:]
rbm1.vbias = w8[l8,:]

rbm2.W = w2[l2,:]
rbm2.hbias = w2[l2,:]
rbm2.vbias = w7[l7,:]

rbm3.W = w3[l3,:]
rbm3.hbias = w3[l3,:]
rbm3.vbias = w6[l6,:]

rbm4.W = w4[l4,:]
rbm4.hbias = w4[l4,:]
rbm4.vbias = w5[l5,:]
