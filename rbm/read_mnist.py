import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab 

def read_images(filename):
    binfile = open(filename,'rb')
    buf = binfile.read()
    index = 0
    magic, train_img_num, numRows, numColums = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    images = []
    for i in range(train_img_num):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im)
        images.append(im)
    binfile.close()
    return np.array(images)

def read_labels(filename):
    binfile=open(filename,'rb')
    buf = binfile.read()
    head = struct.unpack_from('>II' , buf ,0)
    imgNum=head[1]
    offset = struct.calcsize('>II')
    numString='>'+str(imgNum)+"B"
    labels= struct.unpack_from(numString , buf , offset)
    binfile.close()
    labels=np.reshape(labels,[imgNum,1])
    return np.array(labels)

def get_each(images, labels):
    img0=[];img1=[];img2=[];img3=[];img4=[];img5=[];img6=[];img7=[];img8=[];img9=[]
    for i in range(len(labels)):
        label = labels[i]
        if label==0:
            img0.append(images[i])
        elif label==1:
            img1.append(images[i])
        elif label==2:
            img2.append(images[i])
        elif label==3:
            img3.append(images[i])
        elif label==4:
            img4.append(images[i])
        elif label==5:
            img5.append(images[i])
        elif label==6:
            img6.append(images[i])
        elif label==7:
            img7.append(images[i])
        elif label==8:
            img8.append(images[i])
        elif label==9:
            img9.append(images[i])
    return np.array(img0),np.array(img1),np.array(img2),np.array(img3),np.array(img4),np.array(img5),np.array(img6),np.array(img7),np.array(img8),np.array(img9)


def display(img):
    one_image = img.reshape(28,28)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    pylab.show()

