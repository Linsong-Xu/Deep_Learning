import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab 
import read_mnist
from sklearn.decomposition import PCA

def display_reduced(data, labels):
    plt.figure(figsize=(12, 10))
    axes = plt.subplot(111)
    type0_x = []
    type0_y = []
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []
    type8_x = []
    type8_y = []
    type9_x = []
    type9_y = []
    for i in range(len(labels)):
        if labels[i] == 0:
            type0_x.append(data[i][0])
            type0_y.append(data[i][1])

        elif labels[i] == 1: 
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])

        elif labels[i] == 2:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])

        elif labels[i] == 3:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])

        elif labels[i] == 4: 
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])

        elif labels[i] == 5:
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])

        elif labels[i] == 6:
            type6_x.append(data[i][0])
            type6_y.append(data[i][1])

        elif labels[i] == 7: 
            type7_x.append(data[i][0])
            type7_y.append(data[i][1])

        elif labels[i] == 8:
            type8_x.append(data[i][0])
            type8_y.append(data[i][1])

        elif labels[i] == 9:
            type9_x.append(data[i][0])
            type9_y.append(data[i][1])
    type0 = axes.scatter(type0_x, type0_y, s=10, marker='.')
    type1 = axes.scatter(type1_x, type1_y, s=10, marker='o')
    type2 = axes.scatter(type2_x, type2_y, s=10, marker='v')
    type3 = axes.scatter(type3_x, type3_y, s=10, marker='^')
    type4 = axes.scatter(type4_x, type4_y, s=10, marker='8')
    type5 = axes.scatter(type5_x, type5_y, s=10, marker='s')
    type6 = axes.scatter(type6_x, type6_y, s=10, marker='p')
    type7 = axes.scatter(type7_x, type7_y, s=10, marker='*')
    type8 = axes.scatter(type8_x, type8_y, s=10, marker='+')
    type9 = axes.scatter(type9_x, type9_y, s=10, marker='x')
    axes.legend((type0, type1, type2, type3, type4, type5, type6, type7, type8, type9), ('0','1','2','3','4','5','6','7','8','9'), loc=1)
    plt.show()

images = read_mnist.read_images('train-images.idx3-ubyte')
labels = read_mnist.read_labels('train-labels.idx1-ubyte')

X = np.array(images)
pca = PCA(n_components=2)
X_new = pca.fit_transform(X)

display_reduced(X_new, labels[:,0])