## Deep Learning

### Classes：
* RBM：
  
  * First you should download the MNIST data from [here](http://yann.lecun.com/exdb/mnist/)
  
  * Then you can run pca.py, this code uses PCA to reduce the dimensionality of MNIST data from 784 to 2 and display it in graphic displays 
  
    * When I use PCA to reduce the dimensionality, I get this graphic:
    
    <div align=center>
     <img src='https://github.com/Linsong-Xu/Deep_Learning/blob/master/rbm/pca_reduce.png'>
    </div>
  
  * Run reducing_with_rbm.py, this code uses 784-1000-500-250-2 RBMs to train a model that can reduce the dimensionality of MNIST data from 784 to 2. Then fed the 784-dimension data to the 4-layers RBM model, and get the 2-dimension result. It will consume lots of time, so I don't get the result.(I reduce the maxepoch, but don't get the expected result as paper, even though it is better than PCA's result)

### Reference:
[1] : Reducing the Dimensionality of Data with Neural Networks

[2] : Learning Multiple Layers of Features from Tiny Images

[3] : A Practical Guide to Training Restricted Boltzmann Machines

[4] : [Training a deep autoencoder or a classifier on MNIST digits](http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html)

