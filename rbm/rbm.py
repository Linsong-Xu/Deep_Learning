import sys
import numpy as np

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

class rbm(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3, W=None, hbias=None, vbias=None, epsilonw=0.1, epsilonvb=0.1, epsilonhb=0.1, weightcost=0.0002, initialmomentum=0.5, finalmomentum=0.9):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.epsilonw = epsilonw
        self.epsilonvb = epsilonvb
        self.epsilonhb = epsilonhb
        self.weightcost = weightcost
        self.initialmomentum = initialmomentum
        self.finalmomentum = finalmomentum
            
        if W is None:
            W = np.random.normal(loc=0.0, scale=0.1, size=(n_visible,n_hidden))
        
        if hbias is None:
            hbias = np.zeros((1,n_hidden))
            
        if vbias is None:
            vbias = np.zeros((1,n_visible))
            
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        
    def contrastive_divergence(self, k=1, maxepoch=10):
        
        vishidinc = np.zeros((self.n_visible, self.n_hidden))
        visbiasinc = np.zeros(self.n_visible)
        hidbiasinc = np.zeros(self.n_hidden)
        batchposhidprobs = np.zeros((self.input.shape[0],self.n_hidden,self.input.shape[2]));

        for epoch in range(maxepoch):
            errsum = 0
            for batch in range(self.input.shape[2]):
                print('epoch: {} -> batch : {}....'.format(epoch,batch))
                data = self.input[:,:,batch]

                '''
                    in Hiton's source code is:(CD-1)
                        data(V) -> p(hidden layer with sigmoid) -> state(hidden layer 0/1) -> p(visiable layer with sigmoid) -> p(hidden layer with sigmoid)
                    in this code is:
                        data(V) -> state(hidden layer 0/1) -> state(visiable layer 0/1) -> state(hidden layer 0/1) -> ...
                        for example CD1:
                            data : v0
                            ph_mean : p(hidden layer -> h0)
                            ph_sample : state(hidden layer -> h0)

                            nv_means : p(visiable layer -> v1)
                            nv_sample : state(visiable layer -> v1)
                            nh_means : p(hidden layer -> h1)
                            nh_samples : state(hidden layer -> h1)
                    I find use Hinton's method can dicrease the error quickly, k=1
                '''

                ph_mean, ph_sample = self.sample_h_given_v(data)
                batchposhidprobs[:,:,batch] = ph_mean
                chain_start = ph_sample

                '''
                    Hinton's function: nostate_gibbs_hvh()
                    and you can also try this function: gibbs_hvh()
                '''

                for step in range(k):
                    if step == 0:
                        nv_means, nv_samples, nh_means, nh_samples = self.nostate_gibbs_hvh(chain_start)
                    else:
                        nv_means, nv_samples, nh_means, nh_samples = self.nostate_gibbs_hvh(nh_samples)
                
                err = np.sum((data - nv_means)**2)
                errsum += err


                if epoch > 4:
                    momentum = self.finalmomentum
                else:
                    momentum = self.initialmomentum

                vishidinc = momentum*vishidinc + self.epsilonw*((np.dot(data.T, ph_mean)-np.dot(nv_samples.T, nh_means))/(data.shape[0])-self.weightcost*self.W)
                visbiasinc = momentum*visbiasinc + self.epsilonvb*np.mean(data - nv_samples, axis=0)
                hidbiasinc = momentum*hidbiasinc + self.epsilonhb*np.mean(ph_mean - nh_means, axis=0)

                self.W += vishidinc
                self.vbias += visbiasinc
                self.hbias += hidbiasinc
            print('epoch{} -> error:{}'.format(epoch, errsum))
        
        return batchposhidprobs  

    '''
        we use linear in the output layer instead of sigmoid
        the learing rate is changed!
    '''  
    
    def rbmhiddenlinear(self, k=1, maxepoch=10):

        self.epsilonw = 0.001
        self.epsilonvb = 0.001
        self.epsilonhb = 0.001

        vishidinc = np.zeros((self.n_visible, self.n_hidden))
        visbiasinc = np.zeros(self.n_visible)
        hidbiasinc = np.zeros(self.n_hidden)
        batchposhidprobs = np.zeros((self.input.shape[0],self.n_hidden,self.input.shape[2]));

        for epoch in range(maxepoch):
            errsum = 0
            for batch in range(self.input.shape[2]):
                print('epoch: {} -> batch : {}....'.format(epoch,batch))
                data = self.input[:,:,batch]

                ph_mean, ph_sample = self.linear_hidden(data)
                
                batchposhidprobs[:,:,batch] = ph_mean
                chain_start = ph_sample

                for step in range(k):
                    if step == 0:
                        nv_means, nv_samples, nh_means, nh_samples = self.linear_gibbs_hvh(chain_start)
                    else:
                        nv_means, nv_samples, nh_means, nh_samples = self.linear_gibbs_hvh(nh_samples)
                
                err = np.sum((data - nv_means)**2)
                errsum += err


                if epoch > 4:
                    momentum = self.finalmomentum
                else:
                    momentum = self.initialmomentum

                vishidinc = momentum*vishidinc + self.epsilonw*((np.dot(data.T, ph_mean)-np.dot(nv_samples.T, nh_means))/(data.shape[0])-self.weightcost*self.W)
                visbiasinc = momentum*visbiasinc + self.epsilonvb*np.mean(data - nv_samples, axis=0)
                hidbiasinc = momentum*hidbiasinc + self.epsilonhb*np.mean(ph_mean - nh_means, axis=0)

                self.W += vishidinc
                self.vbias += visbiasinc
                self.hbias += hidbiasinc
            print('epoch{} -> error:{}'.format(epoch, errsum))
        
        return batchposhidprobs    


    '''
        v->sigmoid->state->sigmoid->state->sigmoid->state->sigmoid...
    '''

    def sample_h_given_v(self, v0_sample):
        h1_mean = sigmoid(np.dot(v0_sample, self.W) + self.hbias)
        h1_sample = np.random.binomial(n=1, p=h1_mean, size=h1_mean.shape)
        return [h1_mean, h1_sample]
        
    def sample_v_given_h(self, h0_sample):
        v1_mean = sigmoid(np.dot(h0_sample, self.W.T) + self.vbias)
        v1_sample = np.random.binomial(n=1, p=v1_mean, size=v1_mean.shape)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [v1_mean, v1_sample, h1_mean, h1_sample]

    '''
        Hnton: v->sigmoid->state->sigmoid->sigmoid->sigmoid...
    '''

    def nostate_gibbs_hvh(self, h0_sample):
        v1_mean = sigmoid(np.dot(h0_sample, self.W.T) + self.vbias)
        h1_mean = sigmoid(np.dot(v1_mean, self.W) + self.hbias)
        return [v1_mean, v1_mean, h1_mean, h1_mean]

    '''
        rbmhidlinear
    '''

    def linear_hidden(self, v0_sample):
        h1_mean = np.dot(v0_sample, self.W) + self.hbias
        h1_sample = h1_mean + np.random.normal(loc=0.0, scale=1.0, size=(h1_mean.shape[0],h1_mean.shape[1]))
        return [h1_mean, h1_sample]
    
    def linear_gibbs_hvh(self, h0_sample):
        v1_mean = sigmoid(np.dot(h0_sample, self.W.T) + self.vbias)
        h1_mean = np.dot(v1_mean, self.W) + self.hbias
        return [v1_mean, v1_mean, h1_mean, h1_mean]
