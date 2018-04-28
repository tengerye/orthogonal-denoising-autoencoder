import numpy as np
from scipy.optimize import minimize
from scipy import optimize

# array operations

class OrthAE(object):
    
    def __init__(self, views, latent_spaces, x = None, knob = 0):
        # x: input, column-wise
        # y: output, column-wise
        # h: hidden layer
        
        # views and latent_spaces: specify number of neurons in each view and latent space respectively
        # params: initial weights of the orthogonal autoencoders
        self.x = x
        self.knob = knob
        self.views = views
        self.latent_spaces = latent_spaces
        
        # sets of weights
        self.w1 = np.zeros((np.sum(latent_spaces), np.sum(views) + 1))
        self.w2 = np.zeros((np.sum(views), np.sum(latent_spaces) + 1))
        
        # add orthogonal constraints
        start_idx_hidden = np.cumsum(self.latent_spaces)
        start_idx_input = np.cumsum(np.append([0],self.views))
        
        # public latent space has fully connection
        self.w1[0:start_idx_hidden[0],:] = 1
        
        # private latent spaces only connect to respective views
        for i in range(latent_spaces.size-1):

            self.w1[start_idx_hidden[i]: start_idx_hidden[i+1], start_idx_input[i]: start_idx_input[i+1]] = 1
            
        self.w2[:, :-1] = np.transpose(self.w1[:, :-1])
        self.w2[:, -1] = 1
        
        # index of effective weigths
        self.index1 = np.where( self.w1 != 0 )
        self.index2 = np.where( self.w2 != 0 )
        

#         # randomly initialize weights
#         self.w1[self.index1] = randInitWeights(self.w1.shape)[self.index1]
#         self.w2[self.index2] = randInitWeights(self.w2.shape)[self.index2]

        
#         print self.w1
#         print self.w2
        
    def feedforward(self):
        instance_count = self.x.shape[1]
        bias = np.ones((1, instance_count))
        
        # a's are with bias
        self.a1 = np.concatenate(( self.x, bias ))

        # before activation

        self.h = np.dot(self.w1, self.a1)

        self.a2 = np.concatenate(( activate(self.h), bias ))
        self.y = np.dot(self.w2, self.a2)
        self.a3 = activate(self.y)
        
        return self.a3
    
    def backpropogateT(self, weights, y):
        self.weightSplit(weights)
        self.feedforward()
        
        return self.backpropogate(y)
 
    def backpropogate(self, y): 
        # knob is to prevent over-fitting
        instance_count = self.x.shape[1]
        delta = (self.a3 - y)/instance_count * activateGradient(self.y)
        

        self.w2_gradient = np.dot(delta, np.transpose(self.a2))
        
        # regularization
        self.w2_gradient[:, :-1] += np.dot(self.knob, self.w2[:, :-1])/instance_count

        delta = np.dot(np.transpose(self.w2[:, :-1]), delta) * activateGradient(self.h)
        self.w1_gradient = np.dot(delta, np.transpose(self.a1))
        
        # regularization
        self.w1_gradient[:, :-1] += np.dot(self.knob, self.w1[:, :-1])/instance_count
        
        return np.concatenate(( self.w1_gradient[self.index1].flatten(), self.w2_gradient[self.index2].flatten() ), axis=1)
        
    

    def costT(self, weights, y):

        self.weightSplit(weights)
        self.feedforward()
        return self.cost(y)


    def cost(self, y):
        
        instance_count = self.x.shape[1]
        result = np.sum(np.square(self.w1)) + np.sum(np.square(self.w2))
        
        result = np.sum(np.square(self.a3 - y)) + np.dot(self.knob, result)

        return result/2/instance_count
    
    
    
    
    def tuneT(self, y = None):
        
        if y is None:
            y=self.x
        
        # randomly initialize weights
        w1 = randInitWeights(self.w1.shape)[self.index1].flatten()
        w2 = randInitWeights(self.w2.shape)[self.index2].flatten()
        
        
        res = minimize(self.costT, np.concatenate((w1, w2), axis=1), args=(y,), method='CG',\
                        jac=self.backpropogateT, options={'disp': True, 'gtol': 1e-10, 'maxiter': 1e+1})
        
        return res.x
        
#         res = optimize.fmin_cg(self.cost, np.concatenate((w1, w2)), fprime=self.backpropogate,\
#                                args=self.x, gtol = 1e-10, disp = True)
#         minimize(self.cost,np.concatenate((self.w1, self.w2), axis=1),args=(self.x),method='CG',jac=self.backpropogate)



    def tune(self, y = None):
        if y is None:
            y=self.x

        # randomly initialize weights

        w1 = randInitWeights(self.w1.shape)[self.index1].flatten()
        w2 = randInitWeights(self.w2.shape)[self.index2].flatten()

        # set parameters for stochastic gradient descent
        x = self.x
        gtol = 1e-7
        maxiter = 2100
        w = np.concatenate((w1, w2), axis=1)
        step = 1.5e-8
 
        # stochastic sampling
        stochast_sample_count = 500
        sample_index = np.arange(stochast_sample_count)
        self.x = x[:, sample_index]
        stoch_y = y[:, sample_index]
        sample_index += stochast_sample_count

        # start gradient descent
        self.weightSplit(w)
        self.feedforward()

        iter = 0
        while iter < maxiter:

            weights_gradient = self.backpropogate(stoch_y)
            if np.max(weights_gradient) < gtol:
                break

            w -= weights_gradient * step
           
            self.x = x[:, sample_index]
            stoch_y = y[:, sample_index]
            sample_index = (sample_index + stochast_sample_count)%x.shape[1]
            
            self.weightSplit(w)
            self.feedforward()
            iter += 1

        print('iteration times:', iter)
        return w



    def weightSplit(self, weights):
        
        # weights is expected to be a row vector(narray)
#         weights = np.squeeze(weights)
        split = self.index1[0].size
        self.w1[self.index1] = weights[:split]
        
        self.w2[self.index2] = weights[split:]



#
class OrthdAE(OrthAE):
    
    def __init__(self, views, latent_spaces, x = None, knob = 0):
        m, n = x.shape
        
        ###################################################
       # trivial denoising
        x = np.tile(x, [1, m])
        self.denoise_x = x
        
        for i in range(m):
            x[i, i*n : i*n+n] = 0
            
        ###################################################
         # evenly denoising
       # x = np.tile(x, [1, 10])
       # self.denoise_x = x
       # 
       # step = int(m/10)
       # for i in range(9):
       #     x[i*step : i*step+step, i*n : i*n+n] = 0
       #    
       # x[9*step-m : , -n] = 0
        ################################################### 
        
        super(OrthdAE, self).__init__(views, latent_spaces, x, knob)
        
        
    def tune(self, y = None):
        
        if y is None:
            y=self.denoise_x
        
        return super(OrthdAE, self).tune(y)
    


# random initialize weights for AE
def randInitWeights(L_in, L_out):
    epsilon = np.sqrt(6.0 / (L_in + L_out))
    return np.random.rand(L_in, L_out) * 2 * epsilon - epsilon


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigGradient(x):
    g = sig(x)
    return g * (1.0 - g)


def tanh(x):
    return 2.0 /(1.0 + np.exp(-2*x)) - 1

def tanhGradient(x):
    g = tanh(x)
    return 1.0 - np.square(g)

def MSGD(x, y, f):
    pass
      
def activate(x):
    return sig(x)

def activateGradient(x):
    return sigGradient(x)  
