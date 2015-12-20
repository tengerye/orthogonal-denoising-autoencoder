import numpy as np
import orthAE as ae
import scipy as sp

# generate toy data for multi-view learning from paper "Factorized Latent Spaces with Structured Sparsity"
t = np.arange(-1, 1, 0.02)

x = np.sin(2*np.pi*t) # share latent space
x_noise = 0.02*np.sin(3.6*np.pi*t) # correlated noise


# private latent spaces
z1 = np.cos(np.pi*np.pi*t)
z2 = np.cos(5*np.pi*t)

# shared private spaces
m1 = np.vstack((x, z1));
m2 = np.vstack((x, z2));

m1 = np.dot(np.random.rand(20, 2), m1)
m2 = np.dot(np.random.rand(20, 2), m2)
# m1 = orth(rand(20, size(x, 1)+1))*m1;
# m2 = orth(rand(20, size(x, 1)+1))*m2;



# add gaussian noise with mean=0, standard deviation=0.01
m1 = m1 + np.random.randn(*m1.shape)*0.01; 
m2 = m2 + np.random.randn(*m2.shape)*0.01;


# add correlated noise
m1 = np.vstack((m1, x_noise))
m2 = np.vstack((m2, x_noise))

m1 = m1 - np.mean(m1, axis=0)
m2 = m2 - np.mean(m2, axis=0)

x2 = np.concatenate((m1, m2,))

print('start tuning')

obj2 = ae.OrthdAE(np.array([21, 21]), np.array([1, 1, 1]), x2)
new_weights = obj2.tune()

print('finish tuning')

import scipy as sp
import scipy.io


obj2.weightSplit(new_weights)
obj2.x = x2
obj2.feedforward()
sp.io.savemat('exp1-1.mat', mdict={'y':obj2.h, 't':t})
