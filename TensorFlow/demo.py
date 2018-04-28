import matplotlib
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io

from sklearn.cross_decomposition import CCA
from orthAE import OrthdAE


# generate toy data for multi-view learning from paper "Factorized Latent Spaces with Structured Sparsity"
t = np.arange(-1, 1, 0.02)

x = np.sin(2*np.pi*t) # share latent space
x_noise = 0.02*np.sin(3.6*np.pi*t) # correlated noise


# private latent spaces
z1 = np.cos(np.pi*np.pi*t)
z2 = np.cos(5*np.pi*t)

##########################################################
# Fig.2.(a)

f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(t, x, color='blue')
axarr[0].plot(t, z1, color='green')
axarr[0].plot(t, x_noise, color='red')
axarr[0].set_title('Fig.2.(a)')

axarr[1].plot(t, x, color='blue')
axarr[1].plot(t, z2, color='green')
axarr[1].plot(t, x_noise, color='red')
plt.show()
##########################################################

# shared private spaces
m1 = np.vstack((x, z1));
m2 = np.vstack((x, z2));

m1 = np.random.rand(20, 2).dot(m1)
m2 = np.random.rand(20, 2).dot(m2)
# m1 = np.matmul(np.random.rand(20, 2), m1)
# m2 = np.matmul(np.random.rand(20, 2), m2)
# m1 = np.dot(np.random.rand(20, 2), m1)
# m2 = np.dot(np.random.rand(20, 2), m2)


# add gaussian noise with mean=0, standard deviation=0.01
m1 = m1 + np.random.randn(*m1.shape)*0.01; 
m2 = m2 + np.random.randn(*m2.shape)*0.01;


# add correlated noise
m1 = np.vstack((m1, x_noise))
m2 = np.vstack((m2, x_noise))


##########################################################
# Fig.2.(b)

f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(t, m1.transpose())
axarr[0].set_title('Fig.2.(b)')
axarr[1].plot(t, m2.transpose())

plt.show()
##########################################################
# Fig.3 CCA
cca = CCA(n_components=3)
cca.fit(m1.T, m2.T)

X_c = cca.transform(m1.T)

fig, ax = plt.subplots()
ax.set_title('Fig.2.(c)')
# ax.set_color_cycle(['blue', 'green', 'red'])
ax.set_prop_cycle('color', ['blue', 'red', 'green'])
ax.plot(X_c)
# ax.plot(Y_c)
plt.show()

##########################################################
# Use TensorFlow.
x2 = np.concatenate((m1, m2,))
# y2 = trivial_denoising(x2)
# print('shape of y2=', np.shape(y2))

odae = OrthdAE([21, 21], [1, 1, 1])
odae.train(x2.T, max_iter=1000000)
result = odae.transform(x2.T)
plt.plot(result)
plt.show()
