# orthogonal-denoising-autoencoder
Implementation and improvement of algorithm from paper 'Learning Multiple Views with Orthogonal Denoising Autoencoders'. 

Multi-view learning techniques are necessary when data is described by multiple distinct feature sets because single-view learning algorithms tend to overfit on these high-dimensional data. Prior successful approaches followed either consensus or complementary principles. Recent work has focused on learning both the shared and private latent spaces of views in order to take advantage of both principles. However, these methods can not ensure that the latent spaces are strictly independent through encouraging the orthogonality in their objective functions. Also little work has explored representation learning techniques for multi-view learning. In this paper, we use the denoising autoencoder to learn shared and private latent spaces, with orthogonal constraints -- disconnecting every private latent space from the remaining views. Instead of computationally expensive optimization, we adapt the backpropagation algorithm to train our model.

To run the first experiment: python exp1
