# Orthogonal Denoising Autoencoders
Orthogonal denoising autoencoders are for multi-view learning problem (a.k.a. multimodal, model fusion). Data samples are collected from diverse sensors or described by various features and inherently have multiple disjoint feature sets.

The project is extensive light-weight, which is an implementation of [the MMM2016 best paper](https://www.researchgate.net/publication/291520610_Learning_Multiple_Views_with_Orthogonal_Denoising_Autoencoders).
------------------------------------------------------------------------

## Getting started
Each package is an implementation using different languages (frameworks). Therefore, you can use codes within a package independently and install corresponding dependent libraries.

### Contents
* ```python```: implementation uses numpy and scipy to implement feed-forward, back-propogation and stochastic gradient descent.
     * ```orthdAE```:implementation BLABLABLA.
* ```tensorflow```: implementation is build on the [TensorFlow](https://github.com/tensorflow/tensorflow). So the details mentioned above are controlled by TensorFlow.

### Dependencis
* ```python```: 2.7.13
* ```TensorFlow```: 1.3.0
* ```numpy```: 1.11.3
* ```scipy```: 0.18.1

### Notice
* The size of input for Python code is: NUM_OF_FEATURE x NUM_OF_EXAMPLE. The size of placeholder for TensorFlow code is: NUM_OF_EXAMPLE x NUM_OF_FEATURE.

## Demo
Each package contains a demo (demo.py) which implements the first experiment of [the MMM2016 best paper](https://www.researchgate.net/publication/291520610_Learning_Multiple_Views_with_Orthogonal_Denoising_Autoencoders).


In order to run the demo, go to the corresponding directory and run the demos in shell:
```shell
python demo.py
```
You will see 3 figures pop up and then the terminal will print training process. In the end, the result of our model will pop up as figure.

### Python implementation
Finish.
### TensorFlow implementation
Finish.
### Pytorch implementation
In progress.

------------------------------------------------------------------------

## Acknowledgments
Orthogonal denoising autoencodrs are proposed in the paper, *Learning Multiple Views with Orthogonal Denoising Autoencoders*, which won the best paper award in MMM2016.
![Certificate](./other/MMM2016.png)

If you have an question or suggestion, please contact the author: TengQi Ye(yetengqi@gmail.com). Your citation will be my biggest inspiration.

> BibTeX Style Citation

```
@inproceedings{ye2016learning,
  title={Learning multiple views with orthogonal denoising autoencoders},
  author={Ye, TengQi and Wang, Tianchun and McGuinness, Kevin and Guo, Yu and Gurrin, Cathal},
  booktitle={International Conference on Multimedia Modeling},
  pages={313--324},
  year={2016},
  organization={Springer}
}
