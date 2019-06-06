# COMP5328 Assignment 2

## Abstract

In this paper, our target is to solve the label noise problem in machine learning which is to improve the performance on test data when training classifier on label noise dataset. Two datasets have been used for training and testing, which are modified Fashion-MNIST and CIFAR. We implement two methods which are importance reweighting approach and corrected loss approach and do comparison in terms of the performance on test data with all correct labels and the robustness of these two methods. We firstly introduce the problem and give an overview of our implementations. Then we review the related work, illustrate our methods. In experiments section, we make very detailed analysis of two methods and related setup. 

## Introduction

In reality, because of some reasons, such as the subjective labelling task[1] , some labels  in training data in the area of supervised learning may incorrect which is represented as label noise. People cannot guarantee the completely correctness of labels because its both expensive and difficult so that labels noise becomes an important problem[1]. For instance, it is impossible to ensure that the medical diagnosis are 100 percent accurate [2-4].Numerous studies shows that there is a significant negative impact of classification performances caused by label noise [5]. Due to unreliable of the accuracy of label and its destructive effect on classification performance, some techniques need to be developed to reduce the consequences of it[6].

In the literature, we focus on class-dependent label noise which means that the flip probability depends on the class[7]. Two approaches have been developed. We firstly do feature scaling on both training and test data. Then we build importance reweighting approach, using the probabilistic classification method to estimate the conditional probability of label noise given training data and estimate flip rates by chosen two minimum values of the probability. Another approach is corrected loss, which build a three-layer CNN with the backward correction procedure and the forward correction procedure.  To ensure fairness we randomly select 8,000 training data using bootstrapping method 10 times and calculate the average metrics. 

## Related Work

Researchers create a lot of methods to deal with label noise. Natarajan et al. [7] consider two approaches. One is to modify the loss function to its unbiased estimation. They do a lot of works to ensure that the proxy loss is convex whether in a symmetry condition or not. The other approach is to provide a weighted loss function as its proxy loss function. This method is based on study by Scott in 2012. These two methods increase the performance a lot in some datasets with some specific classifiers. Methods can be used widely for any given surrogate loss function and some methods, for example, biased SVM and weighted logistic regression shows noise-tolerant. However, in this literature, flip rates are observable, which may cause some problem when we cannot observe them.

Xiao et al. [8] studied on real word image dataset. They introduced two types of label noise based on observations as one latent variable. Based on it, they create a novel probabilistic model which can illustrate the relationship between images, noise labels, noise types, ground truth labels and integrate it into a CNN framework. They explored some strategies for better performance. The main idea of this research is to use the novel probabilistic model supervise the training of the network. On CIFAR 10 dataset, their method leads to a good performance when noise level is less or equal to 40 percent. However, the performance decrease when noise level is 50 percent. 

Lawrence & Scholkopf [9] also do some research on probabilistic model of label noise, they try to optimize the parameters by using expectation-maximization (EM) algorithm. They optimize the log-likelihood with an additional item $$ H(Q(y|\mathbf{x}, \tilde{y})) $$ where $$ H(P(\cdot)) $$ is the entropy of $$ p(\cdot) $$, $$ Q(y|\mathbf{x}, \tilde{y}) = P(y|\mathbf{x}, \tilde{y}, \boldsymbol{\theta}) $$. Unlabelled data can be used with this approach. In contrast, the approach has some disadvantages, such as the cross validation method is useless when implementing this approach.

Sukhbaatar & Fergus [10] build a modified deep neural networks which can generate good results in practice. They add one additional linear layer on top of the softmax layer which can fix the output to match the noisy label better. 

Li et al. [11] tried to use a different method which inspired by the concept of "distillation" on label noise. They purpose to build a textual knowledge graph $$g$$ based on Wikipedia which form of a matrix $$G \in R_{+}^{L  \times  L}$$, each item of $G$ illustrate the relationship between two labels.  A distillation process which learned from a small clean dataset is used to promote the performance of the classifier. Knowledge graph is a guideline of distillation. Their system shows impressive results on test dataset. However, it has some limitations. First is that building the knowledge graph relies on a small "clean" dataset. The other constraint is that the relationship between two labels need to be clear, and the definition of labels on Wikipedia need to be reliable.

## Methods

### Data preprocessing

We firstly reshape the original data, random select 8,000 training data and make train MINST data to shape (8000, 28, 28, 1), validation MINST data to (2000, 28, 28, 1). Similar reshape process is implemented on CIFAR dataset which lead to (8000, 32, 32, 3) as the shape of training data and (2000, 32, 32, 3) as the shape of validation data.

Then we implement feature scaling method on all training, validation and testing data, with the following formula:
$$
X^{'} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$
where $$ X^{'} $$ is the normalized dataset, $$ X_{min} $$ is the matrix of the minimum value of dataset which is 0 in our dataset and  $$ X_{max} $$ is the matrix of the maximum value of dataset which is 255 in our dataset.

### Evaluation methods

Effectiveness and robustness of our implementation need to be evaluated. For effectiveness, we use top-1 accuracy which set the maximum probability prediction as result, if the prediction is equal to the true, we assign it as one correctly classified example. The top-1 accuracy is calculated by using the number of correctly classified examples except the total number of test examples. The formula of top-1 accuracy is shown below:
$$
accuracy = \frac{num(TP)}{num(total)}\times100%
$$
where $$ num(TP) $$ is the number of correctly classified examples, $$ num(totoal) $$ is the total number of test examples.

For robustness, we training classifiers using different training data 10 times (the strategy of extract samples is described in section 4.1 in detail) and calculate the loss of test data. The standard derivation of loss and accuracy can shows the robustness. If the approach has a very good robustness, each classifier of this approach will result a very similar loss or accuracy which lead to a small value of standard derivation. standard derivation and robustness are negatively correlated.

## Experiment

### Dataset

Original datasets which modified on Fashion-MNIST and CIFAR dataset have two parts. The first part is 10,000 training data with the same number of noisy label. The label noise is class-dependent. The flip rates of it are $$ P(S = 1 | Y = 0) = 0.2, P(S = 0 | Y = 1) = 0.4 $$, where S is the variable of noisy labels and Y is the variable of true labels. Each image of Fashion-MNIST dataset has been reshape to a 784 dimension vector and each image of CIFAR dataset has been reshape to a 3072 dimension vector. The second part is 2,000 test data, it has been reshaped as same as training data. All labels of test data are "clean".

To split the data, we use bootstrap method, one sample has same probability on each sampling. We randomly sample 8,000 samples as training data and set the other 2,000 as validation data. Then we train the classifier and get the metrics on test data. This several steps repeat 10 times so that at each time, we have a difference distribution of samples and different metrics.

## Conclusion

In this paper, we implement importance reweighting framework and corrected loss framework on Fashion-MNIST and CIFAR dataset. Both two approaches show better accuracy improvement on CIFAR dataset and show better accuracy on Fashion-MNIST dataset. The robustness of corrected loss approach is a little worse than importance reweighting. However, when corrected loss method implemented on more complex dataset (CIFAR), it improve the accuracy a lot. Totally, in our experiments, corrected loss method is more suitable for complex image dataset, importance reweighting method has better robustness.

In future work, we firstly need to import more dataset with different percentage of label noise to prove our options exactly. These two methods cannot predict the flip rate very well, we need to do future analysis on how to learn flip rate and conditional probability distributiWon accurately.

## Reference

1. Frénay, B., & Kabán, A. (2015). Special issue on advances in learning with label noise. *Neurocomputing*, *160*, 1-2. doi: 10.1016/j.neucom.2015.01.067
2. I. Bross, “Misclassification in 2 x 2 tables,” *Biometrics*, vol. 10, no. 4, pp. 478–486, 1954.
3. L. Joseph, T. W. Gyorkos, and L. Coupal, “Bayesian estimation of disease prevalence and the parameters of diagnostic tests in the absence of a gold standard,” *Am. J. Epidemiol*., vol. 141, no. 3, pp. 263–272, 1995.
4. A. Hadgu, “The discrepancy in discrepant analysis,” *The Lancet*, vol.348, no. 9027, pp. 592–593, 1996.
5. X. Zhu and X. Wu, “Class noise vs. attribute noise: A quantitative study,” *Artif. Intell. Rev.*, vol. 22, pp. 177–210, 2004.
6. Frénay, B., & Verleysen, M. (2014). Classification in the presence of label noise: a survey. *IEEE transactions on neural networks and learning systems*, *25*(5), 845-869.
7. Natarajan, N., Dhillon, I. S., Ravikumar, P. K., & Tewari, A. (2013). Learning with noisy labels. In *Advances in neural information processing systems* (pp. 1196-1204).
8. Xiao, T., Xia, T., Yang, Y., Huang, C., & Wang, X. (2015). Learning from massive noisy labeled data for image classification. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2691-2699).
9. Lawrence, N. D., & Schölkopf, B. (2001, June). Estimating a kernel Fisher discriminant in the presence of label noise. In *ICML* (Vol. 1, pp. 306-313).
10. Sukhbaatar, S., & Fergus, R. (2014). Learning from noisy labels with deep neural networks. *arXiv preprint arXiv:1406.2080*, *2*(3), 4.
11. Li, Y., Yang, J., Song, Y., Cao, L., Luo, J., & Li, L. J. (2017, March). Learning from Noisy Labels with Distillation. In *ICCV*(pp. 1928-1936).