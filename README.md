# Machine Learning Refined Jupyter notebooks 

This repository contains supplementary Python files associated the texbook [Machine Learning Refined](http://www.mlrefined.com) published by Cambridge University Press, as well as a [blog made up of Jupyter notebooks](https://jermwatt.github.io/mlrefined/index.html) that was used to rough draft the second edition of the text.  To successfully run the Jupyter notebooks contained in this repo we highly recommend downloading the [Anaconda Python 3 distribution](https://www.anaconda.com/download/#macos).  Many of these notebooks also employ the Automatic Differentiator [autograd](https://github.com/HIPS/autograd) which can be installed by typing the following command at your terminal
      
      pip install autograd
      
With minor adjustment users can also run these notebooks using the GPU/TPU extended version of autograd  [JAX](https://github.com/google/jax).

Note: to pull a minimial sized clone of this repo (including only the most recent commit) use a shallow pull as follows
      
      git clone --depth 1 https://github.com/jermwatt/mlrefined.git
      
      
## [Blog contents](https://jermwatt.github.io/mlrefined/index.html)

### Chapter 2: Zero order / derivative free optimization

<a href="https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_0_Motivation.html" target="_blank">2.1  Motivation</a>

[2.1  Motivation](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_0_Motivation.html){:target="_blank"}

[2.1  Motivation](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_0_Motivation.html)  
[2.2 Zero order optimiality conditions](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_1_Zero.html)  
[2.3 Global optimization](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_2_Global.html)    
[2.4 Local optimization techniques](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_3_Local.html)   
[2.5 Random search methods](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_4_Random.html)
 
### Chapter 3: First order optimization methods

[3.1 Introduction](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_0_Introduction.html)   
[3.2 The first order optimzliaty condition](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_1_First.html)  
[3.3 The anatomy of lines and hyperplanes](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_2_Hyperplane.html)   
[3.4 Automatic differentiation and autograd](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_4_Automatic.html)   
[3.5 Gradient descent](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_5_Descent.html)   
[3.6 Two problems with the negative gradient direction](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_6_Problems.html)   
[3.7 Momentum acceleration](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_7_Momentum.html)   
[3.8 Normalized gradient descent procedures](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_8_Normalized.html)   
[3.9 Advanced first order methods](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_9_Advanced.html)   
[3.10 Mini-batch methods](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_10_Minibatch.html)   
[3.11 Conservative steplength rules](https://jermwatt.github.io/mlrefined/blog_posts/3_First_order_methods/3_11_Conservative.html)  

### Chapter 4: Second order optimization methods

4.1  The anatomy of quadratic functions   
[4.2 Curvature and the second order optimality condition](https://jermwatt.github.io/mlrefined/blog_posts/4_Second_order_methods/4_2_Second.html)   
[4.3 Newton's method](https://jermwatt.github.io/mlrefined/blog_posts/4_Second_order_methods/4_3_Newtons.html)   
[4.4 Two fundamental problems with Newton's method](https://jermwatt.github.io/mlrefined/blog_posts/4_Second_order_methods/4_4_Problems.html)   
4.5 Quasi-newton's methods 

### Chapter 5: Linear regression
[5.1 Least squares regression](https://jermwatt.github.io/mlrefined/blog_posts/5_Linear_regression/5_1_Least.html)   
[5.2 Least absolute deviations](https://jermwatt.github.io/mlrefined/blog_posts/5_Linear_regression/5_2_Absolute.html)   
[5.3 Regression metrics](https://jermwatt.github.io/mlrefined/blog_posts/5_Linear_regression/5_3_Metrics.html)   
[5.4 Weighted regression](https://jermwatt.github.io/mlrefined/blog_posts/5_Linear_regression/5_4_Weighted.html)   
[5.5 Multi-output regression](https://jermwatt.github.io/mlrefined/blog_posts/5_Linear_regression/5_5_Multi.html)  

### Chapter 6: Linear two-class classification

[6.1 Logistic regression and the cross-entropy cost](https://jermwatt.github.io/mlrefined/blog_posts/6_Linear_twoclass_classification/6_1_Cross_entropy.html)   
[6.2 Logistic regression and the softmax cost](https://jermwatt.github.io/mlrefined/blog_posts/6_Linear_twoclass_classification/6_2_Softmax.html)   
[6.3 The perceptron](https://jermwatt.github.io/mlrefined/blog_posts/6_Linear_twoclass_classification/6_3_Perceptron.html)   
[6.4 Support vector machines](https://jermwatt.github.io/mlrefined/blog_posts/6_Linear_twoclass_classification/6_4_SVMs.html)   
[6.5 Categorical labels](https://jermwatt.github.io/mlrefined/blog_posts/6_Linear_twoclass_classification/6_5_Categorical.html)   
[6.6 Comparing two-class schemes](https://jermwatt.github.io/mlrefined/blog_posts/6_Linear_twoclass_classification/6_6_Comparison.html)   
[6.7 Quality metrics](https://jermwatt.github.io/mlrefined/blog_posts/6_Linear_twoclass_classification/6_7_Metrics.html)   
[6.8 Weighted two-class classification](https://jermwatt.github.io/mlrefined/blog_posts/6_Linear_twoclass_classification/6_8_Weighted.html)  

### Chapter 7: Linear multi-class classification

[7.1 One-versus-All classification](https://jermwatt.github.io/mlrefined/blog_posts/7_Linear_multiclass_classification/7_1_OvA.html)   
[7.2 The multi-class perceptron](https://jermwatt.github.io/mlrefined/blog_posts/7_Linear_multiclass_classification/7_2_Perceptron.html)   
[7.3 Comparing multi-class schemes](https://jermwatt.github.io/mlrefined/blog_posts/7_Linear_multiclass_classification/7_3_Comparison.html)   
[7.4 The categorical cross-entropy cost](https://jermwatt.github.io/mlrefined/blog_posts/7_Linear_multiclass_classification/7_4_Categorical.html)   
[7.5 Multi-class quality metrics](https://jermwatt.github.io/mlrefined/blog_posts/7_Linear_multiclass_classification/7_5_Metrics.html)  


### Chapter 8: Unsupervised learning

[8.1 Spanning sets and vector algebra](https://jermwatt.github.io/mlrefined/blog_posts/8_Linear_unsupervised_learning/8_1_Spanning.html)   
[8.2 Learning proper spanning sets](https://jermwatt.github.io/mlrefined/blog_posts/8_Linear_unsupervised_learning/8_2_PCA.html)   
[8.3 The linear Autoencoder](https://jermwatt.github.io/mlrefined/blog_posts/8_Linear_unsupervised_learning/8_3_Autoencoder.html)   
[8.4 The class PCA solution](https://jermwatt.github.io/mlrefined/blog_posts/8_Linear_unsupervised_learning/8_4_Classic.html)   
[8.5 Recommender systems](https://jermwatt.github.io/mlrefined/blog_posts/8_Linear_unsupervised_learning/8_5_Recommender.html)  
[8.6 K-means clustering](https://jermwatt.github.io/mlrefined/blog_posts/8_Linear_unsupervised_learning/8_6_Kmeans.html)   
[8.7 Matrix factorization techniques](https://jermwatt.github.io/mlrefined/blog_posts/8_Linear_unsupervised_learning/8_7_Factorization.html)  

### Chapter 9: Principles of feature selection and engineering

[9.1 Histogram-based features](https://jermwatt.github.io/mlrefined/blog_posts/9_Feature_engineer_select/9_1_Histogram.html)   
[9.2 Standard normalization and feature scaling](https://jermwatt.github.io/mlrefined/blog_posts/9_Feature_engineer_select/9_2_Scaling.html)   
[9.3 Imputing missing values](https://jermwatt.github.io/mlrefined/blog_posts/9_Feature_engineer_select/9_3_Cleaning.html)   
[9.4 PCA-sphereing](https://jermwatt.github.io/mlrefined/blog_posts/9_Feature_engineer_select/9_4_PCA_sphereing.html)   
[9.5 Feature selection via boosting](https://jermwatt.github.io/mlrefined/blog_posts/9_Feature_engineer_select/9_5_Boosting.html)   
[9.6 Feature selection via regularization](https://jermwatt.github.io/mlrefined/blog_posts/9_Feature_engineer_select/9_6_Regularization.html)  

### Chapter 10: Introduction to nonlinear learning

[10.1 Nonlinear regression](https://jermwatt.github.io/mlrefined/blog_posts/10_Nonlinear_intro/10_1_Regression.html)  
[10.2 Nonlinear multi-output regression](https://jermwatt.github.io/mlrefined/blog_posts/10_Nonlinear_intro/10_2_MultReg.html)  
[10.3 Nonlinear two-class classification](https://jermwatt.github.io/mlrefined/blog_posts/10_Nonlinear_intro/10_3_Twoclass.html)  
[10.4 Nonlinear multi-class classification](https://jermwatt.github.io/mlrefined/blog_posts/10_Nonlinear_intro/10_4_Multiclass.html)  
[10.5 Nonlinear unsupervised learning](https://jermwatt.github.io/mlrefined/blog_posts/10_Nonlinear_intro/10_5_Unsupervised.html)  

### Chapter 11: Principles of feature learning

[11.1 Universal approximation](https://jermwatt.github.io/mlrefined/blog_posts/11_Feature_learning/11_1_Universal.html)  
[11.2 The bias-variance trade-off](https://jermwatt.github.io/mlrefined/blog_posts/11_Feature_learning/11_2_BiasVariance.html)  
[11.3 Cross-validation via boosting](https://jermwatt.github.io/mlrefined/blog_posts/11_Feature_learning/11_3_Boosting.html)  
[11.4 Cross-validation via regularization](https://jermwatt.github.io/mlrefined/blog_posts/11_Feature_learning/11_4_Regularization.html)  
[11.5 Ensembling techniques](https://jermwatt.github.io/mlrefined/blog_posts/11_Feature_learning/11_5_Ensembling.html)  
[11.6 K-folds cross-validation](https://jermwatt.github.io/mlrefined/blog_posts/11_Feature_learning/11_6_Kfolds.html)  
[11.7 Testing data](https://jermwatt.github.io/mlrefined/blog_posts/11_Feature_learning/11_7_Testing.html)  


### Chapter 12: Kernels

12.1 The variety of kernel-based learners    
12.2 The kernel trick    
12.3 Kernels as similarity measures    
12.4 Scaling kernels    
  
### Chapter 13: Fully connected networks / multi-layer perceptrons

[13.1 Fully connected networks](https://jermwatt.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_1_Multi_layer_perceptrons.html)  
[13.2 Optimization issues](https://jermwatt.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_2_Optimization.html)  
13.3 Activation functions  
13.4 Backpropogation  
[13.5 Batch normalization](https://jermwatt.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_5_Batch_normalization.html)  
[13.6 Early-stopping](https://jermwatt.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_6_early_stopping.html)  

### Chapter 14: Tree-based learners

14.1 Varieties of tree-based learners  
14.2 Regression trees  
14.3 Classification trees  
14.4 Gradient boosting  
14.5 Random forests  
14.6 Cross-validating individual trees  


--- 
This repository is in active development by [Jeremy Watt](mailto:jeremy@dgsix.com) and [Reza Borhani](mailto:reza@dgsix.com) - please do not hesitate to reach out with comments, questions, typos, etc.
