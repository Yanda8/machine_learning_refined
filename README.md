# Machine Learning Refined Jupyter notebooks 

This repository contains supplementary Python files associated the texbook [Machine Learning Refined](http://www.mlrefined.com) published by Cambridge University Press, as well as a [blog made up of Jupyter notebooks](https://jermwatt.github.io/mlrefined/index.html) that was used to rough draft the second edition of the text.  To successfully run the Jupyter notebooks contained in this repo we highly recommend downloading the [Anaconda Python 3 distribution](https://www.anaconda.com/download/#macos).  Many of these notebooks also employ the Automatic Differentiator [autograd](https://github.com/HIPS/autograd) which can be installed by typing the following command at your terminal
      
      pip install autograd
      
With minor adjustment users can also run these notebooks using the GPU/TPU extended version of autograd  [JAX](https://github.com/google/jax).

Note: to pull a minimial sized clone of this repo (including only the most recent commit) use a shallow pull as follows
      
      git clone --depth 1 https://github.com/jermwatt/mlrefined.git
      
      
## [Blog contents](https://jermwatt.github.io/mlrefined/index.html)

### Zero order / derivative free optimization

[2.1  Motivation](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_0_Motivation.html)

[2.2 Zero order optimiality conditions](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_1_Zero.html)

[2.3 Global optimization](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_2_Global.html)

[2.4 Local optimization techniques](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_3_Local.html)

[2.5 Random search methods](https://jermwatt.github.io/mlrefined/blog_posts/2_Zero_order_methods/2_4_Random.html)

### First order optimization methods

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



--- 
This repository is in active development by [Jeremy Watt](mailto:jeremy@dgsix.com) and [Reza Borhani](mailto:reza@dgsix.com) - please do not hesitate to reach out with comments, questions, typos, etc.
