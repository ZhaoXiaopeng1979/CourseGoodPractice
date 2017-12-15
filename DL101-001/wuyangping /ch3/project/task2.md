# 任务二：解释交叉熵的优化效果

1. 当使用Quadratic cost fucntion时，  
  ![Quadratic cost function](https://github.com/wuyangping/DeepLearning101/blob/master/ch3/note/figures/0.png)  

  那么，W, b的梯度分别为：  

  ![梯度](https://github.com/wuyangping/DeepLearning101/blob/master/ch3/note/figures/1.png)  

  其中，w和b的梯度都包含sigmoid函数的一阶导数项，根据sigmod函数的图形：     

  ![Sigmoid](https://github.com/wuyangping/DeepLearning101/blob/master/ch3/note/figures/2.png)   

  可见，sigmoid函数在y接近0或1时曲线都非常平缓，一阶导数值接近于0，因此使用Quadratic cost function会导致在y接近0 或1的情况下，梯度下降非常缓慢，出现饱和，学习速率太低的问题。  

2. 当使用交叉熵cost function时,   

  ![cross-entropy cost function](https://github.com/wuyangping/DeepLearning101/blob/master/ch3/note/figures/3.png)   

  对w, b求偏导，得到梯度公式如下：   

  ![w](https://github.com/wuyangping/DeepLearning101/blob/master/ch3/note/figures/4.png)  

  ![b](https://github.com/wuyangping/DeepLearning101/blob/master/ch3/note/figures/5.png)   

  w和b的梯度中已经没有sigmoid的导数项，只与预测值与实际值的偏差项有关，因此，在y接近0或1时，仍然能够较快收敛，降低cost，找到最优参数。  

  因此，相对于quadratic成本函数，交叉熵可以解决学习速率太低的问题，避免在y接近0或1的时候出现饱和，更快实现收敛。
