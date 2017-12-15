## ch3: Tensorflow 与神经网络

### lectures

* 使梯度下降程序更通用
* Tensorflow 导论
* tf 实现线性回归
	* tensorboard: 需要为 docker 多映射一个端口
	* 三种梯度下降法：batch, stochastic, mini-batch
	* tf.placeholder
* 分类问题与逻辑回归
* 决策界面与神经网络

### tasks

* task1: 使用 tensorflow 实现线性回归的随机梯度下降法
	* 见 [tf_linear_sgd.ipynb](tf_linear_sgd.ipynb)
* task2: 解释交叉熵的优化效果
	* 记 sigmoid(z) = s(z) 
	* 以差值平方作为损失函数时，梯度中包含 s’(z) 项，其后果是：当预测误差很大时，梯度非常小，导致最初的迭代步中学习速度很慢。 
	* 若以交叉熵为损失函数，因为它「神奇」的数学特性，偏导数中刚好不包含 s’(z) 项，而且与预测误差 (s(z) - y) 成正比。这样就不存在上面的学习速度减慢的问题。预测误差越大，初期学习速度越快。 
* task3: 实现无隐层和单隐层神经网络
    - 见 [simple_neural_network.ipynb](simple_neural_network.ipynb)

