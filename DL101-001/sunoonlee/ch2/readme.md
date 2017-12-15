## ch2: 分类器模型

* [x] task1: 利用贝叶斯公式，说明为什么 $P(y|w_1,w_2) \neq P(y|w_1)P(y|w_2)$ （即使做了独立假设）
  - 先约定 $P(y) < 1$
  + 再假设 $w_1$ 与 $w_2$ 独立且关于 y 条件独立 (注意独立与条件独立不是一回事)
  - 于是, $P(y|w_1,w_2) = P(w_1,w_2,y) / P(w_1,w_2) = P(w_1|y) P(w_2|y) P(y) / (P(w_1) x P(w_2)) > P(w_1,y) / P(w_1) x P(w_2,y) / P(w_2) = P(y|w_1)P(y|w_2)$
* [x] task2: 朴素贝叶斯法-实现 [naive_bayes_implementation](naive_bayes_implementation.ipynb)
* [x] 朴素贝叶斯笔记: [naive_bayes.md](naive_bayes.md)
* [x] task3: 线性回归的梯度下降法-实现 [gradient_descent](gradient_descent.ipynb)
* [] 线性模型笔记
