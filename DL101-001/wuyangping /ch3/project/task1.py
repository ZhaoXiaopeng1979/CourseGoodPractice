
# coding: utf-8

# In[1]:

get_ipython().magic(u'load_ext watermark')
get_ipython().magic(u'watermark')


# In[2]:

import tensorflow as tf
import numpy as np
print(tf.__version__)


# In[3]:

get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt


# In[4]:

def generate_weight_and_bias(input_dim):
    # 1 x input_dim 行向量，相当于课程中的 w^T
    w = tf.Variable(tf.random_uniform([1, input_dim], -1, 1))
    # b, 1 * 1
    bias = tf.Variable(tf.zeros([1, 1]))
    return (w, bias)


# In[6]:

w0 = np.random.rand()
b0 = 0

print w0, b0


# In[7]:

x = np.concatenate((np.random.rand(1, 50), np.random.rand(1, 50) + 1), axis=1)
x = x.reshape(100,1)

noise = np.random.normal(size=len(x)).reshape(x.shape) * 0.1
y = w0 * x + b0 + noise

print x.shape
print y.shape
print noise.shape


# In[8]:

plt.scatter(x, y)


# In[9]:

tf.reset_default_graph()


# In[10]:

x_train = tf.placeholder('float32')
y_train = tf.placeholder('float32')


# In[12]:

w, b = generate_weight_and_bias(1)
print w
print b


# In[13]:


y_bar = w * x_train + b

cost = tf.reduce_mean(tf.square(y_bar - y_train))


# In[14]:

mini_batch_size = 1


# In[15]:

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

costs = []
w_out = 0
b_out = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    offset = 0
    epoch = 500
    data_size = len(x) 
    iteration = epoch * data_size / mini_batch_size
    for i in range(iteration):
        offset = (i * mini_batch_size) % data_size 
        x_mini_batch = x[offset : offset + mini_batch_size]
        y_mini_batch = y[offset : offset + mini_batch_size]
        feed_dict = {x_train:x_mini_batch, y_train:y_mini_batch}
        sess.run(train_step, feed_dict=feed_dict)  
        cost_i = sess.run(cost, feed_dict=feed_dict)
        costs.append(cost_i)
        if i % 500 == 0:
            print i
            w_out = sess.run(w)
            b_out = sess.run(b)
            print cost_i
            print w_out
            print b_out


# In[16]:

y_pred = x * w_out + b_out

print costs[-1:]
print w0, b0

print w_out, b_out


# In[17]:

plt.plot(x, y_pred)
plt.scatter(x,y)


# In[18]:

cnt = len(costs)
plt.plot(range(len(costs[:cnt])), costs[:cnt])


# In[19]:

writer = tf.summary.FileWriter("/root/log", graph=tf.get_default_graph())


# In[ ]:



