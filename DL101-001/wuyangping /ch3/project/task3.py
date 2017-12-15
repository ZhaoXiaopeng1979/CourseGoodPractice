
# coding: utf-8

# In[168]:

get_ipython().magic(u'load_ext watermark')
get_ipython().magic(u'watermark')


# In[286]:

import tensorflow as tf
import numpy as np
import os
print(tf.__version__)


# In[287]:

get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[288]:

csv_file = './../code/data.csv'


# In[289]:

def load_csv_data(file_name):
    dataset = []
    labels = []
    if not os.path.exists(file_name):        
        print csv_file, 'is not existing!'
        return dataset, labels
    
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:            
            items = line.strip('\n').split(',')            
            assert(len(items) == 3)
            dataset.append(items[:2])
            labels.append(items[-1:])
    return dataset, labels
            
train_dataset, train_labels = load_csv_data(csv_file)

num_labels = 2
num_input = 2
data_size = len(train_dataset)

train_dataset = np.array(train_dataset)
train_labels = np.array(train_labels)

print(train_dataset.shape)
print(train_labels.shape)

print(train_dataset[:10])
print(train_labels[:10])


# In[290]:

plt.scatter(train_dataset[:,0],train_dataset[:,1],c=train_labels, alpha=0.8)


# In[291]:

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 2)).astype(np.float32)
    
    #create one-hot matrix
    lables_matrix = np.zeros((data_size, num_labels))
    
    i = 0
    for label in labels:
        hot_index = 0
        tmp = float(label[0])
        if (tmp >0.1):
            hot_index = 1
        lables_matrix[i][hot_index] = 1.0
        i += 1
            
    return dataset, lables_matrix

train_dataset, train_labels = reformat(train_dataset, train_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print(train_labels[:10])


# In[292]:

def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[293]:

X = train_dataset

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

print xx.shape, yy.shape

test_dataset = np.c_[xx.ravel(), yy.ravel()]
print test_dataset.shape
#test_dataset = test_dataset.reshape((-1, 2)).astype(np.float32)

testdata_size = test_dataset.shape[0]
print testdata_size


# In[294]:

tf.reset_default_graph()


# In[295]:

#No hidden layer
batch_size = 100

graph1 = tf.Graph()
with graph1.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_input))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))  
    tf_test_dataset = tf.placeholder(tf.float32, shape=(testdata_size, num_input))
    
    weights = tf.Variable(tf.truncated_normal([num_input, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))   
    
    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.sigmoid(logits)
    test_prediction = tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights) + biases)
    


# In[296]:

num_steps = 10000

with tf.Session(graph=graph1) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (data_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_test_dataset : test_dataset}
        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if ((step+1) % 1000 == 0):
            print("Minibatch loss at step %d: %f" % (step+1, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
    feed_dict = {tf_test_dataset : test_dataset}
    test_predict1 = sess.run(test_prediction, feed_dict=feed_dict)
     


# In[297]:

Z = np.argmax(test_predict1, 1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(train_dataset[:,0], train_dataset[:,1],c=np.argmax(train_labels, 1), alpha=0.8)


# In[310]:

#With hidden layer
batch_size = 100
num_hidden = 10


tf.reset_default_graph()

graph2 = tf.Graph()
with graph2.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_input))    
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))   
    
    tf_test_dataset = tf.placeholder(tf.float32, shape=(testdata_size, num_input))
    
    weights1 = tf.Variable(tf.truncated_normal([num_input, num_hidden]))
    biases1 = tf.Variable(tf.zeros([num_hidden]))   
    
    logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
    hidden = tf.nn.tanh(logits1)
    
    weights2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels])) 
    
    # Training computation.
    logits2 = tf.matmul(hidden, weights2) + biases2
    loss2 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits2))
        #tf.losses.sigmoid_cross_entropy(labels=tf_train_labels, logits=logits2))
  
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss2)
    
    # Predictions for the training and test data.
    train_prediction = tf.nn.sigmoid(logits2)
    
    test_logits1 = tf.matmul(tf_test_dataset, weights1) + biases1
    test_logits2 = tf.matmul(tf.nn.tanh(test_logits1), weights2) + biases2
    test_prediction = tf.nn.sigmoid(test_logits2)


# In[311]:

num_steps = 10000

with tf.Session(graph=graph2) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (data_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = sess.run(
            [optimizer, loss2, train_prediction], feed_dict=feed_dict)

        if ((step+1) % 1000 == 0):
            print("Minibatch loss at step %d: %f" % (step+1, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

    feed_dict = {tf_test_dataset : test_dataset}
    test_predict2 = sess.run(test_prediction, feed_dict=feed_dict)
  


# In[313]:

Z = np.argmax(test_predict2, 1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(train_dataset[:,0], train_dataset[:,1],c=np.argmax(train_labels, 1), alpha=0.8)


# In[309]:

writer = tf.summary.FileWriter("/root/log", graph=tf.get_default_graph())


# In[ ]:



