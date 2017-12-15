
# coding: utf-8

# In[30]:

import tensorflow as tf
import sys
import os
from model import Model
from csdataset import MyDataset
from network import *
from datetime import datetime
from scipy import ndimage
import scipy
from matplotlib import pyplot as plt
import matplotlib as mplf
from collections import Counter


# In[15]:

def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            labels.append(line.strip('\r\n'))
    return labels

# In[27]:

def predict(input_folder, model_path, model_name, output_folder, output_summary=False):
    t0 = datetime.now()
    # Dataset path

    labels = load_labels(model_path+'labels.txt')
    saved_model = model_path + model_name

    n_classes = len(labels)

    #print(n_classes)

    # Load dataset
    files, images = load_predict_data(input_folder)

    predict_num = len(images)

    #print(predict_num)
    #print(images.shape)

    if predict_num == 0:
        print('No images to predict!')
        return

    tf.reset_default_graph()

    # Graph input
    x = tf.placeholder(tf.float32, [predict_num, 227, 227, 3])

    # Model
    pred = Model.alexnet(x, n_classes, 1.0)

    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        # load the variables from disk.
        print('Loading model from:', saved_model)
        saver.restore(sess, saved_model)

        print('Start predicting')

        t2 = datetime.now()

        predication = sess.run(pred, feed_dict={x: images})

        print("Finish!")
        t1 = datetime.now()
        print(t1-t0)
        print(t1-t2)

        result_file = get_predict_result(files, predication, labels, output_folder, output_summary)
        print(result_file)
        return result_file



# In[82]:

def get_predict_result(files, pred, labels, output_folder, output_summary=False):
    text_result = ''
    predictions = []
    for i in range(len(files)):
        index = np.argmax(pred[i])
        predictions.append(index)
        label = labels[index]
        line = '%s,%s\r\n' % (files[i], label)
        #print(line)
        text_result += line

    pred_cnt = Counter(predictions)
    total = len(predictions)

    result_file = output_folder + 'predict_result.txt'
    with open(result_file, 'w') as f:
        f.write(text_result)
        #print(output_summary)
        #print(type(output_summary))
        if output_summary:
            result = '-------------\r\n'
            for (k,v) in pred_cnt.most_common(1):
                result += 'Conclusion:%d - %s, statistics:%d/%d, %.1f%%' % (k+1,labels[k], v, total, 100.0*v/total)
            f.write(result)
            print(result)

        return text_result


# In[7]:

def load_predict_data(input_folder):
    if not os.path.isdir(input_folder):
        raise Exception(
            'Specified file is not existing: %s' % (input_folder))
        return

    files = get_img_files(input_folder)
    count = len(files)

    images = np.ndarray([count, 227, 227, 3])

    if count == 0:
        print('No images to predict in the folder:', input_folder)
        return images


    for i in range(count):
        images[i] = load_image_file(files[i], 256, 227)

    print(count)
    #print(images.shape)


    return files, images


# In[8]:

def get_img_files(folder):
    if not os.path.isdir(folder):
        raise Exception(
            'Specified folder is not valid: %s' % (folder))
    data_files = [
        os.path.join(folder, d).replace('\\', '/') for d in sorted(os.listdir(folder))
            if not os.path.isdir(os.path.join(folder, d))]

    return data_files


# In[22]:

def load_image_file(file_name, scale_size, crop_size):
    #print('Loading file: ', file_name)

    if not os.path.exists(file_name):
        raise Exception(
            'Specified file is not existing: %s' % (file_name))

    img = ndimage.imread(file_name)
    h, w, c = img.shape
    assert c==3

    ratio = float(1.0 * h / w)
    if (h >= w):
        w = scale_size
        h = int(w * ratio)
    else:
        h = scale_size
        w = int (h / ratio)

    img = scipy.misc.imresize(img, (h, w))

    img = img.astype(np.float32)

    shift_h = int((h - crop_size)/2)
    shift_w = int((w - crop_size)/2)

    img_crop = img[shift_h:shift_h+crop_size, shift_w:shift_w+crop_size, :]

    return img_crop
