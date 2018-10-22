
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.contrib import rnn


# In[2]:


import numpy as np
import json
import pickle


# In[3]:




# In[4]:


# In[5]:

class Config(object):
    n_word_features = 2

    pass
class Classifier(object):
    """分类器
    
    Attributes:
        learning_rate
        trainning_steps
        display_step
    """
    
    def __init__(self, num_input=None, timesteps=None, num_hidden=128, num_classes=None,
                 batch_size=64, learning_rate=0.0001, trainning_steps=10000, display_step=200,
                 _input=None):
        """设置分类器学习速率"""
        self.learning_rate = learning_rate
        self.trainning_steps = trainning_steps
        self.display_step = display_step
        
        assert num_input is not None
        assert timesteps is not None
        assert num_classes is not None
        assert _input is not None
        
        self.num_input = num_input
        self.timesteps = timesteps
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        
        self.batch_size = batch_size
        self.input = _input
        
        
    def training(self):
        
        X = tf.placeholder("float", [None, self.timesteps, self.num_input], name='X')
        Y = tf.placeholder("float", [None, self.num_classes], name="Y")
        weights = tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
        biases = tf.Variable(tf.random_normal([self.num_classes]))
        
        def RNN(x, weights, biases):
            x = tf.unstack(x, self.timesteps, 1)
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
            return tf.matmul(outputs[-1], weights) + biases
        
        logits = RNN(X, weights, biases)
        prediction = tf.nn.softmax(logits, name='predict')
        
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)
        
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            for step in range(0, self.trainning_steps):
                x_batch, y_batch = self.input.get_batch(step, self.batch_size)
                
                sess.run(train_op, feed_dict={X:x_batch, Y: y_batch})
        #         print(step)
                if step % self.display_step == 0:
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X:x_batch, Y:y_batch})
                    print("Step " + str(step) + ", Minibatch Loss= " +                       "{:.4f}".format(loss) + ", Training Accuracy= " +                       "{:.3f}".format(acc))
            print ("Optimization Finished")
            saver = tf.train.Saver()
            saver.save(sess, './checkpoint_dir3/MyModel')
    


# In[6]:


data = Data()
data.load_data('dev.json')

_, timesteps, num_classes, num_input = data.get_metadata()
classifier = Classifier(num_input=num_input, timesteps=timesteps, num_classes=num_classes, _input=data)
classifier.training()



# In[24]:


# import pickle
# with open('data', 'wb') as f:
#     pickle.dump(char_one_hot_embedding, f, protocol=0)
#     pickle.dump(max_length, f, protocol=0)
#     pickle.dump(labels_count, f, protocol=0)
#     pickle.dump(label_map, f, protocol=0)
    

