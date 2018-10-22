
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.contrib import rnn
import logging


# In[2]:


import numpy as np
import json
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
# In[3]:




# In[4]:


# In[5]:

class Config(object):
    n_word_features = 1

    n_word_embed_size = -1
    n_classes = -1
    max_length = -1
    
    dropout = 0.5
    
    batch_size = 32
    n_epochs = 50
    lr = 0.001
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
        
    def add_placeholders(self):
        """Some docs..."""
        self.input_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.max_length, self.config.n_features])
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.n_classes])
        self.mask_placehoder = tf.placeholder(
            tf.bool, shape=[None, self.max_length])
        self.dropout_placehoder = tf.placeholder(tf.float32, shape[])
    
    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        """Some docs..."""
        
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.mask_placehoder: mask_batch,
            self.dropout_placehoder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, self.input_placeholder)
        embeddings = tf.reshpae(
            embeddings, 
            (-1, self.max_length, self.config.n_word_embed_size * self.config.n_word_features))
        return embeddings
    
    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placehoder

        preds = []
        lstm_cell = None

    def add_loss_op(self, preds):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.boolean_mask(self.labels_placeholder, self.mask_placehoder)
                logits=tf.boolean_mask(preds, self.mask_placehoder),
                name="loss"))
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op
    
    def train_on_batch(elf, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(
            inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch
            dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            batch = getbatch()
            for x in batch: 
                loss = self.train_on_batch(sess, x)
            logger.info("training finished")
            
            logger.info("Evaluating on development data")
            token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
            score = entity_scores[-1]
            
            if score > best_score:
                best_score = score
                logger.info("New best score! Saving model in %s", self.config.model_output)
                saver.save(sess, self.config.model_output)

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def __init__(self, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placehoder = None
        self.dropout_placehoder = None
        
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
    

