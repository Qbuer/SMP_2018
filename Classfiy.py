
# coding: utf-8



import tensorflow as tf
from tensorflow.contrib import rnn
import logging
import util
import time


import numpy as np
import json
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


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
    
    # def __init__(self, num_input=None, timesteps=None, num_hidden=128, num_classes=None,
    #              batch_size=64, learning_rate=0.0001, trainning_steps=10000, display_step=200,
    #              _input=None):
    #     """设置分类器学习速率"""
    #     self.learning_rate = learning_rate
    #     self.trainning_steps = trainning_steps
    #     self.display_step = display_step
        
    #     assert num_input is not None
    #     assert timesteps is not None
    #     assert num_classes is not None
    #     assert _input is not None
        
    #     self.num_input = num_input
    #     self.timesteps = timesteps
    #     self.num_hidden = num_hidden
    #     self.num_classes = num_classes
        
    #     self.batch_size = batch_size
    #     self.input = _input
        
    def add_placeholders(self):
        """Some docs..."""
        max_length = self.config.max_length
        self.input_placeholder = tf.placeholder(
            tf.int32, shape=[None, max_length, self.config.n_features])
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.n_classes])
        self.mask_placehoder = tf.placeholder(
            tf.bool, shape=[None, max_length])
        self.dropout_placehoder = tf.placeholder(tf.float32, shape=[])
    
    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Some docs..."""
        
        feed_dict = {
            self.input_placeholder: inputs_batch,
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

        lstm_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size)
        initial_state = rnn_cell.zero_state(self.batch_size, dypte=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=initial_state)
        U = tf.get_variable(
            "U",
            shape=[self.config.hidden_size, self.config.n_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(
            "b2",
            shape=[self.config.n_classes],
            initializer=tf.constant_initializer(0.))
        
        preds = tf.matmul(outputs, U) + b2
        return preds
        
    def add_loss_op(self, preds):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels_placeholder,
                logits=preds,
                name="loss"))
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op
    
    def train_on_batch(elf, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(
            inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def fit(self, sess, saver, train_data, dev_data):
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
        """Some docs"""
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def __init__(self, pretrained_embeddings, config):
        """Some docs"""
        self.pretrained_embeddings = pretrained_embeddings
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placehoder = None
        self.dropout_placehoder = None
        self.config = config


def do_train():
    pretrained_embeddings = util.load_embedding()
    train_data = util.Data('./data/train.json', 'L:\\workspace\\ltp_data\\')
    dev_data = util.Data('./data/dev.json', 'L:\\workspace\\ltp_data\\')
    config = Config()
    
    

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = Classifier(pretrained_embeddings, config)
        logger.info("took %.2f seconds", time.time() - start)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            # model.fit(session, saver, train_data )


if __name__ == '__main__':
    do_train()
    

