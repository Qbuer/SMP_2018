
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
    hidden_size = 128
    batch_size = 32
    n_epochs = 50
    lr = 0.001
    pass

def preprocess_data(data, token2id):
    ret = []
    for sentence, label in data:
        token_sentence = [token2id[x] for x in sentence]
        ret.append((token_sentence, label))
    return ret
    
class Classifier(object):
    """分类器
    
    Attributes:
        
    """

        
    def add_placeholders(self):
        """Some docs..."""
        max_length = self.config.max_length
        self.input_placeholder = tf.placeholder(
            tf.int32, shape=[None, None, self.config.n_word_features])
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None])
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
        embeddings = tf.reshape(
            embeddings, 
            (-1, self.config.max_length, self.config.n_word_embed_size * self.config.n_word_features))
        return embeddings
    
    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placehoder

        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        initial_state = lstm_cell.zero_state(self.config.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=initial_state)
        U = tf.get_variable(
            "U",
            shape=[self.config.hidden_size, self.config.n_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(
            "b2",
            shape=[self.config.n_classes],
            initializer=tf.constant_initializer(0.))
        
        preds = tf.matmul(state.h, U) + b2
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
    
    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(
            inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def fit(self, sess, saver, train_data_raw, dev_data):
        best_score = 0.
        train_data = preprocess_data(train_data_raw)
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            batch = util.minibatches(train_data, self.config.batch_size, shuffle=True)
            for x in batch: 
                loss = self.train_on_batch(sess, *x)
            logger.info("training finished")
            
            logger.info("Evaluating on development data")
            # token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
            # score = entity_scores[-1]
            
            # if score > best_score:
            #     best_score = score
            #     logger.info("New best score! Saving model in %s", self.config.model_output)
            #     saver.save(sess, self.config.model_output)

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
        
        self.build()


def do_train():
    pretrained_embeddings, _ = util.load_word_embedding(cache='cache')
    train_data = util.Data('./data/train.json', 'L:\\workspace\\ltp_data\\')
    dev_data = util.Data('./data/dev.json', 'L:\\workspace\\ltp_data\\')
    config = Config()
    # 配置参数. 测试集如何设置?
    _, config.max_length = train_data.get_metadata()
    config.n_classes = len(train_data.LABELS)
    config.n_word_embed_size = len(pretrained_embeddings[0])

    


    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = Classifier(pretrained_embeddings, config)
        logger.info("took %.2f seconds", time.time() - start)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            
            session.run(init)
            model.fit(session, saver, train_data, dev_data)


if __name__ == '__main__':
    do_train()
    

