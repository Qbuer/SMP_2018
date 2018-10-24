
# coding: utf-8



import tensorflow as tf
from tensorflow.contrib import rnn
import logging
import util
import time
import sys
import argparse
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
        token_sentence = np.array([token2id[x] if token2id.get(x) else 0 for x in sentence])
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
            tf.int32, shape=[None, max_length])
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None, ])
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
            (-1, self.config.max_length * self.config.n_word_embed_size * self.config.n_word_features))
        return embeddings
    
    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placehoder

        # lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
        # initial_state = lstm_cell.zero_state(self.config.batch_size, dtype=tf.float32)
        # outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=initial_state)
        U = tf.get_variable(
            "U",
            shape=[self.config.max_length * self.config.n_word_embed_size, self.config.n_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(
            "b2",
            shape=[self.config.n_classes],
            initializer=tf.constant_initializer(0.))
        
        # preds = tf.matmul(state.h, U) + b2
        preds = tf.matmul(x , U)
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

    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(
            inputs_batch, dropout=self.config.dropout)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def evaluate(self, sess, dev_data):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        batch = util.minibatches(dev_data, self.config.batch_size)
        preds = []
        labels = []
        for x in batch:
            # tmp = self.predict_on_batch(sess, x[0])
            preds += list(self.predict_on_batch(sess, x[0]))
            labels += list(x[1])
        total_preds = len(preds)
        for pred, label in zip(preds, labels):
            if pred == label:
                correct_preds += 1
        
        return correct_preds / total_preds

    def fit(self, sess, saver, train_data_raw, dev_data_raw):
        best_score = 0.
        train_data = preprocess_data(train_data_raw.padding_data, self.token2id)
        dev_data = preprocess_data(dev_data_raw.padding_data, self.token2id)
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            batch = util.minibatches(train_data, self.config.batch_size, shuffle=True)
            for x in batch: 
                loss = self.train_on_batch(sess, *x)
            logger.info("training finished")
            
            logger.info("Evaluating on development data")
            
            print(self.evaluate(sess, dev_data))
            # batch = util.minibatches(dev_data, self.config.batch_size, shuffle=False)
            # for x in batch: 
            #     loss = self.predict_on_batch(sess, *x)
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

    def __init__(self, pretrained_embeddings, token2id, config):
        """Some docs"""
        self.pretrained_embeddings = pretrained_embeddings
        self.token2id = token2id
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placehoder = None
        self.dropout_placehoder = None
        self.config = config
        
        self.build()


def do_train(args):
    pretrained_embeddings, token2id = util.load_word_embedding(input_file=args.vectors, cache='cache')
    train_data = util.Data(args.data_train, args.ltp_data)
    dev_data = util.Data(args.data_dev, args.ltp_data, max_length=train_data.max_length)
    config = Config()
    # 配置参数. 测试集如何设置?
    _, config.max_length = train_data.get_metadata()
    config.n_classes = len(train_data.LABELS)
    config.n_word_embed_size = len(pretrained_embeddings[0])

    


    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = Classifier(pretrained_embeddings, token2id, config)
        logger.info("took %.2f seconds", time.time() - start)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            
            session.run(init)
            model.fit(session, saver, train_data, dev_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMP2018 -- Text Classification')
    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', default="data/train.json", help="Training data")
    command_parser.add_argument('-dd', '--data-dev',  default="data/dev.json", help="Dev data")
    command_parser.add_argument('-vv', '--vectors',  default="embeding_terse", help="Path to word vectors file")
    command_parser.add_argument('-ltp', '--ltp-data',  help="Path to ltp_data")
    command_parser.set_defaults(func=do_train)
    ARGS = parser.parse_args()
    
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
    

