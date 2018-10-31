
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
    
    dropout = 1
    hidden_size = 300
    batch_size = 32
    n_epochs = 50
    lr = 0.001

    model_output = "./model"
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
        
        h_dropout = tf.nn.dropout(state.h, dropout_rate)
        
        preds = tf.matmul(h_dropout, U) + b2
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
            inputs_batch)
        predictions = np.argmax(sess.run(self.pred, feed_dict=feed), axis=1)
        return predictions

    def evaluate(self, sess, data, data_obj):
        correct_preds, total_correct, total_preds = 0., 0., 0.

        labels , preds = self.output(sess, data_obj, data)

        for pred, label in zip(preds, labels):
            if pred == label:
                correct_preds += 1
        total_preds = len(preds)
        
        
        return correct_preds / total_preds

    def output(self, sess, inputs_obj, inputs=None):
        if inputs is None:
            inputs = preprocess_data(inputs_obj.padding_data, self.token2id)

        batch = util.minibatches(inputs, self.config.batch_size, shuffle=False)
        predictions = []
        labels = []
        for x in batch:
            if len(x[0]) < self.config.batch_size:
                continue
            predictions += list(self.predict_on_batch(sess, x[0]))
            labels += list(x[1])
        
        return labels, predictions

    def fit(self, sess, saver, train_data_obj, dev_data_obj):
        best_score = 0.
        train_data = preprocess_data(train_data_obj.padding_data, self.token2id)
        dev_data = preprocess_data(dev_data_obj.padding_data, self.token2id)
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            batch = util.minibatches(train_data, self.config.batch_size, shuffle=True)
            loss_list = []
            for x in batch: 
                # print(len(x[0]))
                if len(x[0]) < self.config.batch_size:
                    logger.info('insufficient batch with length of %d' % (len(x[0])))
                    continue
                loss_list.append(self.train_on_batch(sess, *x))
                
            logger.info("average loss: %f" % (np.average(loss_list)))
            logger.info("training finished")
            
            logger.info("Evaluating on development data")
            
            
            score = self.evaluate(sess, dev_data, train_data_obj)
            logger.info("P: %f" % (score))
            
            
            if score > best_score:
                best_score = score
                logger.info("New best score! Saving model in %s", self.config.model_output)
                saver.save(sess, self.config.model_output)

        return best_score

    
            

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
    stopwords = util.load_stopwords()
    stopwords = None
    train_data = util.Data(args.data_train, args.ltp_data, stopwords=stopwords)
    dev_data = util.Data(args.data_dev, args.ltp_data, max_length=train_data.max_length, stopwords=stopwords)
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
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            
            session.run(init)
            score = model.fit(session, saver, train_data, dev_data) 
            print("\n")
            logger.info("training finished, took %.2f seconds with P: %.2f", time.time() - start, score)

def do_predict(args):

    pretrained_embeddings, token2id = util.load_word_embedding(input_file=args.vectors, cache='cache')
    stopwords = util.load_stopwords()
    stopwords = None
    train_data = util.Data(args.data_train, args.ltp_data, stopwords=stopwords)
    test_data = util.Data(args.data_test, args.ltp_data, max_length=train_data.max_length, stopwords=stopwords)
    config = Config()
    # 配置参数. 测试集如何设置?
    _, config.max_length = train_data.get_metadata()
    config.n_classes = len(train_data.LABELS)
    config.n_word_embed_size = len(pretrained_embeddings[0])
    config.batch_size = len(test_data.data)
    

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = Classifier(pretrained_embeddings, token2id, config)
        logger.info("took %.2f seconds", time.time() - start)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            
            session.run(init)
            saver.restore(session, model.config.model_output)
            labels, prediction = model.output(session, test_data, None)
            print(labels)
            print(prediction)
            test_data.update_labels(prediction).save_result()
            # print(model.evaluate(session, None, test_data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMP2018 -- Text Classification')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', default="data/train.json", help="Training data")
    command_parser.add_argument('-dd', '--data-dev',  default="data/dev.json", help="Dev data")
    command_parser.add_argument('-vv', '--vectors',  default="embeding_terse", help="Path to word vectors file")
    command_parser.add_argument('-ltp', '--ltp-data', default="L:/workspace/ltp_data/", help="Path to ltp_data")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('predict', help='')
    command_parser.add_argument('-dt', '--data-train', default="data/train.json", help="Training data")
    command_parser.add_argument('-dd', '--data-test', default="data/test.json", help="Training data")
    command_parser.add_argument('-vv', '--vectors',  default="embeding_terse", help="Path to word vectors file")
    command_parser.add_argument('-ltp', '--ltp-data', default=".", help="Path to ltp_data")
    command_parser.add_argument('-out', '--out-put', default="out.json", help="predict file")
    command_parser.set_defaults(func=do_predict)

    command_parser = subparsers.add_parser('env_test', help='')
    command_parser.set_defaults(func=util.env_testing)


    ARGS = parser.parse_args()
    
    

    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
    

