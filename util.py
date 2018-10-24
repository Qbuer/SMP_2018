import json
import logging
import numpy as np
import os
import pickle
from pyltp import Segmentor


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
"""
SMP 分类工具类
"""


class Data(object):
    """Load and parse the json data file
    
    Attributes:
        segment_data: 经过分词处理的query列表
        labels: 标签列表
    """
    
    LABELS = ['website', 'tvchannel', 'lottery', 'chat', 'match',
          'datetime', 'weather', 'bus', 'novel', 'video', 'riddle',
          'calc', 'telephone', 'health', 'contacts', 'epg', 'app', 'music',
          'cookbook', 'stock', 'map', 'message', 'poetry', 'cinemas', 'news',
          'flight', 'translation', 'train', 'schedule', 'radio', 'email']


    def __init__(self, data_file=None, ltp_data_path=None):
        assert ltp_data_path is not None
        assert data_file is not None
        self.data_file = data_file
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(ltp_data_path, 'cws.model'))
        
        self.data = []
        self.padding_data = []

        self.load_data()
        self.pad_data()

    def destory(self):
        self.segmentor.release()
        
    def get_metadata(self):
        """数据集的大小信息"""
        data_size = len(self.data)
        data_max_length = max([len(x[0]) for x in self.data])
        return data_size, data_max_length
    
    def load_data(self):
        """load and parse the data"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        label_count = 0
        char_count = 0
        
        for key in json_data.keys():
            label = json_data.get(key).get('label')
            query = json_data.get(key).get('query')
            words = list(self.segmentor.segment(query))
            self.data.append((words, self.LABELS.index(label)))
    
    def pad_data(self):
        max_length = max([len(x[0]) for x in self.data])
        padding = [0] 
        for sentence, label in self.data:
            new_sentence = sentence[:max_length] if len(sentence) >= max_length else sentence + padding * (max_length - len(sentence))
            self.padding_data.append((np.array(new_sentence), label))
            
    
            

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)


def load_word_embedding(input_file='./data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', cache=None):
    assert input_file is not None
    if cache is not None:
        with open(cache, 'rb') as f:
            embedding = pickle.load(f)
            token2id = pickle.load(f)
        return embedding, token2id
    with open(input_file, encoding='utf-8', errors='ignore') as f:
        metadata = next(f).strip()
        word_sum, dim = metadata.split(' ')
        embedding = np.array(np.random.randn(int(word_sum) + 1, int(dim)), dtype=np.float32)
        token2id = {}
        count = 1
        
        for line in f:
            items = line.strip().split(' ')
            token2id[items[0]] = count
            embedding[count] = np.array(items[1:])
            count += 1
        logging.info("Initialized embedding.")
        if cache is not None:
            with open("cache", 'wb') as f:
                pickle.dump(embedding, f)
                pickle.dump(token2id, f)
        return embedding, token2id



def test1():
    train_data = Data('./data/train.json', 'L:\\workspace\\ltp_data\\')
    
    data = train_data.padding_data
    batch = minibatches(data, 12)
    for x in batch: 
        print(x)
        

if __name__ == '__main__':
    # embedding, token2id = load_word_embedding(cache='cache')
    test1()

