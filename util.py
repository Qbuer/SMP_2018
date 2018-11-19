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


    def __init__(self, data_file=None, ltp_data_path=None, max_length=None, stopwords=None):
        assert ltp_data_path is not None
        assert data_file is not None
 
        self.max_length = max_length
        self.data_file = data_file
        self.segmentor = Segmentor()
        
        self.segmentor.load(os.path.join(ltp_data_path, 'cws.model'))
        
        self.data = []
        self.padding_data = []
        self.__data__ = {}
        self.keys = []
        
        self.load_data(stopwords=stopwords)
        self.pad_data()

    def destory(self):
        self.segmentor.release()
        
    def get_metadata(self):
        """数据集的大小信息"""
        return len(self.data), self.max_length
    
    def load_data(self, stopwords=None):
        """load and parse the data"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        label_count = 0
        char_count = 0
        
        for key in json_data.keys():
            label = json_data.get(key).get('label')
            query = json_data.get(key).get('query')

            if label is None:
                label = 'website'
                
            self.__data__[key] = {
                "query": query,
                "label": label
            }
            self.keys.append(key)
            # TODO 截断 
            if self.max_length is not None:
                words = list(self.segmentor.segment(query))[:self.max_length]
            else:
                words = list(self.segmentor.segment(query))
            if stopwords is not None:
                words = [word for word in words if word not in stopwords] 
            self.data.append((words, self.LABELS.index(label)))
        
        if self.max_length is None:
            self.max_length = max([len(x[0]) for x in self.data])
    
    def update_labels(self, labels):
        for index, key in enumerate(self.keys):
            self.__data__.get(key)['label'] = self.LABELS[int(labels[index])]
        return self

    def save_result(self, outputfile="out.json"):
        with open(outputfile, 'w', encoding='utf-8') as f:
            json.dump(self.__data__, f, ensure_ascii=False)

        
        

    def pad_data(self):
        max_length = self.max_length
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
    if cache is not None and os.path.exists(cache):
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

def load_stopwords(input_file='data/stopwords.txt'):
    """some doces"""
    with open(input_file, encoding='utf-8', errors='ignore') as f:
        stopwords = set()
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def embedding_simplify(embedding, token2id, data):
    """保留需用到的词向量, 全部太大了"""
    new_embedding = []
    for item in data:
        for word in item[0]:
            index = token2id.get(word)
            if index is not None:
                new_embedding.append((word, embedding[index]))
    with open('embeding_terse', 'w', encoding='utf-8') as f:
        f.write("%d %d\n" % (len(new_embedding), 300))
        for item in new_embedding:
            f.write("%s %s\n" % (item[0], " ".join([str(x) for x in item[1]])))


def env_testing(_):
    """测试程序执行的环境"""

    path = "./" #文件夹目录
    files= os.listdir(path) #得到文件夹下的所有文件名称
    s = []
    for _file in files: #遍历文件夹
        print(_file)



def test1():
    train_data = Data('./data/train.json', 'L:\\workspace\\ltp_data\\')
    
    data = train_data.padding_data
    batch = minibatches(data, 12)
    for x in batch: 
        print(x)

def diff(grand_file, predict_file, outputfile):
    with open(grand_file, 'r', encoding='utf-8') as f:
        json_data1 = json.load(f)
    with open(predict_file, 'r', encoding='utf-8') as f:
        json_data2 = json.load(f)
    
    result = {}

    
    for key in json_data1.keys():
        label1 = json_data1.get(key).get('label')
        label2 = json_data2.get(key).get('label')
        if label1 != label2:
            result[key] = {}
            result[key]['query'] = json_data1.get(key).get('query')
            result[key]['grand'] = label1
            result[key]['predict'] = label2

    with open(outputfile, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

        

if __name__ == '__main__':

    grand_file = 'data/dev.json'
    predict_file = 'out.json'
    diff(grand_file, predict_file, 'diff.json')
    exit()

    pass
    stopwords = load_stopwords()
    train_data = Data('./data/train.json', 'L:\\workspace\\ltp_data\\', stopwords=stopwords)
    prediction = []
    for i in range(len(train_data.data)):
        prediction.append(0)
    
    train_data.update_labels(prediction)
    pass
        
    # data = train_data.data
    # pass
    # embedding, token2id = load_word_embedding(cache='cache')
    # embedding_simplify(embedding, token2id, data)
    

