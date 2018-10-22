import json
import numpy as np
import os
from pyltp import Segmentor

"""
SMP 分类工具类
"""

class Data(object):
    """Load and parse the json data file
    
    Attributes:
        segment_data: 经过分词处理的query列表
        labels: 标签列表
    """
    
    def __init__(self, data_file=None, ltp_data_path=None):
        assert ltp_data_path is not None
        assert data_file is not None
        self.data_file = data_file
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(ltp_data_path, 'cws.model'))
        self.labels = []
        self.segment_data = []
       
    def destory():
        self.segmentor.release()
        
    def get_metadata(self):
        """数据集的大小信息"""
        data_size = len(self.data)
        data_max_length = max([len(x) for x in self.data])
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
            self.segment_data.append(words)
            self.labels.append(label)
            
    def get_batch(self, batch_size, shuffle=True):
        """Get a batch of the data"""
        data_size = len(self.segment_data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            minibatch_indices = indices[batch_start:batch_start + batch_size]
            yield [self.segment_data[i] for i in minibatch_indices]
    
    def minibatches(self, batch_size, shuffle):
        return self.get_batch(batch_size, shuffle)
   
def test1():
    data = Data('./data/train.json', 'L:\\workspace\\ltp_data\\')
    data.load_data()
    data = data.get_batch(10)
    for item in data:
        print(item)

if __name__ == '__main__':
    test1()
