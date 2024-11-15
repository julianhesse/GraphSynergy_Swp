import json
import pandas as pd
import random
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def create_folds(df, k, columns):
    '''k defines the number of folds, columns is a list of names that define the columns that are not supposed to have data leakage'''
    bucket_size = len(df) // k
    buckets_space = [bucket_size // k] * 5
    buckets = [[] for _ in range(k)]
    grouped_df = df.groupby(columns)
    result_data = []
    for (col1, col2), group in grouped_df:
        indices = group.index.tolist()
        count = len(indices)
        result_data.append([col1, col2, indices, count])
    result_df = pd.DataFrame(result_data, columns=(columns+['indices', 'count']))
    result_df = result_df.sort_values(by='count', ascending=False).reset_index(drop=True)

    for _, row in result_df.iterrows():
        bucket_index = random.randint(0,k-1)
        # check if bucket has enough space
        if buckets_space[bucket_index] < row.iloc[3]:
            checked = 1
            while checked < k:
                if bucket_index == k-1:
                    bucket_index = 0
                else:
                    bucket_index += 1
                if buckets_space[bucket_index] > row.iloc[3]:
                    break
                checked += 1
        buckets[bucket_index] += row.iloc[2]
        buckets_space[bucket_index] = buckets_space[bucket_index] - row.iloc[3]
    return buckets

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
