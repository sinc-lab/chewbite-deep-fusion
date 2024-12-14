import os
import logging
import joblib
import hashlib
from datetime import datetime as dt

import pandas as pd

from chewbite_fusion.data.settings import CACHE_DIR

logger = logging.getLogger(__name__)


CACHE_INDEX_NAME = 'cache_index.txt'


class DatasetCache():
    ''' Implement cache-like structure to store and retrieve datasets files. '''

    def __init__(self):
        self.timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

        self.index_file = os.path.join(CACHE_DIR, CACHE_INDEX_NAME)

        if not os.path.isfile(self.index_file):
            self.cache_index = pd.DataFrame(columns=['item', 'params'])
        else:
            self.cache_index = pd.read_csv(self.index_file, index_col=0, names=['item', 'params'], sep='\t')

    def __get_filters_names__(self, **kargs):
        filters = []
        if ('filters' in kargs) and kargs['filters']:
            for i in kargs['filters']:
                filters.append((i[0].__name__, i[1]))

        kargs['filters'] = filters

        return kargs

    def load(self, **kargs):
        kargs = self.__get_filters_names__(**kargs)
        cache_item_key = '__'.join([f'{str(key)}-{str(value)}' for key, value in kargs.items()])
        logger.info(cache_item_key)

        item_key = hashlib.sha256(cache_item_key.encode(encoding='UTF-8')).hexdigest()

        if item_key in self.cache_index.index:
            cache_item_match = self.cache_index.loc[item_key]
            assert sum([m == item_key for m in self.cache_index.index]) == 1, \
                f'Duplicated entries for cache item ! {cache_item_key}'

            cache_item = joblib.load(os.path.join(CACHE_DIR, cache_item_match['item']))
            X = cache_item['X']
            y = cache_item['y']

            return (X, y)

        return None

    def save(self, X, y, **kargs):
        kargs = self.__get_filters_names__(**kargs)
        cache_item_key = '__'.join([f'{str(key)}-{str(value)}' for key, value in kargs.items()])

        cache_item = {
            'X': X,
            'y': y
        }

        cache_item = joblib.dump(cache_item, os.path.join(CACHE_DIR, self.timestamp + '.pkl'))

        item_key = hashlib.sha256(cache_item_key.encode(encoding='UTF-8')).hexdigest()
        self.cache_index.loc[item_key] = [self.timestamp + '.pkl', cache_item_key]
        self.cache_index.to_csv(self.index_file, header=None, sep='\t')
