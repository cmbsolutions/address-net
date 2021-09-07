import os
import tensorflow as tf

from addressnet.dataset import dataset
from addressnet.model import model_fn
from functools import lru_cache


@lru_cache(maxsize=2)
def _get_estimator(model_fn, model_dir):
    return tf.estimator.Estimator(model_fn=model_fn,
                                  model_dir=model_dir)


model_dir = "E:\\My Documents\\localRepos\\address-net\\addressnet\\netherlands"
assert os.path.isdir(model_dir), "invalid model_dir provided: %s" % model_dir
address_net_estimator = _get_estimator(model_fn, model_dir)
result = address_net_estimator.train(input_fn=dataset([os.path.join(model_dir, 'tf.dataset.train')], batch_size=100,
                                                      shuffle_buffer=1000, prefetch_buffer_size=10000,
                                                      num_parallel_calls=8), steps=1000)
