# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Language model input generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.ops import io_ops

from lingvo.core import base_input_generator
from lingvo.core import py_utils
from lingvo.core import tokenizers
from lingvo.core.ops import py_x_ops



class LmInput(base_input_generator.BaseSequenceInputGenerator):
  """Reads tokenized plain text input such as from lm1b."""

  @classmethod
  def Params(cls):
    """Defaults params for LmInput."""
    p = super(LmInput, cls).Params()
    p.Define('use_chunks', False, 'Flag to use chunks during training.')
    p.Define('use_sst', False, 'Flag to use sst files as input source.')
    p.tokenizer = tokenizers.SimpleTokenizer.Params()

    p.Define('contiguous', False, 'Flag to use contigous input')
    # from BaseTinyDatasetInput
    p.Define('eval', False, 'Eval mode. This means we cannot use global step to infer batch id')
    p.Define('ckpt', None, 'A TensorFlow checkpoint.')
    p.Define('data', '', 'The tensor name in the ckpt.')
    p.Define('data_dtype', tf.int32, 'The tensor dtype in the ckpt.')
    p.Define('data_shape', (0,), 'The tensor dtype in the ckpt.')
    p.Define('num_steps', 0, 'Number of unrolling steps')
    p.Define('batch_size', 20, 'Batch size')

    return p

  @staticmethod
  def GetChunks(ids, labels, paddings):
    # TODO(jmluo) this happens before trimming and transposing
    py_utils.HasRank(ids, 2)
    # separate BIO tags from true ids
    tags = ids[:, 0::2] # Note that this also includes <S>
    ids = tf.concat([ids[:, 0:1], ids[:, 1:-1:2]], axis=1)
    # adjust labels accordingly
    labels = labels[:, 0::2]
    paddings = paddings[:, 0::2]

    # compute chunk ids
    is_B = tf.equal(tags, 4)
    is_I = tf.equal(tags, 5)
    is_O = tf.equal(tags, 6)
    is_BI = tf.logical_or(is_B, is_I)
    chunk_ids = tf.cumsum(tf.to_int32(is_B), axis=1) * tf.to_int32(is_BI) # shouldn't have overflow issues here
    # is_BO = tf.logical_or(is_B, is_O)
    # last_word_marks = tf.logical_and(is_BI, tf.logical_not(tf.concat([is_I[:, 1:], tf.zeros([tf.shape(ids)[0], 1], dtype=tf.bool)], axis=1)))
    # # last_word_marks => chunk_ids
    # # tf.assert_equal(tf.logical_or(tf.logical_not(last_word_marks), tf.greater(chunk_ids, 0)), tf.ones_like(chunk_ids, dtype=tf.bool))
    # # have the same number of chunks
    # last_word_marks = tf.to_int32(last_word_marks)
    # tf.assert_equal(tf.reduce_max(chunk_ids, axis=1), tf.reduce_sum(last_word_marks, axis=1))
    return ids, labels, paddings, chunk_ids #(chunk_ids, last_word_marks)

  def __init__(self, params):
    params.pad_to_max_seq_length = True
    super(LmInput, self).__init__(params)
    p = self.params

    if p.contiguous:
      assert len(p.data_shape) == 1
      assert not p.use_chunks
    else:
      text, self._word_count = self._BuildDataSource()
      self._ids, self._labels, self._paddings = self.StringsToIds(text)
      if p.use_chunks:
        self._ids, self._labels, self._paddings, self._chunk_ids = LmInput.GetChunks(self._ids, self._labels, self._paddings)
      self._input_batch_size = tf.shape(self._ids)[0]
      tf.summary.histogram('examples/sequence_length',
                           tf.reduce_sum(1.0 - self._paddings, axis=1))
      self._weights = 1.0 - self._paddings

      if py_utils.use_tpu():
        # When flush_every_n is on, at end of each epoch, our input
        # generator can generate a batch smaller than
        # bucket_batch_limit.
        assert not p.flush_every_n, 'flush_every_n is not allowed on TPU.'
        assert min(self.scaled_bucket_batch_limit) == max(
            self.scaled_bucket_batch_limit)
        bs = min(self.scaled_bucket_batch_limit)

        def SetShape(x):
          x.set_shape([bs, p.target_max_length])

        SetShape(self._ids)
        SetShape(self._labels)
        SetShape(self._paddings)
        SetShape(self._weights)
        self._word_count.set_shape([bs])

  def _DataSourceFromFilePattern(self, file_pattern):
    
    if self.params.use_sst:
      return py_x_ops.lm_text_input(
        file_pattern=file_pattern,
        normalization='',
        proto='string',
        **self.CommonInputOpArgs())
    else:
      def ReadInput(line):
        word_count = tf.size(tf.strings.split([line]))
        strlen = tf.size(tf.strings.split([line], ''))
        return line, word_count, strlen

      return py_x_ops.generic_input(
          file_pattern=file_pattern,
          processor=ReadInput,
          **self.CommonInputOpArgs())
    
  def InputBatchSize(self):
    return self.params.batch_size
    
  def _InputBatchFromCKPT(self):
    p = self.params

    @function.Defun()
    def ReadData():
      x, = io_ops.restore_v2(p.ckpt, [p.data], [''],
                               [p.data_dtype])
      return x

    # Loads data and label into memory and keep it around.
    data, = py_x_ops.cached_call(f=ReadData, T=[p.data_dtype])
    
    
    b = p.batch_size
    total_length = p.data_shape[0]
    total_batches = total_length // b
    total_steps = total_batches // p.num_steps
    left_over = total_batches % p.num_steps > 0
    if left_over:
      total_steps += 1
    
    if p.eval:
      dataset = tf.data.Dataset.range(total_steps).repeat()
      iterator = dataset.make_one_shot_iterator()
      global_step = iterator.get_next()
    else:
      global_step = py_utils.GetOrCreateGlobalStep() - 1
    
    batch_id = tf.to_int32(global_step % total_steps)
    
    data = data[:total_batches * b]
    data = tf.reshape(data, [b, total_batches])
    
    start = p.num_steps * batch_id
    end = tf.minimum(tf.to_int32(total_batches), start + p.num_steps)
    raw = tf.gather(data, tf.range(start, end, dtype=tf.int32), axis=1, name='ids')
    label_end = tf.minimum(end + 1, tf.to_int32(total_batches))
    label = tf.gather(data, tf.range(start + 1, label_end, dtype=tf.int32), axis=1, name='labels')
    raw = py_utils.PadOrTrimTo(raw, [b, end - start])
    ret = py_utils.NestedMap()
    # raw = tf.reshape(data[:700], [20, 35])
    # ret.ids = raw
    # ret.labels = raw
    # ret.weights = tf.ones([20, 35])
    # ret.paddings = 1.0 - ret.weights
    # ret.word_count = 700
    # ret.take_last_state = py_utils.GetOrCreateGlobalStep() > 0
    ret.ids = raw
    ret.labels = py_utils.PadOrTrimTo(label, [b, end - start])
    ret.weights = py_utils.PadOrTrimTo(tf.ones([b, label_end - start], dtype=tf.float32), [b, end - start])
    ret.paddings = 1.0 - ret.weights
    ret.word_count = b * (label_end - start - 1)
    ret.take_last_state = batch_id > 0
    
    return ret

  def InputBatch(self):
    p = self.params
    if p.contiguous:
      return self._InputBatchFromCKPT()

    ret = py_utils.NestedMap()
    ret.ids = self._ids
    ret.labels = self._labels
    ret.paddings = self._paddings
    ret.weights = self._weights
    ret.word_count = self._word_count
    if self.params.use_chunks:
        ret.chunk_ids = self._chunk_ids
        
    return ret
                       