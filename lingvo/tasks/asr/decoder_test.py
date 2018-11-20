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
"""Tests for speech decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import numpy as np
from six.moves import range
from six.moves import zip

import tensorflow as tf
from google.protobuf import text_format

from lingvo.core import layers as lingvo_layers
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.core.ops.hyps_pb2 import Hypothesis
from lingvo.tasks.asr import decoder

FLAGS = tf.flags.FLAGS


class DecoderTest(tf.test.TestCase):

  def _DecoderParams(self,
                     vn_config,
                     num_classes=32,
                     num_rnn_layers=1):
    """Create a small decoder for testing."""
    p = decoder.AsrDecoder.Params()

    p.name = 'decoder'
    uniform_init = py_utils.WeightInit.Uniform(0.1)

    # Set up embedding params.
    p.emb.vocab_size = num_classes
    p.emb.max_num_shards = 1
    p.emb.params_init = uniform_init

    # Set up decoder RNN layers.
    p.rnn_layers = num_rnn_layers
    rnn_params = p.rnn_cell_tpl
    rnn_params.params_init = uniform_init

    # Set up attention.
    p.attention.hidden_dim = 16
    p.attention.params_init = uniform_init

    # Set up final softmax layer.
    p.softmax.num_classes = num_classes
    p.softmax.params_init = uniform_init

    # Set up variational noise params.
    p.vn = vn_config
    p.vn.scale = tf.constant(0.1)

    p.target_seq_len = 5
    p.source_dim = 8
    p.emb_dim = 2
    p.rnn_cell_dim = 4

    return p

  def _testDecoderFPropHelper(self, params):
    """Computes decoder from params and computes loss with random inputs."""
    dec = decoder.AsrDecoder(params)
    src_seq_len = 5
    src_enc = tf.random_normal([src_seq_len, 2, 8],
                               seed=982774838,
                               dtype=py_utils.FPropDtype(params))
    src_enc_padding = tf.constant(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=py_utils.FPropDtype(params))
    # shape=[4, 5]
    target_ids = tf.transpose(
        tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15], [5, 6, 7, 8],
                     [10, 5, 2, 5]],
                    dtype=tf.int32))
    # shape=[4, 5]
    target_labels = tf.transpose(
        tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13],
                     [5, 7, 8, 10], [10, 5, 2, 4]],
                    dtype=tf.int32))
    # shape=[4, 5]
    target_paddings = tf.transpose(
        tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                     [1, 1, 1, 0]],
                    dtype=py_utils.FPropDtype(params)))
    target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
    target_weights = 1.0 - target_paddings
    # ids/labels/weights/paddings are all in [batch, time] shape.
    targets = py_utils.NestedMap({
        'ids': target_ids,
        'labels': target_labels,
        'weights': target_weights,
        'paddings': target_paddings,
        'transcripts': target_transcripts,
    })
    metrics, per_sequence_loss = dec.FPropWithPerExampleLoss(
        src_enc, src_enc_padding, targets, None)
    loss = metrics['loss']

    return loss, per_sequence_loss

  def _testDecoderFPropFloatHelper(self,
                                   func_inline=False,
                                   num_decoder_layers=1,
                                   target_seq_len=5,
                                   residual_start=0):
    """Computes decoder from params and computes loss with random inputs."""
    config = tf.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                do_function_inlining=func_inline)))
    with self.session(graph=tf.Graph(), use_gpu=False, config=config) as sess:
      tf.set_random_seed(8372749040)
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._DecoderParams(vn_config)
      p.rnn_layers = num_decoder_layers
      p.residual_start = residual_start
      p.target_seq_len = target_seq_len
      dec = p.cls(p)
      src_seq_len = 5
      src_enc = tf.random_normal([src_seq_len, 2, 8], seed=9283748)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float32)
      target_ids = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15],
                       [5, 6, 7, 8], [10, 5, 2, 5]],
                      dtype=tf.int32))
      target_labels = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13],
                       [5, 7, 8, 10], [10, 5, 2, 4]],
                      dtype=tf.int32))
      target_paddings = tf.transpose(
          tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                       [1, 1, 1, 1]],
                      dtype=tf.float32))
      target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
      target_weights = 1.0 - target_paddings
      targets = py_utils.NestedMap({
          'ids': target_ids,
          'labels': target_labels,
          'weights': target_weights,
          'paddings': target_paddings,
          'transcripts': target_transcripts,
      })
      metrics = dec.FPropDefaultTheta(src_enc, src_enc_padding, targets, None)
      loss = metrics['loss'][0]
      correct_predicts = metrics['fraction_of_correct_next_step_preds'][0]
      summaries = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

      tf.global_variables_initializer().run()
      loss_v, _ = sess.run([loss, correct_predicts])

      summaries.eval()

      return loss_v

  # Actual tests follow.

  def testDecoderConstruction(self):
    """Test that decoder can be constructed from params."""
    p = self._DecoderParams(
        vn_config=py_utils.VariationalNoiseParams(None, True, False))
    _ = decoder.AsrDecoder(p)

  def testDecoderFPropHelper(self):
    """Create decoder with default params, and verify that FProp runs."""
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(8372749040)

      p = self._DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(None, True, False))

      loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
      tf.global_variables_initializer().run()
      loss_val, per_sequence_loss_val = sess.run([loss, per_sequence_loss])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testDecoderFPropHelperWithProjection(self):
    """Create decoder with projection layers, and verify that FProp runs."""
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(8372749040)

      p = self._DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(None, True, False))
      p.rnn_cell_hidden_dim = 6

      loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
      tf.global_variables_initializer().run()
      loss_val, per_sequence_loss_val = sess.run([loss, per_sequence_loss])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testDecoderFPropDtype(self):
    """Create decoder with different fprop_type, and verify that FProp runs."""
    with self.session(use_gpu=False, graph=tf.Graph()) as sess:
      tf.set_random_seed(8372749040)

      p = self._DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(None, True, False))
      p.fprop_dtype = tf.float64

      loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
      tf.global_variables_initializer().run()
      loss_val, per_sequence_loss_val = sess.run([loss, per_sequence_loss])

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      # Target batch size is 4. Therefore, we should expect 4 here.
      self.assertEqual(per_sequence_loss_val.shape, (4,))

  def testDecoderFPropDeterministicAttentionDropout(self):
    """Verify that attention dropout is deterministic given fixed seeds."""
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(8372749040)
      p = self._DecoderParams(
          py_utils.VariationalNoiseParams(None, True, False, seed=1792))

      p.use_while_loop_based_unrolling = False
      p.attention.atten_dropout_prob = 0.5
      p.attention.atten_dropout_deterministic = True

      loss, per_sequence_loss = self._testDecoderFPropHelper(params=p)
      global_step = tf.train.get_global_step()
      tf.global_variables_initializer().run()
      loss_val, per_sequence_loss_val, global_steps_val, time_steps_val = (
          sess.run([
              loss, per_sequence_loss, 'decoder_1/accumulated_global_steps:0',
              'decoder_1/accumulated_time_steps:0'
          ]))

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      self.assertAllClose([3.473008, 15.0], loss_val)
      self.assertAllClose([13.563036, 10.053869, 10.362661, 18.115553],
                          per_sequence_loss_val)
      self.assertAllEqual([0, 0, 0, 0, 0], global_steps_val)
      self.assertAllEqual([1, 2, 3, 4, 5], time_steps_val)

      # Run another step to test global_step and time_step are incremented
      # correctly.
      sess.run(tf.assign_add(global_step, 1))
      loss_val, per_sequence_loss_val, global_steps_val, time_steps_val = (
          sess.run([
              loss, per_sequence_loss, 'decoder_1/accumulated_global_steps:0',
              'decoder_1/accumulated_time_steps:0'
          ]))

      print('loss = ', loss_val, 'per sequence loss = ', per_sequence_loss_val)
      self.assertAllClose([3.567736, 15.0], loss_val)
      self.assertAllClose([14.730419, 10.176270, 10.73501, 17.87434578],
                          per_sequence_loss_val)
      self.assertAllEqual([1, 1, 1, 1, 1], global_steps_val)
      self.assertAllEqual([1, 2, 3, 4, 5], time_steps_val)

  def testLabelSmoothing(self):
    """Verify that loss computation with label smoothing is as expected.."""
    with self.session(use_gpu=False) as sess:
      tf.set_random_seed(8372749040)

      p = self._DecoderParams(vn_config=py_utils.VariationalNoiseParams(None))
      p.label_smoothing = lingvo_layers.LocalizedLabelSmoother.Params()
      p.label_smoothing.offsets = [-2, -1, 1, 2]
      p.label_smoothing.weights = [0.015, 0.035, 0.035, 0.015]

      loss, _ = self._testDecoderFPropHelper(params=p)
      tf.global_variables_initializer().run()
      loss_val = sess.run(loss[0])

      print('loss = ', loss_val)
      self.assertAllClose(loss_val, 3.449249)

  def testDecoderFPropFloatNoInline(self):
    actual_value = self._testDecoderFPropFloatHelper(func_inline=False)
    test_utils.CompareToGoldenSingleFloat(self, 3.512219, actual_value)

  def testDecoderFPropFloatNoInlinePadTargetsToLongerLength(self):
    actual_value = self._testDecoderFPropFloatHelper(
        func_inline=False, target_seq_len=10)
    test_utils.CompareToGoldenSingleFloat(self, 3.512219, actual_value)

  def testDecoderFPropFloatInline(self):
    actual_value = self._testDecoderFPropFloatHelper(func_inline=True)
    test_utils.CompareToGoldenSingleFloat(self, 3.512219, actual_value)

  def testDecoderFPropFloatNoInline2Layers(self):
    actual_value = self._testDecoderFPropFloatHelper(
        func_inline=False, num_decoder_layers=2)
    test_utils.CompareToGoldenSingleFloat(self, 3.512603, actual_value)

  def testDecoderFPropFloatInline2Layers(self):
    actual_value = self._testDecoderFPropFloatHelper(
        func_inline=True, num_decoder_layers=2)
    test_utils.CompareToGoldenSingleFloat(self, 3.512603, actual_value)

  def testDecoderFPropFloat2LayersResidual(self):
    actual_value = self._testDecoderFPropFloatHelper(
        num_decoder_layers=2, residual_start=2)
    test_utils.CompareToGoldenSingleFloat(self, 3.513235, actual_value)

  def testDecoderFPropDouble(self):
    with self.session(use_gpu=False):
      tf.set_random_seed(8372749040)
      np.random.seed(827374)

      p = self._DecoderParams(
          vn_config=py_utils.VariationalNoiseParams(None, False, False))
      p.dtype = tf.float64

      dec = decoder.AsrDecoder(p)
      src_seq_len = 5
      src_enc = tf.constant(
          np.random.uniform(size=(src_seq_len, 2, 8)), tf.float64)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float64)
      target_ids = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15],
                       [5, 6, 7, 8], [10, 5, 2, 5]],
                      dtype=tf.int32))
      target_labels = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13],
                       [5, 7, 8, 10], [10, 5, 2, 4]],
                      dtype=tf.int32))
      target_paddings = tf.transpose(
          tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                       [1, 1, 1, 1]],
                      dtype=tf.float64))
      target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
      target_weights = 1.0 - target_paddings
      targets = py_utils.NestedMap({
          'ids': target_ids,
          'labels': target_labels,
          'weights': target_weights,
          'paddings': target_paddings,
          'transcripts': target_transcripts,
      })
      metrics = dec.FPropDefaultTheta(src_enc, src_enc_padding, targets, None)
      loss = metrics['loss'][0]

      tf.global_variables_initializer().run()

      test_utils.CompareToGoldenSingleFloat(self, 3.483581, loss.eval())
      # Second run to make sure the function is determistic.
      test_utils.CompareToGoldenSingleFloat(self, 3.483581, loss.eval())

  def _testDecoderFPropGradientCheckerHelper(self, func_inline=False):
    config = tf.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                do_function_inlining=func_inline)))
    with self.session(graph=tf.Graph(), use_gpu=False, config=config) as sess:
      tf.set_random_seed(8372749040)
      np.random.seed(274854)
      vn_config = py_utils.VariationalNoiseParams(None, False, False)
      p = self._DecoderParams(vn_config)
      p.dtype = tf.float64

      dec = p.cls(p)
      src_seq_len = 5
      src_enc = tf.constant(
          np.random.uniform(size=(src_seq_len, 2, 8)), tf.float64)
      src_enc_padding = tf.constant(
          [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
          dtype=tf.float64)
      target_ids = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 15],
                       [5, 6, 7, 8], [10, 5, 2, 5]],
                      dtype=tf.int32))
      target_labels = tf.transpose(
          tf.constant([[0, 1, 2, 3], [1, 2, 3, 4], [10, 11, 12, 13],
                       [5, 7, 8, 10], [10, 5, 2, 4]],
                      dtype=tf.int32))
      target_paddings = tf.transpose(
          tf.constant([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                       [1, 1, 1, 1]],
                      dtype=tf.float64))
      target_transcripts = tf.constant(['abcd', 'bcde', 'klmp', 'fghi', 'kfcf'])
      target_weights = 1.0 - target_paddings

      targets = py_utils.NestedMap({
          'ids': target_ids,
          'labels': target_labels,
          'weights': target_weights,
          'paddings': target_paddings,
          'transcripts': target_transcripts,
      })
      metrics = dec.FPropDefaultTheta(src_enc, src_enc_padding, targets, None)
      loss = metrics['loss'][0]
      all_vars = tf.all_variables()
      grads = tf.gradients(loss, all_vars)

      def DenseGrad(var, grad):
        if isinstance(grad, tf.Tensor):
          return grad
        elif isinstance(grad, tf.IndexedSlices):
          return tf.unsorted_segment_sum(grad.values, grad.indices,
                                         tf.shape(var)[0])

      dense_grads = [DenseGrad(x, y) for (x, y) in zip(all_vars, grads)]

      tf.global_variables_initializer().run()

      test_utils.CompareToGoldenSingleFloat(self, 3.493656, loss.eval())
      # Second run to make sure the function is determistic.
      test_utils.CompareToGoldenSingleFloat(self, 3.493656, loss.eval())

      symbolic_grads = [x.eval() for x in dense_grads if x is not None]
      numerical_grads = []
      for v in all_vars:
        numerical_grads.append(test_utils.ComputeNumericGradient(sess, loss, v))

      for x, y in zip(symbolic_grads, numerical_grads):
        self.assertAllClose(x, y)

  def testDecoderFPropGradientCheckerNoInline(self):
    self._testDecoderFPropGradientCheckerHelper(func_inline=False)

  def testDecoderFPropGradientCheckerInline(self):
    self._testDecoderFPropGradientCheckerHelper(func_inline=True)

  def _testDecoderBeamSearchDecodeHelperWithOutput(self,
                                                   params,
                                                   src_seq_len=None,
                                                   src_enc_padding=None):
    config = tf.ConfigProto(
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(do_function_inlining=False)))
    p = params
    with self.session(use_gpu=False, config=config) as sess:
      tf.set_random_seed(837274904)
      np.random.seed(837575)
      p.beam_search.num_hyps_per_beam = 4
      p.dtype = tf.float32
      p.target_seq_len = 5
      p.is_eval = True
      dec = p.cls(p)
      if src_seq_len is None:
        src_seq_len = 5
      src_enc = tf.constant(
          np.random.uniform(size=(src_seq_len, 2, 8)), tf.float32)
      if src_enc_padding is None:
        src_enc_padding = tf.constant(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=tf.float32)

      done_hyps = dec.BeamSearchDecode(src_enc, src_enc_padding).done_hyps
      tf.global_variables_initializer().run()

      softmax_wts = sess.run(dec.vars.softmax)
      print('softmax wts = ', softmax_wts)

      done_hyps_serialized = sess.run([done_hyps])[0]
      hyp = Hypothesis()
      print('done hyps shape = ', done_hyps_serialized.shape)
      for i in range(5):
        for j in range(8):
          print(i, j, len(done_hyps_serialized[i, j]))
      hyp.ParseFromString(done_hyps_serialized[2, 5])
      print('hyp = ', hyp)
      return hyp

  def _VerifyHypothesesMatch(self, hyp1, hyp2):
    tf.logging.info('hyp1 = %s', hyp1)
    tf.logging.info('hyp2 = %s', hyp2)
    self.assertEqual(hyp1.beam_id, hyp2.beam_id)
    self.assertEqual(list(hyp1.ids), list(hyp2.ids))
    self.assertAllClose(hyp1.scores, hyp2.scores)
    self.assertEqual(len(hyp1.atten_vecs), len(hyp2.atten_vecs))
    for av1, av2 in zip(hyp1.atten_vecs, hyp2.atten_vecs):
      self.assertAllClose(av1.prob, av2.prob)

  def testDecoderBeamSearchDecode(self):
    np.random.seed(837575)

    p = self._DecoderParams(
        vn_config=py_utils.VariationalNoiseParams(None, False, False),
        num_classes=8)
    p.beam_search.num_hyps_per_beam = 4
    p.dtype = tf.float32
    p.target_seq_len = 5

    expected_str = """
      beam_id: 1
      ids: 4
      ids: 6
      ids: 2
      scores: -1.99632573128
      scores: -1.99929893017
      scores: -1.95849049091
      atten_vecs {
        prob: 0.332584857941
        prob: 0.333580821753
        prob: 0.333834320307
        prob: 0.0
        prob: 0.0
      }
      atten_vecs {
        prob: 0.33258497715
        prob: 0.333580613136
        prob: 0.333834379911
        prob: 0.0
        prob: 0.0
      }
      atten_vecs {
        prob: 0.332585155964
        prob: 0.333580434322
        prob: 0.333834439516
        prob: 0.0
        prob: 0.0
      }
    """
    expected_hyp = Hypothesis()
    text_format.Merge(expected_str, expected_hyp)

    decoded_hyp = self._testDecoderBeamSearchDecodeHelperWithOutput(params=p)
    self._VerifyHypothesesMatch(expected_hyp, decoded_hyp)


if __name__ == '__main__':
  tf.test.main()
