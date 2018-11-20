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
"""Speech model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six
from six.moves import range
from six.moves import zip

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import lr_schedule
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.tasks.asr import decoder
from lingvo.tasks.asr import decoder_utils
from lingvo.tasks.asr import encoder

# hyps: [num_beams, num_hyps_per_beam] of serialized Hypothesis protos.
# ids: [num_beams * num_hyps_per_beam, max_target_length].
# lens: [num_beams * num_hyps_per_beam].
# scores: [num_beams, num_hyps_per_beam].
# decoded: [num_beams, num_hyps_per_beam].
DecoderTopK = collections.namedtuple(
    'topk', ['hyps', 'ids', 'lens', 'scores', 'decoded'])  # pyformat: disable


class AsrModel(base_model.BaseTask):
  """Speech model."""

  @classmethod
  def Params(cls):
    p = super(AsrModel, cls).Params()
    p.encoder = encoder.AsrEncoder.Params()
    p.decoder = decoder.AsrDecoder.Params()
    p.Define(
        'target_key', '', 'If non-empty, will use the specified key from '
        'input_batch.additional_tgts to set training targets.')

    tp = p.train
    tp.lr_schedule = (
        lr_schedule.PiecewiseConstantLearningRateSchedule.Params().Set(
            boundaries=[350000, 500000, 600000], values=[1.0, 0.1, 0.01,
                                                         0.001]))
    tp.vn_start_step = 20000
    tp.vn_std = 0.075
    tp.l2_regularizer_weight = 1e-6
    tp.learning_rate = 0.001
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    tp.tpu_steps_per_loop = 20

    return p

  @base_layer.initializer
  def __init__(self, params):
    if not params.name:
      raise ValueError('params.name not set.')
    super(AsrModel, self).__init__(params)
    p = self.params

    with tf.variable_scope(p.name):
      # Construct the model.
      if p.encoder:
        if not p.encoder.name:
          p.encoder.name = 'enc'
        self.CreateChild('encoder', p.encoder)
      if p.decoder:
        if not p.decoder.name:
          p.decoder.name = 'dec'
        self.CreateChild('decoder', p.decoder)

  def ComputePredictions(self, theta, input_batch):
    src_enc, src_enc_padding, src_seg_id = self.encoder.FProp(
        theta.encoder, input_batch.src)
    if self.params.target_key:
      tf.logging.info(
          'Using batch.additional_tgts[%s] to source '
          'tgts instead of batch.tgts.', self.params.target_key)
      tgt = input_batch.additional_tgts[self.params.target_key]
    else:
      tgt = input_batch.tgt
    return self.decoder.ComputePredictions(theta.decoder, src_enc,
                                           src_enc_padding, tgt, src_seg_id)

  def ComputeLoss(self, theta, input_batch, predictions):
    tgt = input_batch.tgt
    if self.params.target_key:
      tgt = input_batch.additional_tgts[self.params.target_key]
    return self.decoder.ComputeLoss(theta.decoder, predictions, tgt)

  def _GetTopK(self, decoder_outs, tag=''):
    hyps = decoder_outs.topk_hyps
    ids = tf.identity(decoder_outs.topk_ids, name='TopKLabelIds' + tag)
    lens = tf.identity(decoder_outs.topk_lens, name='TopKLabelLengths' + tag)
    scores = decoder_outs.topk_scores
    decoded = decoder_outs.topk_decoded

    if ids is not None:
      decoded = self.input_generator.IdsToStrings(ids, lens - 1)
      decoded = tf.identity(decoded, name='top_k_decoded')
      decoded = tf.reshape(decoded, tf.shape(hyps))
    if scores is not None and hyps is not None:
      scores = tf.reshape(scores, tf.shape(hyps))
    return DecoderTopK(hyps, ids, lens, scores, decoded)

  def _ComputeNormalizedWER(self, hyps, refs):
    # Filter out all '<epsilon>' tokens for norm_wer computation.
    hyps_no_epsilon = tf.regex_replace(hyps, '(<epsilon>)+', ' ')
    # norm_wer is size [num_transcripts * hyps_per_beam, 2]
    norm_wer = decoder_utils.ComputeWer(hyps_no_epsilon, refs)
    # Split into two tensors of size [num_transcripts * hyps_per_beam, 1]
    norm_wer_errors, norm_wer_words = tf.split(norm_wer, [1, 1], 1)
    shape = [-1, self.params.decoder.beam_search.num_hyps_per_beam]
    norm_wer_errors = tf.reshape(norm_wer_errors, shape)
    norm_wer_words = tf.reshape(norm_wer_words, shape)

    return norm_wer_errors, norm_wer_words

  def AddAdditionalDecoderMetricsToGraph(self, topk_hyps, filtered_hyps,
                                         filtered_refs):
    """Returns a dict of metrics which should be computed from decoded hyps."""
    # The base class implementation returns an empty dictionary. Sub-classes can
    # provide their own implementation.
    return {}

  def Decode(self, input_batch):
    """Constructs the inference graph."""
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      src_enc, src_enc_padding, _ = (
          self.encoder.FPropDefaultTheta(input_batch.src))

      if hasattr(self.decoder, 'contextualizer'):
        self.decoder.contextualizer.SetContextMap(input_batch.tgt)
      decoder_outs = self.decoder.BeamSearchDecode(src_enc, src_enc_padding)
      topk = self._GetTopK(decoder_outs)

      utt_ids = input_batch.sample_ids
      tgt = input_batch.tgt
      if p.target_key:
        tgt = input_batch.additional_tgts[p.target_key]
      transcripts = self.input_generator.IdsToStrings(
          tgt.labels,
          tf.cast(tf.reduce_sum(1.0 - tgt.paddings, 1) - 1.0, tf.int32))

      # Filter out all isolated '<noise>' tokens.
      noise_pattern = ' <noise> |^<noise> | <noise>$|^<noise>$'
      filtered_refs = tf.regex_replace(transcripts, noise_pattern, ' ')
      filtered_hyps = tf.regex_replace(topk.decoded, noise_pattern, ' ')
      # Compute translation quality scores for all hyps.
      filtered_refs = tf.tile(
          tf.reshape(filtered_refs, [-1, 1]),
          [1, p.decoder.beam_search.num_hyps_per_beam])
      filtered_hyps = tf.reshape(filtered_hyps, [-1])
      filtered_refs = tf.reshape(filtered_refs, [-1])
      norm_wer_errors, norm_wer_words = self._ComputeNormalizedWER(
          filtered_hyps, filtered_refs)

      ret_dict = {
          'target_ids': tgt.ids,
          'target_labels': tgt.labels,
          'target_weights': tgt.weights,
          'target_paddings': tgt.paddings,
          'utt_id': utt_ids,
          'transcripts': transcripts,
          'topk_decoded': topk.decoded,
          'topk_ids': topk.ids,
          'topk_lens': topk.lens,
          'topk_scores': topk.scores,
          'norm_wer_errors': norm_wer_errors,
          'norm_wer_words': norm_wer_words,
      }

      ret_dict.update(
          self.AddAdditionalDecoderMetricsToGraph(topk, filtered_hyps,
                                                  filtered_refs))
      return ret_dict

  def CreateAdditionalDecoderMetrics(self):
    """Returns a dictionary of additional metrics which should be computed."""
    # The base class implementation returns an empty dictionary. Sub-classes can
    # provide their own implementation.
    return {}

  def CreateDecoderMetrics(self):
    base_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'wer': metrics.AverageMetric(),  # Word error rate.
        'norm_wer': metrics.AverageMetric(),  # Normalized word error rate.
        'sacc': metrics.AverageMetric(),  # Sentence accuracy.
        'ter': metrics.AverageMetric(),  # Token error rate.
        'corpus_bleu': metrics.CorpusBleuMetric(),
        'oracle_norm_wer': metrics.AverageMetric(),
    }

    # Add any additional metrics that should be computed.
    base_metrics.update(self.CreateAdditionalDecoderMetrics())
    return base_metrics

  def UpdateAdditionalMetrics(self, dec_out_dict, dec_metrics_dict):
    """Updates and returns a dictionary of metrics based on decoded hyps."""
    # Can be implemented in sub-classes to perform any model specific behavior.
    # The default implementation just returns the metrics unchanged.
    del dec_out_dict
    return dec_metrics_dict

  # TODO(prabhavalkar): Add support to save out the decoded hypotheses.
  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    p = self.params
    topk_scores = dec_out_dict['topk_scores']
    topk_decoded = dec_out_dict['topk_decoded']
    transcripts = dec_out_dict['transcripts']
    utt_id = dec_out_dict['utt_id']
    norm_wer_errors = dec_out_dict['norm_wer_errors']
    norm_wer_words = dec_out_dict['norm_wer_words']
    target_labels = dec_out_dict['target_labels']
    target_paddings = dec_out_dict['target_paddings']
    topk_ids = dec_out_dict['topk_ids']
    topk_lens = dec_out_dict['topk_lens']
    assert len(transcripts) == len(target_labels)
    assert len(transcripts) == len(target_paddings)
    assert len(transcripts) == len(topk_decoded)
    assert len(utt_id) == len(transcripts)
    assert (len(topk_ids) == p.decoder.beam_search.num_hyps_per_beam *
            len(transcripts))
    assert len(norm_wer_errors) == len(transcripts)
    assert len(norm_wer_words) == len(transcripts)

    dec_metrics_dict['num_samples_in_batch'].Update(len(transcripts))

    def GetRefIds(ref_ids, ref_paddinds):
      assert len(ref_ids) == len(ref_paddinds)
      return_ids = []
      for i in range(len(ref_ids)):
        if ref_paddinds[i] == 0:
          return_ids.append(ref_ids[i])
      return return_ids

    total_errs = 0
    total_oracle_errs = 0
    total_ref_words = 0
    total_token_errs = 0
    total_ref_tokens = 0
    total_norm_wer_errs = 0
    total_norm_wer_words = 0
    total_accurate_sentences = 0
    key_value_pairs = []
    for i in range(len(transcripts)):
      ref_str = transcripts[i]
      tf.logging.info('utt_id: %s', utt_id[i])
      tf.logging.info('  ref_str: %s', ref_str)
      hyps = topk_decoded[i]
      ref_ids = GetRefIds(target_labels[i], target_paddings[i])
      hyp_index = i * p.decoder.beam_search.num_hyps_per_beam
      top_hyp_ids = topk_ids[hyp_index][:topk_lens[hyp_index]]
      total_ref_tokens += len(ref_ids)
      _, _, _, token_errs = decoder_utils.EditDistanceInIds(
          ref_ids, top_hyp_ids)
      total_token_errs += token_errs

      assert p.decoder.beam_search.num_hyps_per_beam == len(hyps)
      filtered_ref = decoder_utils.FilterNoise(ref_str)
      filtered_ref = decoder_utils.FilterEpsilon(filtered_ref)
      oracle_errs = norm_wer_errors[i][0]
      for n, (score, hyp_str) in enumerate(zip(topk_scores[i], hyps)):
        tf.logging.info('  %f: %s', score, hyp_str)
        filtered_hyp = decoder_utils.FilterNoise(hyp_str)
        filtered_hyp = decoder_utils.FilterEpsilon(filtered_hyp)
        ins, subs, dels, errs = decoder_utils.EditDistance(
            filtered_ref, filtered_hyp)
        # Note that these numbers are not consistent with what is used to
        # compute normalized WER.  In particular, these numbers will be inflated
        # when the transcript contains punctuation.
        tf.logging.info('  ins: %d, subs: %d, del: %d, total: %d', ins, subs,
                        dels, errs)
        hyp_norm_wer_errors = norm_wer_errors[i][n]
        hyp_norm_wer_words = norm_wer_words[i][n]
        # Only aggregate scores of the top hypothesis.
        if n == 0:
          total_errs += errs
          total_ref_words += len(decoder_utils.Tokenize(filtered_ref))
          total_norm_wer_errs += hyp_norm_wer_errors
          if hyp_norm_wer_errors == 0:
            total_accurate_sentences += 1
          total_norm_wer_words += hyp_norm_wer_words
          dec_metrics_dict['corpus_bleu'].Update(filtered_ref, filtered_hyp)
        if hyp_norm_wer_errors < oracle_errs:
          oracle_errs = hyp_norm_wer_errors
      total_oracle_errs += oracle_errs

    dec_metrics_dict['wer'].Update(total_errs / total_ref_words,
                                   total_ref_words)
    dec_metrics_dict['oracle_norm_wer'].Update(
        total_oracle_errs / total_ref_words, total_ref_words)
    dec_metrics_dict['sacc'].Update(total_accurate_sentences / len(transcripts),
                                    len(transcripts))
    dec_metrics_dict['norm_wer'].Update(
        total_norm_wer_errs / total_norm_wer_words, total_norm_wer_words)
    dec_metrics_dict['ter'].Update(total_token_errs / total_ref_tokens,
                                   total_ref_tokens)

    # Update any additional metrics.
    dec_metrics_dict = self.UpdateAdditionalMetrics(dec_out_dict,
                                                    dec_metrics_dict)
    return key_value_pairs
