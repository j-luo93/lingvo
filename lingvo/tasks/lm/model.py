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
"""LM models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf
import numpy as np

from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import lr_schedule
from lingvo.core import py_utils
from lingvo.tasks.lm import layers


class LanguageModel(base_model.BaseTask):
  """Language model."""

  @classmethod
  def Params(cls):
    p = super(LanguageModel, cls).Params()
    p.Define('lm', layers.RnnLm.Params(), 'LM layer.')

    tp = p.train
    tp.Define(
        'max_lstm_gradient_norm', 0.0,
        'Clip gradient for vars in lstm layers by setting this value to '
        'something > 0.')
    tp.Define(
        'sum_loss_across_tokens_in_batch', False,
        'Sum the logP across predicted tokens in batch when set to True; '
        'average across predicted tokens in batch o/w (default).')
    tp.Define('isometric', 0.0, 'Weight for isometric constraint')
    tp.Define('chunk_loss_anneal', 0.0, 'Anneal weight for chunk loss to 1.0 at this many steps')

    tp.lr_schedule = lr_schedule.PiecewiseConstantLearningRateSchedule.Params(
    ).Set(
        boundaries=[350000, 500000, 600000], values=[1.0, 0.1, 0.01, 0.001])
    tp.vn_start_step = 20000
    tp.vn_std = 0.0
    tp.learning_rate = 0.001
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    p.Define('batch_size', 20, 'Batch size')
    p.Define('contiguous', False, 'Flag')
    p.Define('partial_restore', False, 'Flag')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(LanguageModel, self).__init__(params)
    p = self.params

    assert p.lm.vocab_size == p.input.tokenizer.vocab_size, (
        'lm.vocab_size does not match input.tokenizer.vocab_size: %d vs %d' %
        (p.lm.vocab_size, p.input.tokenizer.vocab_size))

    with tf.variable_scope(p.name):
      # Construct the model.
      self.CreateChild('lm', p.lm)

    def get_weight_params():
      return py_utils.WeightParams(
        shape=[1, p.batch_size, p.lm.emb.embedding_dim],
        init=py_utils.WeightInit.Constant(scale=np.zeros([p.batch_size, p.lm.emb.embedding_dim])),
        dtype=tf.float32,
        collections=[self.__class__.__name__ + '_vars'])

    # buffs = dict()
    for i in range(p.lm.rnns.num_layers):
      m = get_weight_params()
      c = get_weight_params()
      self.CreateVariable('last_state_%d_m' %i, m, trainable=False)
      self.CreateVariable('last_state_%d_c' %i, c, trainable=False)
      # buffs['last_state_%d_m' %i] = tf.Variable(np.zeros([p.batch_size, p.lm.emb.embedding_dim]), trainable=False, name='last_state_%d_m' %i, dtype=tf.float32)
      # buffs['last_state_%d_c' %i] = tf.Variable(np.zeros([p.batch_size, p.lm.emb.embedding_dim]), trainable=False, name='last_state_%d_c' %i, dtype=tf.float32)
    # self.buffs = buffs
      
  def _TrimIfPossibleThenTranspose(self, ids, paddings, labels, weights, chunk_ids=None):
    data = (ids, paddings, labels, weights)
    if not py_utils.use_tpu():
      max_seq_len = tf.cast(
          tf.reduce_max(tf.reduce_sum(1.0 - paddings, 1)), tf.int32)
      data = (x[:, :max_seq_len] for x in data)
      if chunk_ids is not None:
        chunk_ids = tf.transpose(chunk_ids[:, :max_seq_len])
    return [tf.transpose(x) for x in data] + [chunk_ids] 

  def FPropTower(self, theta, input_batch):
    p = self.params
    chunk_ids = input_batch.chunk_ids if p.lm.use_chunks else None
    ids, paddings, labels_ids, weights, chunk_ids = self._TrimIfPossibleThenTranspose(
        input_batch.ids, input_batch.paddings, input_batch.labels,
        input_batch.weights, chunk_ids=chunk_ids)

    seqlen = tf.shape(ids)[0]
    batch_size = tf.shape(ids)[1]
    zero_state = self.lm.zero_state(batch_size)
    
    with tf.name_scope('prepare_state'):
      if p.contiguous:
        state0 = py_utils.NestedMap(rnn=[])
        for i in range(p.lm.rnns.num_layers):
          if p.is_eval:
            last_m = tf.reshape(self.theta['last_state_%d_m' %i], [p.batch_size, p.lm.emb.embedding_dim])
            last_c = tf.reshape(self.theta['last_state_%d_c' %i], [p.batch_size, p.lm.emb.embedding_dim])
          else:
            last_m = self.theta['last_state_%d_m' %i]
            last_c = self.theta['last_state_%d_c' %i]
          m = tf.cond(input_batch.take_last_state, lambda: last_m, lambda: zero_state.rnn[i].m)
          c = tf.cond(input_batch.take_last_state, lambda: last_c, lambda: zero_state.rnn[i].c)
          # c = tf.Print(c, [c])
          state0.rnn.append(py_utils.NestedMap(c=c, m=m))
      else:
        state0 = zero_state
    labels = py_utils.NestedMap(class_ids=labels_ids, class_weights=weights)
    
    xent_output, state1 = self.lm.FProp(theta.lm, ids, paddings, state0, labels=labels, chunk_ids=chunk_ids)
    
    # self.state1 = state1
    
    if p.contiguous:
      assign_ops = list()
      for i in range(p.lm.rnns.num_layers):
        m = tf.reshape(state1.rnn[i].m, [1, p.batch_size, p.lm.emb.embedding_dim])
        c = tf.reshape(state1.rnn[i].c, [1, p.batch_size, p.lm.emb.embedding_dim])
        if not p.is_eval:
          state1.rnn[i].m = m
          state1.rnn[i].c = c
        assign_ops.append(tf.assign(self.vars['last_state_%i_m' %i], m))
        assign_ops.append(tf.assign(self.vars['last_state_%i_c' %i], c))
      self.last_state_group_op = tf.group(*assign_ops)
    
    # +1 to account for the end of sequence symbol.
    div = 2 if p.input.use_chunks else 1 # tags shouldn't be counted as words
    num_words = tf.cast(
        tf.reduce_sum(input_batch.word_count // div + tf.constant(1, dtype=tf.int32) * (1 - p.contiguous)),
        tf.float32)
    predicted_labels = tf.cast(xent_output.per_example_argmax, labels_ids.dtype)

    num_preds = xent_output.total_weight
    mean_acc = tf.reduce_sum(
        tf.cast(tf.equal(labels_ids, predicted_labels), tf.float32) *
        weights) / (
            num_preds + 1e-4)
    if p.lm.emb.cls == layers.HRREmbeddingLayer:
      if p.train.isometric > 0.0:
        isometric_constraint = 0.0
        nr = p.lm.emb.num_roles
        # TODO(jmluo) rearrange it to divide the code according to three modes
        if 'F' in theta.lm.emb:
          F_wm = theta.lm.emb.F
          nr, nf, d = F_wm.get_shape().as_list()
          # F2d leads to overspefication of parameters in F
          F2d = tf.reshape(F_wm, [nr * nf, d])
          diff = tf.matmul(F2d, tf.transpose(F2d)) - tf.eye(nr * nf)
          # diff = tf.matmul(F_wm, tf.transpose(F_wm, perm=[0, 2, 1])) - tf.eye(nf)
          isometric_constraint += tf.reduce_sum(diff**2) 
        if 'A' in theta.lm:
          d = theta.lm.A.get_shape().as_list()[0]
          A = tf.reshape(theta.lm.A, [d, 2, d])
          A1 = A[:, 0]
          A2 = A[:, 1]
          diff = tf.matmul(A1, tf.transpose(A2)) / 2
          # isometric_constraint += tf.reduce_sum(diff ** 2)

        if nr > 1 and 'r' in theta.lm.emb:
          r_wm = theta.lm.emb.r
          diff = tf.matmul(r_wm, tf.transpose(r_wm)) - tf.eye(nr)
          isometric_constraint += tf.reduce_sum(diff**2)
        if 'R' in theta.lm:
          R_wm = theta.lm.R
          diff = tf.matmul(R_wm, tf.transpose(R_wm)) - tf.eye(p.lm.num_sent_roles)
          isometric_constraint += tf.reduce_sum(diff**2)
        if p.lm.emb.mode == 'rs':
          assert 'rR' in theta.lm.emb
          rR = theta.lm.emb.rR
          diff = tf.matmul(rR, tf.transpose(rR)) - tf.eye(2)
          isometric_constraint += tf.reduce_sum(diff ** 2)

          rs_all = theta.lm.emb.rs.wm
          for rs in rs_all:
            rs = tf.reshape(rs, [-1, 2, 2])
            norm = tf.reduce_sum(rs ** 2, axis=-1)
            isometric_constraint += tf.reduce_sum((norm - 1.0) ** 2) + tf.reduce_sum((rs ** 2) * ((1 - rs) ** 2))

            normalized_rs = tf.nn.l2_normalize(rs, axis=-1)
            dot = tf.matmul(normalized_rs, tf.transpose(normalized_rs, perm=[0, 2, 1]))
            isometric_constraint += tf.reduce_sum(((dot * (tf.ones([2, 2]) - tf.eye(2))) ** 2) * 0.5)
          tf.summary.histogram('rs', tf.stack(rs_all))
        isometric_loss = isometric_constraint * p.train.isometric

    if p.lm.use_chunks:# and not p.is_eval:
      with tf.name_scope('global_decode'):
        assert p.lm.num_sent_roles > 0
        total_chunk_loss = -tf.reduce_sum(xent_output.chunk_log_probs)
        avg_chunk_loss = total_chunk_loss / xent_output.num_chunks
        global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
        temperature = tf.minimum(tf.constant(p.train.chunk_loss_anneal), global_step) / p.train.chunk_loss_anneal
        tf.summary.scalar('chunk/temperature', temperature)
        annealed_total_chunk_loss = temperature * total_chunk_loss
        annealed_avg_chunk_loss = temperature * avg_chunk_loss
        chunk_loss = annealed_avg_chunk_loss

    loss = xent_output.avg_xent
    if p.train.sum_loss_across_tokens_in_batch:
      loss = xent_output.total_xent
      if 'chunk_loss' in locals():
        chunk_loss = annealed_total_chunk_loss

    metrics = {
        'fraction_of_correct_next_step_preds': (mean_acc, num_preds),
        'log_pplx': (xent_output.avg_xent, num_preds),
        'log_pplx_per_word': (xent_output.total_xent / num_words, num_words),
        'num_predictions': (num_preds, 1),
        'num_words': (num_words, 1)
    }
    #tmp_loss = loss# + theta.dummy * theta.dummy
    if 'isometric_loss' in locals():
      #tmp_loss += isometric_loss
      metrics['isometric'] = (isometric_loss, 1)
    if 'chunk_loss' in locals():
      #tmp_loss += chunk_loss
      metrics['chunk_loss'] = (chunk_loss, 1)
      metrics['annealed_total_chunk_loss'] = (annealed_total_chunk_loss, 1)
      metrics['annealed_avg_chunk_loss'] = (annealed_avg_chunk_loss, xent_output.num_chunks)
      metrics['total_chunk_loss'] = (total_chunk_loss, 1)
      metrics['avg_chunk_loss'] = (avg_chunk_loss, xent_output.num_chunks)
      metrics['num_chunks'] = (xent_output.num_chunks, 1)
    #metrics['loss'] = (tmp_loss, num_preds)
    if p.train.sum_loss_across_tokens_in_batch:
        metrics['loss'] = (loss, 1)
    else:
        metrics['loss'] = (loss, num_preds)
    metrics['batch_size'] = (tf.cast(batch_size, tf.float32), 1)

    return metrics

  def AdjustEvalMetrics(self, metrics):
    with tf.name_scope('aggregate_loss'):
      if self.params.train.sum_loss_across_tokens_in_batch:
        loss, w = metrics['loss']
        loss = loss / metrics['batch_size'][0] 
        metrics['loss'] = (loss, w)

    return metrics

  def FProp(self, theta):
    metrics = super(LanguageModel, self).FProp(theta)
    if 'isometric' in metrics:
      self._loss = self._loss + metrics['isometric'][0]
    if 'chunk_loss' in metrics:# and False:
      if self.params.train.sum_loss_across_tokens_in_batch:
        self._loss = self._loss + metrics['annealed_total_chunk_loss'][0] / metrics['batch_size'][0]
      else:
        self._loss = self._loss + metrics['annealed_avg_chunk_loss'][0] 
    return metrics      
    

  def AdjustGradients(self, var_grad):
    """Clip LSTM gradients.

    Args:
      var_grad: a NestedMap of (variable, gradient). You can view
      var_grad as an ordered list of (key, (var, grad)) tuples. Every
      key of var_grad exists in vmap. Every variable in vmap that
      contributes to loss must exist in var_grad. Every var of var_grad
      must exist in vmap.  grad is the corresponding gradient computed
      for var. grad is guaranteed to be not None.

    Returns:
      adjusted version of var_grad that has clipped the LSTM gradients
      if self.params.max_lstm_gradient_norm is set.
    """

    p = self.params
    if p.train.max_lstm_gradient_norm:
      lstm_var_grad = var_grad.lm.rnns
      lstm_vars = lstm_var_grad.Transform(lambda x: x[0]).Flatten()
      lstm_grads = lstm_var_grad.Transform(lambda x: x[1]).Flatten()
      clipped_lstm_grads, _ = tf.clip_by_global_norm(
          lstm_grads, p.train.max_lstm_gradient_norm)
      var_grad.lm.rnns = var_grad.lm.rnns.Pack(
          list(zip(lstm_vars, clipped_lstm_grads)))

    return var_grad


  def Inference(self):
    """Constructs the inference subgraphs.

    Returns:
      {'subgraph_name': (fetches, feeds)}
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
      subgraphs['rnn_step'] = self._InferenceSubgraph_RNNStep()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    """Default inference subgraph.

    Returns:
      fetches: A dictionary of fetches, containing:
        log_pplx_per_token: A matrix of shape [batch, time]. [i, j]
          is i-th input text's j-th token's log prob.
        paddings: A matrix of shape [batch, time]. The padding mask.
        log_pplx_per_sample: A vector of shape [batch]. [i]
          is i-th input text's log prob.
        num_oovs_per_sample: A vector of shape [batch] counting the total number
          of out-of-vocabulary tokens in each input.
        tokens_from_labels: A vector of shape [batch] returning the predicted
          tokens as a sequence after mapping them back to strings from ids using
          the vocabulary.
        ids: A matrix of shape [batch, time]. [i, j]
          is i-th input text's j-th token's id.
      feeds: A dictionary of feeds, containing:
        text: A placeholder for a vector of strings.
    """
    p = self.params
    text = tf.placeholder(tf.string, shape=[None])
    # [batch, time]
    ids, labels, paddings = self.input_generator.StringsToIds(text)
    chunk_ids = None
    if p.lm.gold_chunks:
      ids, labels, paddings, chunk_ids = lm_inp.LmInput.GetChunks(ids, labels, paddings)
    lengths = tf.reduce_sum(tf.to_int32(1 - paddings), axis=1)
    tokens_from_labels = self.input_generator.IdsToStrings(labels, lengths)
    oovs = tf.equal(labels, self.input_generator.tokenizer.unk_id)
    num_oovs_per_sample = tf.to_int32(
        tf.reduce_sum(tf.to_float(oovs) * (1 - paddings), axis=1))
    # [time, batch]
    ids, paddings, labels, weights, chunk_ids = self._TrimIfPossibleThenTranspose(
        ids, paddings, labels, 1.0 - paddings, chunk_ids)
    batch_size = tf.shape(ids)[1]
    state0 = self.lm.zero_state(batch_size)
    if p.lm.num_sent_roles > 0 and not p.lm.global_decode:
      lower_state0 = self.lm.zero_state(batch_size)
      xent_output, _, _ = self.lm.FPropDefaultTheta(
          inputs=ids,
          paddings=paddings,
          state0=state0,
          lower_state0=lower_state0,
          labels=py_utils.NestedMap(class_ids=labels, class_weights=weights),
          chunk_ids=chunk_ids,
          ids=ids)
    else:
      xent_output, _ = self.lm.FPropDefaultTheta(
          inputs=ids,
          paddings=paddings,
          state0=state0,
          labels=py_utils.NestedMap(class_ids=labels, class_weights=weights),
          chunk_ids=chunk_ids,
          ids=ids)

    per_example_xent = py_utils.HasShape(xent_output.per_example_xent,
                                         tf.shape(ids))
    log_pplx_per_sample = tf.reduce_sum(
        per_example_xent * (1 - paddings), axis=0)
    fetches = {
        'log_pplx_per_token':  # [batch, time]
            tf.transpose(per_example_xent),
        'paddings':  # [batch, time]
            tf.transpose(paddings),
        'lengths':  # [batch]
            lengths,
        'log_pplx_per_sample':  # [batch]
            log_pplx_per_sample,
        'num_oovs_per_sample':  # [batch], int32
            num_oovs_per_sample,
        'tokens_from_labels':  # [batch], string
            tokens_from_labels,
        'ids':  # [batch, time], int32
            ids
    }
    feeds = {'text': text}

    # Also pass intermediate results
    if 'inter_res' in xent_output:
      inter_res = xent_output.inter_res
      for key in inter_res:
        new_key = 'inter_res.%s' %key
        assert new_key not in fetches
        fetches[new_key] = getattr(inter_res, key)
    return fetches, feeds

  def _InferenceSubgraph_RNNStep(self):
    """Inference subgraph for one rnn step.

    Returns:
      fetches: A dictionary of fetches, containing:
        zero_m_out_i: A matrix of shape [batch, output_size].
          m values of the i-th layer of zero recurrent state.
        zero_c_out_i: A matrix of shape [batch, hidden_size].
          c values of the i-th layer of zero recurrent state.
        logits: A matrix of shape [batch, num_candidates]. [i, j]
          is i-th input's j-th candidate's logit.
        m_out_i: A matrix of shape [batch, output_size].
          m values of the i-th layer of new recurrent state after one step.
        c_out_i: A matrix of shape [batch, hidden_size].
          c values of the i-th layer of new recurrent state after one step.
      feeds: A dictionary of feeds, containing:
        step_ids: A matrix of shape [batch, 1]. [i, 0]
          is the word id to run one step for the i-th input.
        candidate_ids: A 3D tensor of shape [batch, num_candidates, 2].
          [i, j, 0] = i just for indexing convenience.
          [i, j, 1] is the word id of the i-th input's j-th candidate.
        m_in_i: A matrix of shape [batch, output_size].
          m values of input recurrent state.
        c_in_i: A matrix of shape [batch, hidden_size].
          c values of input recurrent state.
    """
    fetches, feeds = {}, {}

    # Run one step with input ids and return logits.
    # [batch, 1]
    step_ids = tf.placeholder(tf.int32, [None, 1])
    feeds['step_ids'] = step_ids

    # Return logits only for certain candidate ids. This is to avoid returning
    # a big list of logits for all words.
    # This is a 3D tensor and it satisfies that:
    #    candidate_ids[i, j, 0] = i (just for indexing convenience)
    #    candidate_ids[i, j, 1] = the word id of the j-th candidate
    # [batch, num_candidates, 2]
    candidate_ids = tf.placeholder(tf.int32, [None, None, 2])
    feeds['candidate_ids'] = candidate_ids

    # Get initial zero states.
    batch_size = tf.shape(step_ids)[0]
    zero_state = self.lm.zero_state(batch_size)

    # Input LM state.
    state0 = zero_state.Transform(lambda x: tf.placeholder(tf.float32))

    # Run LM for one step
    step_ids_vec = tf.reshape(step_ids, [-1])
    step_paddings = tf.zeros(tf.shape(step_ids_vec), dtype=self.params.dtype)

    p = self.params
    lower_state0 = None
    if p.lm.num_sent_roles > 0 and not p.lm.global_decode:
      lower_zero_state0 = self.lm.lower_rnns.zero_state(batch_size)
      lower_state0 = lower_zero_state0.Transform(lambda x: tf.placeholder(tf.float32))
    res = self.lm.Step(self.lm.theta, step_ids_vec, step_paddings,
                               state0, lower_state0=lower_state0, step_inference=True) # TODO(jmluo) HACKY
    if p.lm.num_sent_roles > 0 and not p.lm.global_decode:
      out, state1, lower_state1 = res
      # add more feeds and fetches for lower level rnn
      feeds['lowerrnnstate:m'] = lower_state0.rnn[0].m
      feeds['lowerrnnstate:c'] = lower_state0.rnn[0].c
      fetches['lowerrnnstate:m'] = lower_state1.rnn[0].m
      fetches['lowerrnnstate:c'] = lower_state1.rnn[0].c
    else:
      out, state1 = res

    # Create feeds/fetches map for states.
    for i, (zero_s, s0, s1) in enumerate(
        zip(zero_state.rnn, state0.rnn, state1.rnn)):
      feeds['rnnstate:m_%02d' % i] = s0.m
      feeds['rnnstate:c_%02d' % i] = s0.c
      fetches['rnnstate:zero_m_%02d' % i] = zero_s.m
      fetches['rnnstate:zero_c_%02d' % i] = zero_s.c
      fetches['rnnstate:m_%02d' % i] = s1.m
      fetches['rnnstate:c_%02d' % i] = s1.c


    # Collect logits for candidates
    # [batch, num_candidates]
    prob = tf.nn.softmax(out.logits)
    candidate_prob = tf.gather_nd(prob, candidate_ids)
    candidate_logits = tf.log(candidate_prob)
    fetches['logits'] = candidate_logits


    if 'gating_probs' in out:
      fetches['gating_probs'] = out.gating_probs
    if 'cce' in out:
      fetches['cce'] = out.cce

    # print('check here', fetches)
    return fetches, feeds
