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
"""Common layers for language models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import range
from six.moves import zip
import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers


class BaseLanguageModel(base_layer.LayerBase):
  """Abstract base class for a language model layer."""

  @classmethod
  def Params(cls):
    p = super(BaseLanguageModel, cls).Params()
    p.Define('vocab_size', 0, 'Number of vocabulary tokens.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(BaseLanguageModel, self).__init__(params)

  def zero_state(self, batch_size):
    raise NotImplementedError('Abstract method')

  def FProp(self, theta, inputs, paddings, state0, *args, **kwargs):
    """Computes xent loss given the language model inputs.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [time, batch] or [time, batch, dims].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A NestedMap containing the initial recurrent state.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      (xent_output, state1). xent_output is a NestedMap as defined by
      SoftmaxLayer's return value and state1 is the next recurrent
      state.
    """
    raise NotImplementedError('Abstract method')

  def Logits(self, theta, inputs, paddings, *args, **kwargs):
    """FProp and returns the logits for the whole sequence."""
    xent_output, _ = self.FProp(
        theta,
        inputs,
        paddings,
        state0=self.zero_state(tf.shape(inputs)[1]),
        *args,
        **kwargs)
    return xent_output.logits

  @classmethod
  def StepOutputDimension(cls, params):
    """Returns dimensions of Step()'s output dimension.

    Args:
      params: Params for this layer.

    Returns:
      output_dims: A NestedMap with fields.
        logits: a python int. the vocab size.
        last_hidden: a python int. The last hidden layer's dimension.
    """
    raise NotImplementedError('Abstract method')

  def Step(self, theta, inputs, paddings, state0, *args, **kwargs):
    """FProp one step.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [batch] or [batch, dims].
      paddings: a 0/1 tensor of shape [batch].
      state0: A NestedMap containing the initial recurrent state.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      output: A NestedMap with fields.
        logits: [batch, vocab_size].
        log_probs: [batch, vocab_size].
        last_hidden: [batch, dims].
      state1: The new recurrent state.
    """

    def ExpandTime(x):
      return tf.expand_dims(x, axis=0)

    xent_output, state1 = self.FProp(
        theta=theta,
        inputs=ExpandTime(inputs),
        paddings=ExpandTime(paddings),
        state0=state0,
        *args,
        **kwargs)

    output = py_utils.NestedMap()
    output.log_probs = tf.squeeze(xent_output.log_probs, axis=0)
    output.probs = tf.squeeze(xent_output.probs, axis=0)
    output.last_hidden = tf.squeeze(xent_output.last_hidden, axis=0)
    if 'logits' in xent_output:
      # FstLm doesn't return logits.
      output.logits = tf.squeeze(xent_output.logits, axis=0)
    return output, state1

  def GetFeedDict(self):
    """Returns an optional feed dict with str keys and Tensor values."""
    return {}


class NullLm(BaseLanguageModel):
  """A trivial language model does nothing really."""

  def zero_state(self, batch_size):
    return py_utils.NestedMap(
        m=tf.zeros([batch_size, 0], dtype=self.params.dtype))

  def FProp(self, theta, inputs, paddings, state0, *args, **kwargs):
    p = self.params
    time = tf.shape(inputs)[0]
    batch = tf.shape(inputs)[1]
    logits = tf.zeros([time, batch, p.vocab_size], dtype=p.dtype)
    return py_utils.NestedMap(
        logits=logits,
        probs=tf.nn.softmax(logits),
        log_probs=tf.nn.log_softmax(logits),
        last_hidden=tf.zeros([time, batch, 0], dtype=p.dtype)), state0

  def Logits(self, theta, inputs, paddings, *args, **kwargs):
    """FProp and returns the logits for the whole sequence."""
    p = self.params
    del theta, paddings
    time, batch = tf.unstack(tf.shape(inputs)[:2])
    return tf.zeros([time, batch, p.vocab_size], dtype=p.dtype)

  @classmethod
  def StepOutputDimension(cls, params):
    """Returns dimensions of Step()'s output dimension."""
    return py_utils.NestedMap(logits=params.vocab_size, last_hidden=0)

  def Step(self, theta, inputs, paddings, state0, *args, **kwargs):
    """FProp one step."""
    p = self.params
    batch = tf.shape(inputs)[0]
    logits = tf.zeros([batch, p.vocab_size], dtype=p.dtype)
    return py_utils.NestedMap(
        logits=logits,
        log_probs=tf.nn.log_softmax(logits),
        probs=tf.nn.softmax(logits),
        last_hidden=tf.zeros([batch, 0], dtype=p.dtype)), state0


def _RnnOutputSize(rnns):
  cell = rnns.cell_tpl[-1]
  return cell.num_output_nodes


class RnnLmNoEmbedding(BaseLanguageModel):
  """Stacked RNN based language model layer."""

  @classmethod
  def Params(cls):
    p = super(RnnLmNoEmbedding, cls).Params()
    p.Define('rnns', rnn_layers.StackedFRNNLayerByLayer.Params(),
             'The stacked-RNNs layer params.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'The softmax layer params.')
    p.Define('pred_proj', layers.ProjectionLayer.Params(),
             'The projection layer params.')
    p.Define('pred_rnn', rnn_layers.StackedFRNNLayerByLayer.Params(),
             'The rnn layer for chunk prediction')
    p.Define(
        'direct_features_dim', 0,
        'If > 0, then the number of dimensions of direct features '
        'that bypass the RNN and are provided directly to the softmax '
        'input.')
    p.Define('decoded_filler_keep_prob', 1.0, 'Keep prob for the decoded (noisy) filler embedding')
    p.Define('num_word_roles', 0, 'Number of roles on word level')
    p.Define('num_sent_roles', 0, 'Number of top/sentence level roles')
    p.Define('sent_role_anneal', 0.0, 'Anneal to 1.0 until this step.')
    p.Define('use_chunks', False, 'Whether to include chunk loss')
    p.Define('pred_mode', 'trigram', 'Prediction mode')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RnnLmNoEmbedding, self).__init__(params)
    p = self.params
    if not isinstance(p.rnns.cell_tpl, (list, tuple)):
      p.rnns.cell_tpl = [p.rnns.cell_tpl]

    cell_output_size = _RnnOutputSize(p.rnns)
    output_layer_size = cell_output_size + p.direct_features_dim
    
    if p.use_chunks:
        output_layer_size //= 2
    
    actual_output_size = output_layer_size * max(1, p.num_word_roles)    
    if actual_output_size != p.softmax.input_dim:
      raise ValueError(
          'Output layer size %d does not match softmax input size %d! '
          'cell_output_size: %d direct_features_dim: %d ' %
          (actual_output_size, p.softmax.input_dim, cell_output_size,
           p.direct_features_dim))
    if p.softmax.num_classes != p.vocab_size:
      raise ValueError(
          'softmax num of classess %d does not match vocabulary size %d!' %
          (p.softmax.num_classes, p.vocab_size))

    with tf.variable_scope(p.name):
      self.CreateChild('rnns', p.rnns)
      self.CreateChild('softmax', p.softmax)
      
      if p.use_chunks:
        sp = layers.SimpleFullSoftmax.Params()
        sp.name = 'lower_softmax'
        sp.num_classes = p.num_sent_roles
        input_dim = p.rnns.cell_tpl[-1].num_output_nodes // 2
        sp.input_dim = input_dim # Note the output is split into two parts
        self.CreateChild('lower_softmax', sp)

        cc_dim = p.rnns.cell_tpl[0].num_input_nodes
        if p.pred_mode == 'bigram':
          cc_inp = cc_dim
        elif p.pred_mode == 'trigram':
          cc_inp = 2 * cc_dim
        elif p.pred_mode == 'rnn':
          cc_inp = cc_dim
        else:
          raise
        if p.pred_mode == 'rnn':  
          self.CreateChild('pred_rnn', p.pred_rnn)
        else:
          self.CreateChild('pred_proj', p.pred_proj)
        SOS_pc = py_utils.WeightParams(
          shape=[cc_dim], # HACK
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
        EOS_pc = py_utils.WeightParams(
          shape=[p.num_sent_roles, cc_dim], # HACK
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('chunk_SOS', SOS_pc)
        self.CreateVariable('chunk_EOS', EOS_pc)
          
        # used for constructing two orthogonal contextualized word embeddings
        A_pc = py_utils.WeightParams(
          shape=[p.rnns.cell_tpl[0].num_input_nodes, 2 * p.rnns.cell_tpl[0].num_input_nodes], # HACK
          init=p.params_init,
          dtype=p.dtype,
          collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('A', A_pc)

        R_init_val = tf.random_normal(shape=[p.num_sent_roles, input_dim],
              stddev=0.044,
              dtype=tf.float32)
        R_init = py_utils.WeightInit.Constant(scale=R_init_val)
        R_pc = py_utils.WeightParams(
            shape=[p.num_sent_roles, p.rnns.cell_tpl[0].num_input_nodes], # HACK
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        self.CreateVariable('R', R_pc, trainable=False)

  def zero_state(self, batch_size):
    return self.rnns.zero_state(batch_size)

  @classmethod
  def StepOutputDimension(cls, params):
    return py_utils.NestedMap(
        logits=params.vocab_size, last_hidden=params.softmax.input_dim)

  def Step(self,
           theta,
           inputs,
           paddings,
           state0,
           direct_features=None,
           *args,
           **kwargs):
    """FProp one step.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [batch] or [batch, dims].
      paddings: a 0/1 tensor of shape [batch].
      state0: A NestedMap containing the initial recurrent state.
      direct_features: If not None, a tensor of[batch,
        direct_feature_dims] that is concatenated to the output of the last
        RNN layer.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      output: A NestedMap with fields.
        logits: [batch, vocab_size].
        last_hidden: [batch, dims].
      state1: The new recurrent state.
    """

    def ExpandTime(x):
      return tf.expand_dims(x, axis=0)

    if direct_features is not None:
      direct_features = py_utils.HasRank(direct_features, 2)
      direct_features = ExpandTime(direct_features)

    xent_output, state1 = self.FProp(
        theta=theta,
        inputs=ExpandTime(inputs),
        paddings=ExpandTime(paddings),
        state0=state0,
        direct_features=direct_features,
        *args,
        **kwargs)

    output = py_utils.NestedMap()
    output.logits = tf.squeeze(xent_output.logits, axis=0)
    output.probs = tf.squeeze(xent_output.probs, axis=0)
    output.log_probs = tf.squeeze(xent_output.log_probs, axis=0)
    output.last_hidden = tf.squeeze(xent_output.last_hidden, axis=0)
    if 'cce' in xent_output:
      output.cce = tf.squeeze(xent_output.cce, axis=-2)
    # TODO(jmluo) HACKY
    if 'gating_probs' in xent_output:
      output.gating_probs = tf.squeeze(xent_output.gating_probs, axis=0)

    return output, state1

  def FProp(self,
            theta,
            inputs,
            paddings,
            state0,
            labels=None,
            direct_features=None,
            emb_weights=None,
            chunk_ids=None,
            step_inference=False,
            ids=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: input activation. A tensor of shape [time, batch, dims].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A NestedMap containing the initial recurrent state.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.
      direct_features: If not None, a tensor of[time, batch,
        direct_feature_dims] that is concatenated to the output of the last
        RNN layer.

    Returns:
      If labels is not None, returns (xent_output, state1), where
      xent_output is a NestedMap as defined by SoftmaxLayer's return
      value and state1 is the next recurrent state. Otherwise,
      xent_output only contains the softmax logits.
    """
    inputs = py_utils.HasRank(inputs, 3)
    seqlen, batch, _ = tf.unstack(tf.shape(inputs), num=3)
    paddings = py_utils.HasShape(paddings, [seqlen, batch])
    assert state0 is not None
    p = self.params
    
    # Storage for intermediate results
    inter_res = py_utils.NestedMap(emb_word=inputs)
    
    activation, state1 = self.rnns.FProp(theta.rnns, inputs,
                                         tf.expand_dims(paddings, 2), state0)

    if direct_features is not None:
      direct_features = py_utils.HasRank(direct_features, 3)
      activation = tf.concat([activation, direct_features], axis=2)

    # retrieve word level representations from the sentence level ones.
    if p.use_chunks > 0:
      with tf.name_scope('predict_sent_role'):
        sent_act, activation = tf.split(activation, 2, axis=-1)
        lower_logits = self.lower_softmax.Logits(theta=theta.lower_softmax, inputs=tf.reshape(sent_act, [seqlen * batch, -1]))
        lower_sent_role_probs = tf.nn.softmax(lower_logits)
        
        inter_res.h_word = activation
        inter_res.h_sent = sent_act
        inter_res.logits_sent = lower_logits
        inter_res.role_probs = lower_sent_role_probs

          # sanity check -- one role only
          # lower_sent_role_probs = tf.stack([tf.ones([seqlen * batch]), tf.zeros([seqlen * batch])], axis=-1)

          # lower_sent_roles = py_utils.Matmul(lower_sent_role_probs, theta.R) # sl*bs x d

    def forward(softmax_name, act, h=None): # TODO(jmluo) may wanna rename activation
      softmax_layer = getattr(self, softmax_name)
      softmax_theta = getattr(theta, softmax_name)
      if labels is None:
        # We can only compute the logits here.
        logits = softmax_layer.Logits(
            theta=softmax_theta,
            inputs=tf.reshape(act, [seqlen * batch, -1]),
            activation=h,
            return_gating=p.softmax.gating)
        if p.softmax.gating:
          logits, gating_probs = logits

        xent_output = py_utils.NestedMap(
            logits=tf.reshape(logits, [seqlen, batch, -1]))
        xent_output.probs = tf.nn.softmax(xent_output.logits)
        xent_output.log_probs = tf.nn.log_softmax(xent_output.logits)
        if p.softmax.gating:
          xent_output.gating_probs = tf.reshape(gating_probs, [seqlen, batch, -1])
      elif 'class_ids' in labels:
        print(softmax_layer)
        xent_output = softmax_layer.FProp(
            theta=softmax_theta,
            inputs=act,
            class_weights=labels.class_weights,
            class_ids=labels.class_ids,
            activation=h)
      else:
        assert 'class_probabilities' in labels
        xent_output = softmax_layer.FProp(
            theta=softmax_theta,
            inputs=act,
            class_weights=labels.class_weights,
            class_probabilities=labels.class_probabilities,
            activation=h)
      xent_output.last_hidden = activation
      return xent_output

    p = self.params
    if p.num_word_roles == 0:
      return forward('softmax', activation), state1
    else:
      assert emb_weights is not None

      preceding_shape = tf.shape(activation)[:-1]
      f_noisy = self.emb.decode(tf.expand_dims(activation, axis=-2), emb_weights.r) # This is actually a bit hacky -- you don't know you have emb attribute
      if p.decoded_filler_keep_prob > 0 and not p.is_eval:
        f_noisy = tf.nn.dropout(f_noisy, p.decoded_filler_keep_prob)

      cat = tf.reshape(f_noisy, tf.concat([preceding_shape, [p.softmax.input_dim]], axis=0))
      out = forward('softmax', cat, h=activation)

      if p.num_sent_roles > 0:
        out.lower_roles = lower_sent_role_probs
        out.emb = inputs
        batch_major = True 
        if p.use_chunks and not step_inference: # skip chunk loss in step inference mode
          with tf.name_scope('chunk_prediction'):
            last_dim = tf.shape(sent_act)[-1]
            if batch_major:
              bm_inputs = tf.transpose(inputs, perm=[1, 0, 2]) # bs x sl x d
              w = tf.reshape(tf.matmul(tf.reshape(bm_inputs, [-1, last_dim]), theta.A), [-1, p.num_sent_roles, last_dim])
              rw = HRREmbeddingLayer.static_circular_conv(theta.R, w)
            else:
              w = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, last_dim]), theta.A), [-1, p.num_sent_roles, last_dim])
              rw = HRREmbeddingLayer.static_circular_conv(theta.R, w)

            inter_res.chunk_ids = chunk_ids
            inter_res.w = w
            inter_res.rw = rw
            
            if batch_major:
              bm_lower_sent_role_probs = tf.reshape(tf.transpose(tf.reshape(lower_sent_role_probs, [seqlen, batch, p.num_sent_roles]), perm=[1, 0, 2]), [batch * seqlen, p.num_sent_roles])
              clean_w = tf.expand_dims(bm_lower_sent_role_probs, axis=-1) * w # bsxsl x 2 x d
              clean_w = tf.reshape(clean_w, [batch, seqlen, p.num_sent_roles, last_dim])
              clean_w = [clean_w[:, :, ri] for ri in range(p.num_sent_roles)]
            else:
              clean_w = tf.expand_dims(lower_sent_role_probs, axis=-1) * w # size: sl*bs x 2 x d
              # clean_w = [clean_w[:, ri] for ri in range(p.num_sent_roles)] # bs x sl x d for each role
              clean_w = tf.transpose(tf.reshape(clean_w, [seqlen, batch, p.num_sent_roles, last_dim]), perm=[1, 2, 0, 3]) # size: bs x 2 x sl x d
            out.cce = clean_w
            inter_res.w_clean = clean_w


            bs_indices = tf.tile(tf.expand_dims(tf.range(batch), axis=0), [seqlen, 1])
            sl_indices = tf.tile(tf.expand_dims(tf.range(seqlen), axis=1), [1, batch])
            clen = tf.reduce_max(chunk_ids) + 1
            indices = tf.stack([bs_indices, chunk_ids, sl_indices], axis=-1) # size: sl x bs x 3
            sm_shape = [batch, clen, seqlen] # size: bs x cl x sl
            ones = tf.ones_like(chunk_ids)
            sm = tf.to_float(tf.scatter_nd(indices, ones, sm_shape))
            # TODO(jmluo): I don't even remember what sm stands for. Summation matrix?
            inter_res.sm = sm

            non_empty = tf.reduce_max(sm, axis=-1) # size: bs x cl
            last_chunk_id = tf.to_int32(tf.reduce_max(chunk_ids, axis=0)) # size: bs
            chunk_weights = tf.concat([tf.to_float(non_empty > 0)[:, :-1], tf.ones([batch, 1])], axis=-1) # size: bs x cl
            # chunk weight offset positions
            
            if batch_major:
              bm_bound_w = tf.reduce_sum(tf.expand_dims(bm_lower_sent_role_probs, axis=-1) * rw, axis=-2) # bs*sl x d 
              bound_w = tf.reshape(bm_bound_w, [batch, seqlen, last_dim])
            else:
              bound_w = tf.reduce_sum(tf.expand_dims(lower_sent_role_probs, axis=-1) * rw, axis=-2) # size: sl*bs x d
              bound_w = tf.transpose(tf.reshape(bound_w, [seqlen, batch, last_dim]), perm=[1, 0, 2]) # size: bs x sl x d
            
            chunk_emb = tf.matmul(sm, bound_w, name='chunk_e') # size: bs x cl x d
            if batch_major:
              # clean_chunk_emb = tf.matmul(tf.tile(tf.expand_dims(sm, axis=1), [1, p.num_sent_roles, 1, 1]), clean_w, name='chunk_f') # size: bs x 2 x cl x d
              # clean_chunk_emb = [clean_chunk_emb[:, ri] for ri in range(p.num_sent_roles)]
              clean_chunk_emb = [tf.matmul(sm, cw) for cw in clean_w] # bs x cl x d for each role
            else:  
              clean_chunk_emb = [tf.matmul(sm, cw) for cw in clean_w] # bs x cl x d for each role
              # clean_chunk_emb = tf.matmul(tf.tile(tf.expand_dims(sm, axis=1), [1, p.num_sent_roles, 1, 1]), clean_w, name='chunk_f') # size: bs x 2 x cl x d
            
            inter_res.bound_w = bound_w
            inter_res.ce = chunk_emb
            inter_res.cce = clean_chunk_emb

            # get input chunks and target chunks
            SOS_emb = tf.tile(tf.reshape(theta.chunk_SOS, [1, 1, -1]), [batch, 1, 1])
            input_chunk_emb = tf.concat([SOS_emb, chunk_emb[:, 1:]], axis=1) # replace the first chunk with chunk_emb embedding
            # input_chunk_emb = tf.nn.l2_normalize(input_chunk_emb, axis=-1)
            # EOS_emb = tf.tile(tf.reshape(theta.chunk_EOS, [1, p.num_sent_roles, 1, -1]), [batch, 1, 1, 1])
            # target_chunk_emb = tf.concat([clean_chunk_emb[:, :, 1:], EOS_emb], axis=2) # move EOS_emb to the end of sentences. After all paddings!
            EOS_emb = tf.tile(tf.reshape(theta.chunk_EOS, [1, p.num_sent_roles, 1, -1]), [batch, 1, 1, 1])
            EOS_embs = tf.unstack(EOS_emb, axis=1)
            target_chunk_emb = [tf.concat([clean_chunk_emb[ri][:, 1:], EOS_embs[ri]], axis=1) for ri in range(p.num_sent_roles)] # move EOS_emb to the end of sentences. After all paddings!
            
            # only normalize target embeddings (these are ground truth embeddings)
            target_chunk_emb = [tf.nn.l2_normalize(tce, axis=-1) for tce in target_chunk_emb]
            inter_res.input_chunk_emb = input_chunk_emb
            inter_res.target_chunk_emb = target_chunk_emb
            
            def mm3by2(x, y, transpose=False):
              with tf.name_scope('mm3by2'):
                py_utils.HasRank(x, 3)
                py_utils.HasRank(y, 2)
                bs, sl, dx = tf.unstack(tf.shape(x))
                dy = tf.shape(y)[0 if transpose else 1]
                return tf.reshape(tf.matmul(tf.reshape(x, [bs * sl, dx]), y, transpose_b=transpose), [bs, sl, dy])

            def get_predictions(chunk_emb):
              if p.pred_mode == 'rnn':
                input_ = tf.transpose(chunk_emb, [1, 0, 2])
                sent_state0 = self.pred_rnn.zero_state(batch)
                sent_paddings = tf.expand_dims(1.0 - tf.transpose(chunk_weights), 2) # NOTE this happens before deltas are applied
                h_chunk, _ = self.pred_rnn.FProp(theta.pred_rnn, input_, sent_paddings, sent_state0)
                # return h_chunk # NOTE seqlen major to get rid of one transpose
                return tf.transpose(h_chunk, [1, 0, 2])
              elif p.pred_mode == 'bigram':
                cat = chunk_emb
              elif p.pred_mode == 'trigram':
                # note that length dim is the second axis
                bs, cl, d = tf.unstack(tf.shape(chunk_emb))
                prev = tf.concat([tf.zeros([bs, 1, d]), chunk_emb[:, :-1]], axis=1)
                cat = tf.concat([prev, chunk_emb], axis=-1)
              elif p.pred_mode == 'rnn':
                cat = chunk_emb
              # h_chunk = mm3by2(tf.tanh(cat), theta.pred) # size: bs x cl x d
              h_chunk = self.pred_proj.FProp(theta.pred_proj, cat)
              return h_chunk
            
            h_chunk = get_predictions(input_chunk_emb)
            last_pred_pos_indices = tf.stack([tf.range(batch), last_chunk_id], axis=-1) # size: bs x 2
            # if p.pred_mode == 'rnn':
            #   rnn_last_pred_pos_indices = tf.stack([last_chunk_id, tf.range(batch)], axis=-1) # size: bs x 2
            #   f_chunk = HRREmbeddingLayer.static_circular_corr(theta.R, tf.expand_dims(h_chunk, axis=-2)) # size: cl x bs x 2 x d
            #   last_pred = tf.reshape(tf.gather_nd(f_chunk, rnn_last_pred_pos_indices),  [1, batch, p.num_sent_roles, -1]) 
            #   f_chunk = tf.concat([f_chunk[:-1], last_pred], axis=0)
            #   f_hat1, f_hat2 = tf.unstack(f_chunk, axis=-2) # cl x bs x d
            # else:
            f_chunk = HRREmbeddingLayer.static_circular_corr(theta.R, tf.expand_dims(h_chunk, axis=-2)) # size: bs x cl x 2 x d
            last_pred = tf.reshape(tf.gather_nd(f_chunk, last_pred_pos_indices),  [batch, 1, p.num_sent_roles, -1]) 
            f_chunk = tf.concat([f_chunk[:, :-1], last_pred], axis=1)
            f_hat1, f_hat2 = tf.unstack(f_chunk, axis=-2)
            inter_res.h_chunk = h_chunk
            inter_res.f_chunk = f_chunk
            inter_res.f_hat1 = f_hat1
            inter_res.f_hat2 = f_hat2
            

            # gold1, gold2 = tf.unstack(target_chunk_emb, axis=1)
            gold1, gold2 = target_chunk_emb
            # if p.pred_mode == 'rnn':
            #   merged_indices = tf.transpose(tf.reshape(tf.range(batch * clen), [batch, -1]), perm=[1, 0]) # cl x bs
            # else:
            merged_indices = tf.reshape(tf.range(batch * clen), [batch, -1])
            dot1 = mm3by2(f_hat1, tf.reshape(gold1, [batch * clen, -1]), transpose=True) # bs x cl x bs*cl / cl x bs x bs*cl (using rnn)
            dot2 = mm3by2(f_hat2, tf.reshape(gold2, [batch * clen, -1]), transpose=True) # bs x cl x bs*cl
            global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
            temperature = tf.minimum(tf.constant(p.sent_role_anneal), global_step) / p.sent_role_anneal
            tf.summary.scalar('temperature', temperature)
            den_dot = dot1 + dot2 * temperature

            inter_res.gold1 = gold1
            inter_res.gold2 = gold2
            inter_res.dot1 = dot1
            inter_res.dot2 = dot2
            inter_res.dot = den_dot

            with tf.name_scope('chunk_loss'):
              delta = tf.scatter_nd(last_pred_pos_indices, -tf.ones([batch]), [batch, clen])
              chunk_weights = chunk_weights + delta

              one_hot_target = tf.one_hot(merged_indices, batch * clen, off_value=1e-8)
              den_dot = den_dot + tf.reshape(chunk_weights * 99.0 - 99.0, [-1])
              chunk_log_probs = tf.reduce_sum(one_hot_target * tf.nn.log_softmax(den_dot), axis=-1)
              # if p.pred_mode == 'rnn':
              #   out.chunk_log_probs = chunk_log_probs * tf.transpose(chunk_weights, [1, 0])
              # else:
              out.chunk_log_probs = chunk_log_probs * chunk_weights
              out.num_chunks = tf.reduce_sum(chunk_weights) + 1e-8

              inter_res.w_chunk = chunk_weights
              inter_res.target = one_hot_target
              inter_res.masked_dot = den_dot
              inter_res.clp = out.chunk_log_probs
              inter_res.num_chunks = out.num_chunks
        out.inter_res = inter_res
        return out, state1
      else:
        return out, state1

class RnnLm(RnnLmNoEmbedding):
  """Stacked RNN based language model layer."""

  @classmethod
  def Params(cls):
    p = super(RnnLm, cls).Params()
    p.Define('emb', layers.EmbeddingLayer.Params(),
             'The embedding layer params.')
    p.Define('embedding_dropout_keep_prob', 1.0, 'Embedding dropout keep prob.')
    p.Define('embedding_dropout_seed', None, 'Embedding dropout seed.')
    p.Define('tie', False, 'Tie input and output embeddings.')
    p.emb.max_num_shards = 1
    return p

  # TODO(zhifengc): Consider merge Params() and CommonParams().
  @classmethod
  def CommonParams(cls,
                   vocab_size,
                   emb_dim=1024,
                   num_layers=2,
                   rnn_dims=2048,
                   rnn_hidden_dims=0,
                   residual_start=1,
                   softmax_max_alloc=None):
    """A LM model parameterized by vocab size, etc.

    Args:
      vocab_size: Vocab size.
      emb_dim: Embedding dimension.
      num_layers: The number of rnn layers.
      rnn_dims: Each RNN layer has this many output nodes.
      rnn_hidden_dims: If > 0, each RNN layer has this many hidden nodes.
      residual_start: index of the first layer with a residual connection;
        higher index layers also have residuals.
      softmax_max_alloc: If set to a positive integer the soft-max
        computation is chunked into allocations of at most
        softmax_max_alloc; when left to its default value of None no
        chunking is done.

    Returns:
      A RnnLm parameter object.
    """
    p = cls.Params()
    p.vocab_size = vocab_size

    init_scale = 1.0 / math.sqrt(rnn_dims)

    # Embedding.
    p.emb.vocab_size = vocab_size
    p.emb.embedding_dim = emb_dim
    p.emb.scale_sqrt_depth = True
    p.emb.params_init = py_utils.WeightInit.Uniform(init_scale)

    # RNNs
    p.rnns.num_layers = num_layers
    # Which layer starts to have the residual connection.
    p.rnns.skip_start = residual_start
    if num_layers > 1:
      p.rnns.cell_tpl = [
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=emb_dim,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims),
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=rnn_dims,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims)
      ]
    else:
      p.rnns.cell_tpl = [
          rnn_cell.LSTMCellSimple.Params().Set(
              num_input_nodes=emb_dim,
              num_output_nodes=rnn_dims,
              num_hidden_nodes=rnn_hidden_dims)
      ]

    # Softmax
    p.softmax.input_dim = rnn_dims
    p.softmax.num_classes = vocab_size
    p.softmax.params_init = py_utils.WeightInit.Uniform(init_scale)
    if softmax_max_alloc:
      # If the vocab is very large, computes the softmax chunk-by-chunk.
      p.softmax.chunk_size = max(1, int(softmax_max_alloc / vocab_size))

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(RnnLm, self).__init__(params)
    p = self.params

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert p.emb.embedding_dim == p.rnns.cell_tpl[0].num_input_nodes, (
        '{} vs. {}'.format(p.emb.embedding_dim,
                           p.rnns.cell_tpl[0].num_input_nodes))

    with tf.variable_scope(p.name):
      self.CreateChild('emb', p.emb)

  def FProp(self,
            theta,
            inputs,
            paddings,
            state0,
            labels=None,
            direct_features=None,
            chunk_ids=None,
            step_inference=False,
            ids=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: input ids. An int32 tensor of shape [time, batch].
      paddings: a 0/1 tensor of shape [time, batch].
      state0: A NestedMap containing the initial recurrent state.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.
      direct_features: If not None, a tensor of[time, batch,
        direct_feature_dims] that is concatenated to the output of the last
        RNN layer.

    Returns:
      If labels is not None, returns (xent_output, state1), where
      xent_output is a NestedMap as defined by SoftmaxLayer's return
      value and state1 is the next recurrent state. Otherwise,
      xent_output only contains the softmax logits.
    """
    p = self.params
    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    assert state0
    
    def forward(activation):
      # Dropout on embeddings is only applied in training.
      if p.embedding_dropout_keep_prob < 1.0 and not p.is_eval:
        activation = tf.nn.dropout(
            activation,
            keep_prob=p.embedding_dropout_keep_prob,
            seed=p.embedding_dropout_seed)

      return super(RnnLm, self).FProp(theta, activation, paddings, state0,
                                      labels=labels,
                                      direct_features=direct_features,
                                      emb_weights=emb_weights,
                                      chunk_ids=chunk_ids,
                                      step_inference=step_inference,
                                      ids=ids)

    # TODO(jmluo) may wanna get rid of this assertion to obtain a baseline (nr > 0 but w/o HRR)
    # also, should move this into __init__.
    if p.num_word_roles > 0:
      assert p.emb.cls == HRREmbeddingLayer
      assert p.tie

    if p.emb.cls == HRREmbeddingLayer:
      activation, signature, emb_weights = self.emb.EmbLookup(theta.emb, ids, role_anneal=p.softmax.role_anneal)
    else:
      activation = self.emb.EmbLookup(theta.emb, ids)
      emb_weights = None

    if p.tie:
      try:
        num_shards = len(theta.emb.wm)
      except:
        num_shards = len(emb_weights.f)

      def transpose_or_not(w):
        transpose = (p.softmax.num_sampled == 0)
        if transpose:
          return tf.transpose(w)
        else:
          return w

      if p.emb.cls == HRREmbeddingLayer:
        if p.num_word_roles > 0:
          # for i in xrange(p.num_roles):
          #   softmax_theta = getattr(theta, 'softmax_%d' %i)
          for shard_ind in xrange(num_shards):
            f_shard = emb_weights.f[shard_ind]
            reshaped_f_shard = tf.reshape(f_shard, [-1, p.softmax.input_dim])
            theta.softmax['weight_%d' %shard_ind] = transpose_or_not(reshaped_f_shard)
        else:
          for shard_ind in xrange(num_shards):
            theta.softmax['weight_%d' %shard_ind] = transpose_or_not(emb.e[shard_ind])
      else:
        for shard_ind in xrange(num_shards):
          main = transpose_or_not(theta.emb.wm[shard_ind])
          theta.softmax['weight_%d' %shard_ind] = main

    res = forward(activation)
    xent_output = res[0]
    return res

class MoeLm(BaseLanguageModel):
  """Mixture of experts language modeling class."""

  @classmethod
  def Params(cls):
    p = super(MoeLm, cls).Params()
    p.Define(
        'emb',
        layers.EmbeddingLayer.Params().Set(max_num_shards=1),
        'The embedding layer params.')
    p.Define('shared_emb', True, 'If true, uses a single embedding')
    p.Define(
        'add_postgating_rnn', True, 'If true, add an RNNLM post gating. '
        'If false, add only a softmax on top.')
    p.Define('rnns', rnn_layers.StackedFRNNLayerByLayer.Params(),
             'The stacked-RNNs layer params.')
    p.Define('number_of_experts', 7, 'Number of experts.')
    p.Define('merge', RnnLmNoEmbedding.Params(),
             'The LM to use for the merged LM')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MoeLm, self).__init__(params)
    p = self.params
    if not isinstance(p.rnns.cell_tpl, (list, tuple)):
      p.rnns.cell_tpl = [p.rnns.cell_tpl]

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert p.emb.embedding_dim == p.rnns.cell_tpl[0].num_input_nodes, (
        '{} vs. {}'.format(p.emb.embedding_dim,
                           p.rnns.cell_tpl[0].num_input_nodes))
    if p.add_postgating_rnn:
      assert p.merge.vocab_size == p.vocab_size, ('{} vs. {}'.format(
          p.merge.vocab_size, p.vocab_size))

    with tf.variable_scope(p.name):
      # Embeddings
      if p.shared_emb:
        self.CreateChild('emb', p.emb)
      else:
        # 0-th embedding is for the domain predictor.
        self.CreateChildren(
            'emb', [
                p.emb.Copy().Set(name='emb_%d' % i)
                for i in range(1 + p.number_of_experts)
            ])

      # Rnns
      # 0-th rnns is for the domain predictor.
      self.CreateChildren(
          'rnns', [p.rnns.Copy() for i in range(1 + p.number_of_experts)])

      # Softmax
      rnn_output_size = _RnnOutputSize(p.rnns)
      sm_params = layers.SimpleFullSoftmax.Params()
      sm_params.name = 'domain_predictor_softmax'
      sm_params.input_dim = rnn_output_size
      sm_params.num_classes = p.number_of_experts
      self.CreateChild('domain_predictor_softmax', sm_params)

      # Merge
      if p.add_postgating_rnn:
        self.CreateChild('merge', p.merge)
      else:
        output_sm_params = layers.SimpleFullSoftmax.Params()
        output_sm_params.name = 'output_softmax'
        output_sm_params.input_dim = rnn_output_size
        output_sm_params.num_classes = p.vocab_size
        self.CreateChild('output_softmax', output_sm_params)

  def zero_state(self, batch_size):
    p = self.params
    if p.add_postgating_rnn:
      return py_utils.NestedMap(
          rnns=[x.zero_state(batch_size) for x in self.rnns],
          merge=self.merge.zero_state(batch_size))
    else:
      return py_utils.NestedMap(
          rnns=[x.zero_state(batch_size) for x in self.rnns])

  def FProp(self, theta, inputs, paddings, state0, labels=None):
    """Forward compute."""
    p = self.params

    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    seqlen, batch = tf.unstack(tf.shape(inputs), num=2)
    assert state0

    paddings_3d = tf.expand_dims(paddings, axis=2)

    # RNNs
    if p.shared_emb:
      emb_act = [self.emb.EmbLookup(theta.emb, inputs)
                ] * (1 + p.number_of_experts)
    else:
      emb_act = [
          self.emb[i].EmbLookup(theta.emb[i], inputs)
          for i in range(1 + p.number_of_experts)
      ]
    state1 = py_utils.NestedMap(rnns=[])
    rnns_act = []
    for i, act in enumerate(emb_act):
      act, state = self.rnns[i].FProp(theta.rnns[i], act, paddings_3d,
                                      state0.rnns[i])
      act = py_utils.HasRank(act, 3)
      rnns_act += [act]
      state1.rnns += [state]

    # [time, batch, experts, dims].
    expert_stacked = tf.stack(rnns_act[1:], axis=2)

    # Compute gating softmax. The 0-th rnns is used as the expert
    # predictor.  Because SoftmaxLayer.Logits takes a matrix as input,
    # we reshape rnns_act[0], the domain predictor activation, to a
    # matrix here.
    act = tf.reshape(rnns_act[0], [seqlen * batch, -1])
    logits = self.domain_predictor_softmax.Logits(
        theta.domain_predictor_softmax, act)
    # [time, batch, experts]
    gating = tf.reshape(tf.nn.softmax(logits), [seqlen, batch, -1])

    # Mix the experts.
    # [time, batch, dims]
    combined = tf.squeeze(
        tf.matmul(
            # [time, batch, 1, experts]
            tf.expand_dims(gating, axis=2),
            # [time, batch, experts, dims]
            expert_stacked),
        axis=2)

    if p.add_postgating_rnn:
      # Note that this layer includes 1 or more RNN layers followed
      # by a softmax.
      xent_loss, state1.merge = self.merge.FProp(theta.merge, combined,
                                                 paddings, state0.merge, labels)
    else:
      xent_loss = self.output_softmax.FProp(
          theta=theta.output_softmax,
          inputs=combined,
          class_weights=labels.class_weights,
          class_ids=labels.class_ids)

    # return xent_loss, state1
    return xent_loss, state1


class TransformerLmNoEmbedding(BaseLanguageModel):
  """Transformer language model."""

  @classmethod
  def Params(cls):
    p = super(TransformerLmNoEmbedding, cls).Params()
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define(
        'model_dim', 512, 'Model dimension that applies to embedding '
        'layers and all Transformer layers.')
    p.Define('num_trans_layers', 6, 'Number of Transformer layers.')
    p.Define('trans_tpl', layers_with_attention.TransformerLayer.Params(),
             'Transformer Layer params.')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define(
        'residual_dropout_prob', 0.0, 'Dropout prob to the output of '
        'each sub-layer before it is added to the sub-layer input.')
    p.Define(
        'atten_dropout_prob', 0.0, 'Dropout prob to the attention '
        'weights in each Transformer attention sub-layer.')
    p.Define(
        'relu_dropout_prob', 0.0, 'Dropout prob to the inner layer '
        'output (ReLU activation) in each Transformer feed-forward '
        'sub-layer.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'The softmax layer params.')
    p.Define(
        'random_seed', None,
        'If set, this decides the random seed to apply in various random '
        'ops. Set this random_seed only for unittests.')

    # Default config for the transformer layers.
    p.trans_tpl.is_decoder = False
    p.trans_tpl.mask_self_atten = True
    p.trans_tpl.tr_atten_tpl.num_attention_heads = 8
    p.trans_tpl.tr_atten_tpl.atten_tpl.enable_ctx_pre_proj = True
    p.trans_tpl.tr_atten_tpl.atten_tpl.enable_ctx_post_proj = True
    p.trans_tpl.tr_fflayer_tpl.hidden_dim = 2048

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerLmNoEmbedding, self).__init__(params)
    p = self.params
    p.trans_tpl.tr_atten_tpl.residual_dropout_prob = p.residual_dropout_prob
    p.trans_tpl.tr_atten_tpl.atten_dropout_prob = p.atten_dropout_prob
    p.trans_tpl.tr_fflayer_tpl.residual_dropout_prob = p.residual_dropout_prob
    p.trans_tpl.tr_fflayer_tpl.relu_dropout_prob = p.relu_dropout_prob

    with tf.variable_scope(p.name):
      p.position_emb.embedding_dim = p.model_dim
      self.CreateChild('position_emb', p.position_emb)

      dropout_tpl = layers.DropoutLayer.Params().Set(
          keep_prob=(1.0 - p.input_dropout_prob), seed=p.random_seed)
      self.CreateChild('input_dropout', dropout_tpl)

      params_trans_layers = []
      for i in range(p.num_trans_layers):
        params = p.trans_tpl.Copy()
        params.source_dim = p.model_dim
        params.name = 'layer_%d' % i
        params.random_seed = p.random_seed
        params_trans_layers.append(params)
      self.CreateChildren('trans', params_trans_layers)

      p.softmax.input_dim = p.model_dim
      p.softmax.num_classes = p.vocab_size
      self.CreateChild('softmax', p.softmax)

  def zero_state(self, batch_size):
    p = self.params
    return py_utils.NestedMap({
        'layer_%d' % layer: py_utils.NestedMap({
            'key': tf.zeros([batch_size, 0, p.model_dim]),
            'value': tf.zeros([batch_size, 0, p.model_dim]),
        }) for layer in range(p.num_trans_layers)
    })

  @classmethod
  def StepOutputDimension(cls, params):
    return py_utils.NestedMap(
        logits=params.vocab_size, last_hidden=params.softmax.input_dim)

  def Step(self, theta, inputs, paddings, state0, *args, **kwargs):
    """FProp one step.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: a tensor of shape [batch, model_dim].
      paddings: a 0/1 tensor of shape [batch]. Unused here.
      state0: A NestedMap containing the prefix states up to step t-1.
      *args: optional extra arguments.
      **kwargs: optional extra keyword arguments.

    Returns:
      output: A NestedMap with fields.
        logits: [batch, vocab_size].
        last_hidden: [batch, model_dim].
      state1: The updated prefix states including step t.
    """

    _, prefix_len = py_utils.GetShape(state0['layer_0'].key, 2)
    # [1, model_dim]
    posit_embs = self.position_emb.FProp(theta.position_emb,
                                         prefix_len + 1)[-1:, :]
    # [batch, model_dim]
    input_embs = inputs + posit_embs
    input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

    # Make a copy of the input.
    state1 = state0.Pack(state0.Flatten())

    layer_in = input_embs
    for i, (layer, layer_theta) in enumerate(zip(self.trans, theta.trans)):
      layer_prefix_states = state0['layer_%i' % i]
      # [batch, model_dim]
      layer_out, _, updated_prefix_states = layer.ExtendStep(
          layer_theta, layer_in, layer_prefix_states)
      state1['layer_%i' % i] = updated_prefix_states
      layer_in = layer_out

    # [batch, vocab_size]
    logits = self.softmax.Logits(theta=theta.softmax, inputs=layer_out)

    output = py_utils.NestedMap(logits=logits, last_hidden=layer_out)
    return output, state1

  def FProp(self, theta, inputs, paddings, state0=None, labels=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: Input activation. A tensor of shape [time, batch, model_dim].
      paddings: A 0/1 tensor of shape [time, batch].
      state0: Not used for Transformer.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.

    Returns:
      If labels is not None, returns (xent_output, None), where
      xent_output is a NestedMap as defined by SoftmaxLayer's return
      value. Otherwise, xent_output only contains the softmax logits.
    """
    p = self.params
    inputs = py_utils.HasRank(inputs, 3)
    seqlen, batch, _ = tf.unstack(tf.shape(inputs), num=3)
    inputs = py_utils.HasShape(inputs, [seqlen, batch, p.model_dim])
    paddings = py_utils.HasShape(paddings, [seqlen, batch])

    # [time, 1, model_dim]
    posit_embs = tf.expand_dims(
        self.position_emb.FProp(theta.position_emb, seqlen), 1)
    # [time, batch, model_dim]
    input_embs = inputs + posit_embs
    input_embs = self.input_dropout.FProp(theta.input_dropout, input_embs)

    layer_in = input_embs
    for layer, layer_theta in zip(self.trans, theta.trans):
      # [time, batch, model_dim]
      layer_out, _ = layer.FProp(layer_theta, layer_in, paddings)
      layer_in = layer_out

    if labels is None:
      # We can only compute the logits here.
      logits = self.softmax.Logits(
          theta=theta.softmax,
          inputs=tf.reshape(layer_out, [seqlen * batch, -1]))
      xent_output = py_utils.NestedMap(
          logits=tf.reshape(logits, [seqlen, batch, -1]))
    elif 'class_ids' in labels:
      xent_output = self.softmax.FProp(
          theta=theta.softmax,
          inputs=layer_out,
          class_weights=labels.class_weights,
          class_ids=labels.class_ids)
    else:
      assert 'class_probabilities' in labels
      xent_output = self.softmax.FProp(
          theta=theta.softmax,
          inputs=layer_out,
          class_weights=labels.class_weights,
          class_probabilities=labels.class_probabilities)
    xent_output.last_hidden = layer_out
    return xent_output, None


class TransformerLm(TransformerLmNoEmbedding):
  """Stacked RNN based language model layer."""

  @classmethod
  def Params(cls):
    p = super(TransformerLm, cls).Params()
    p.Define('emb', layers.EmbeddingLayer.Params(),
             'The embedding layer params.')
    p.emb.max_num_shards = 1
    return p

  @classmethod
  def CommonParams(cls,
                   model_dim,
                   hidden_dim,
                   num_heads,
                   num_layers,
                   learning_rate,
                   warmup_steps,
                   vocab_size,
                   input_dropout_prob=0.0,
                   residual_dropout_prob=0.1,
                   atten_dropout_prob=0.0,
                   relu_dropout_prob=0.0,
                   softmax_max_alloc=None):
    """Common setup for Transformer language models.

    Args:
      model_dim: model dimension.
      hidden_dim: hidden dimension of feed-forward inner layer.
      num_heads: number of attention heads.
      num_layers: number of layers in the transformer LM.
      learning_rate: learning rate.
      warmup_steps: warmup steps for TransformerLearningRateSchedule.
      vocab_size: vocab size.
      input_dropout_prob: dropout prob to the sums of the token embeddings and
        the position embeddings.
      residual_dropout_prob: dropout prob to the output of each sub-layer before
        it is added to the sub-layer input.
      atten_dropout_prob: dropout prob to the attention weights in each
        Transformer attention sub-layer.
      relu_dropout_prob: dropout prob to the inner layer output (ReLU
        activation) in each Transformer feed-forward sub-layer.
      softmax_max_alloc: If set to a positive integer the soft-max
        computation is chunked into allocations of at most
        softmax_max_alloc; when left to its default value of None no
        chunking is done.

    Returns:
      A Params object containing the parameters that set up a Transformer LM.
    """
    p = cls.Params()
    p.name = 'transformerlm'

    p.model_dim = model_dim
    p.vocab_size = vocab_size
    p.num_trans_layers = num_layers
    p.input_dropout_prob = input_dropout_prob
    p.residual_dropout_prob = residual_dropout_prob
    p.atten_dropout_prob = atten_dropout_prob
    p.relu_dropout_prob = relu_dropout_prob

    default_params_init = py_utils.WeightInit.Xavier(1.0)
    emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(p.model_dim))
    p.emb.Set(
        vocab_size=vocab_size,
        embedding_dim=p.model_dim,
        max_num_shards=16,
        params_init=emb_params_init,
        scale_sqrt_depth=True)

    p.position_emb.Set(embedding_dim=p.model_dim, trainable_scaling=False)

    p.trans_tpl.is_decoder = False
    p.trans_tpl.mask_self_atten = True

    p.trans_tpl.tr_atten_tpl.Set(
        num_attention_heads=num_heads, params_init=default_params_init)

    p.trans_tpl.tr_atten_tpl.atten_tpl.Set(
        enable_ctx_pre_proj=True, enable_ctx_post_proj=True)

    p.trans_tpl.tr_fflayer_tpl.Set(
        hidden_dim=hidden_dim, params_init=default_params_init)

    p.softmax.Set(
        num_classes=vocab_size, num_shards=16, params_init=emb_params_init)

    if softmax_max_alloc:
      # If the vocab is very large, computes the softmax chunk-by-chunk.
      p.softmax.chunk_size = max(1, int(softmax_max_alloc / vocab_size))
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerLm, self).__init__(params)
    p = self.params

    assert p.emb.vocab_size == p.vocab_size, ('{} vs. {}'.format(
        p.emb.vocab_size, p.vocab_size))
    assert p.emb.embedding_dim == p.position_emb.embedding_dim, (
        '{} vs. {}'.format(p.emb.embedding_dim, p.position_emb.embedding_dim))
    assert p.emb.embedding_dim == p.model_dim, ('{} vs. {}'.format(
        p.emb.embedding_dim, p.model_dim))

    with tf.variable_scope(p.name):
      self.CreateChild('emb', p.emb)

  def FProp(self, theta, inputs, paddings, state0=None, labels=None):
    """Computes xent loss given the language model input activations.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.
      inputs: Input ids. An int32 tensor of shape [time, batch].
      paddings: A 0/1 tensor of shape [time, batch].
      state0: Not used for Transformer.
      labels: If not None, a NestedMap contains the following fields:
        class_weights - a tensor with shape [time, batch] containing
          the weights for each target word.
        class_ids - a tensor with shape [time, batch] of int32 dtype
          containing the target class labels.
        class_probabilities - a tensor with shape [time, batch, vocab_size]
          of float values indicating class-membership probabilities.

    Returns:
      If labels is not None, returns (xent_output, state1), where
      xent_output is a NestedMap as defined by SoftmaxLayer's return
      value and state1 is the next recurrent state. Otherwise,
      xent_output only contains the softmax logits.
    """
    ids = py_utils.HasRank(inputs, 2)
    paddings = py_utils.HasShape(paddings, tf.shape(ids))
    activation = self.emb.EmbLookup(theta.emb, ids)
    return super(TransformerLm, self).FProp(
        theta, activation, paddings, labels=labels)



class HRREmbeddingLayer(base_layer.LayerBase):
  """HRR embedding layer"""

  @classmethod
  def Params(cls):
    p = super(HRREmbeddingLayer, cls).Params()
    p.Define('embedding_dim', 0, 'Embedding size')
    p.Define('num_roles', 0, 'Number of different roles (n)')
    # TODO(jmluo)
    # might want to use different m values for different roles.
    p.Define('num_fillers_per_role', 20,
             'Number of different fillers for each role (m)')
    p.Define('e_l', layers.EmbeddingLayer.Params(), 'Lexicalized embedding')
    # note that s is used num_roles times
    p.Define('s', layers.EmbeddingLayer.Params(), 'Signature embedding')
    # p.Define('rs', layers.EmbeddingLayer.Params(), 'Role signature')
    p.Define('mode', 'basic', 'Modes')
    p.Define('merge', False, 'Flag to merge all collections of filler matrices into a big one')
    # TODO(jmluo)
    p.Define('vocab_size', 0, 'Vocabulary size')
    p.Define('actual_shards', -1, 'Actual number of shards used. This should not be specified, but computed during __init__ call')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(HRREmbeddingLayer, self).__init__(params)
    p = self.params
    assert p.embedding_dim > 0
    assert p.num_roles > 0
    assert p.num_fillers_per_role > 0
    assert p.vocab_size > 0
    assert p.e_l.vocab_size == p.vocab_size == p.s.vocab_size
    assert p.e_l.embedding_dim == p.embedding_dim
    assert p.s.embedding_dim == p.num_fillers_per_role * p.num_roles
    assert p.mode in ['basic', 'rs', 'dec_only']
    if p.merge:
      assert p.mode == 'rs', 'Other modes not supported yet'

    r_pc = py_utils.WeightParams(
        shape=[p.num_roles, p.embedding_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    F_pc = py_utils.WeightParams(
        shape=[p.num_roles, p.num_fillers_per_role, p.embedding_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])

    with tf.variable_scope(p.name):
      # TODO(jmluo) disabled for now
      # self.CreateChild('e_l', p.e_l)

      if p.mode == 'rs':
        rr_pc = py_utils.WeightParams(
            shape=[p.num_roles, p.embedding_dim],
            init=p.params_init,
            dtype=p.dtype,
            collections=[self.__class__.__name__ + '_vars'])
        rs = p.s.Copy()
        rs.embedding_dim = 2 * p.num_roles
        rs.name = 'rs'
        rs.params_init = py_utils.WeightInit.PositiveUniform()
        # const = [[1., 0.], [0., 1.]]
        # const = [const] * rs.vocab_size
        # rs.params_init = py_utils.WeightInit.Constant(scale=const)
        # rs.trainable = False

        self.CreateChild('rs', rs)
        self.CreateVariable('rR', rr_pc)
        self.CreateChild('s', p.s)
        self.CreateVariable('F', F_pc)
      elif p.mode == 'basic':
        self.CreateChild('s', p.s)
        self.CreateVariable('r', r_pc)
        self.CreateVariable('F', F_pc)
      else:
        self.CreateChild('e_l', p.e_l)
        self.CreateVariable('r', r_pc)


  def _circular_conv(self, a, b):
    with tf.name_scope('circular_conv'):
      a_fft = tf.fft(tf.complex(a, 0.0))
      b_fft = tf.fft(tf.complex(b, 0.0))
      ifft = tf.ifft(a_fft * b_fft)
      res = tf.cast(tf.real(ifft), 'float32')
    return res

  def _circular_corr(self, a, b):
    with tf.name_scope('circular_corr'):
      a_fft = tf.conj(tf.fft(tf.complex(a, 0.0)))
      b_fft = tf.fft(tf.complex(b, 0.0))
      ifft = tf.ifft(a_fft * b_fft)
      res = tf.cast(tf.real(ifft), 'float32')
    return res

  def decode(self, x, r):
    # r_weight: nr x d
    # x: ? x d
    with tf.name_scope('HRR_decode'):
      res = self._circular_corr(r, x)
    return res

  @staticmethod
  def static_circular_conv(a, b):
    with tf.name_scope('static_circular_conv'):
      a_fft = tf.fft(tf.complex(a, 0.0))
      b_fft = tf.fft(tf.complex(b, 0.0))
      ifft = tf.ifft(a_fft * b_fft)
      res = tf.cast(tf.real(ifft), 'float32')
    return res

  @staticmethod
  def static_circular_corr(a, b):
    with tf.name_scope('static_circular_corr'):
      a_fft = tf.conj(tf.fft(tf.complex(a, 0.0)))
      b_fft = tf.fft(tf.complex(b, 0.0))
      ifft = tf.ifft(a_fft * b_fft)
      res = tf.cast(tf.real(ifft), 'float32')
    return res

  @staticmethod
  def static_decode(x, r):
    # r_weight: nr x d
    # x: ? x d
    with tf.name_scope('static_HRR_decode'):
      res = HRREmbeddingLayer.static_circular_corr(r, x)
    return res

  def EmbLookup(self, theta, ids, role_anneal=False):
    """Looks up embedding vectors for ids.

    Args:
      theta: Named tuple with the weight matrix for the embedding.
      ids: A rank-N int32 tensor.

    Returns:
      embs: A rank-(N+1) params.dtype tensor. embs[indices, :] is the
        embedding vector for ids[indices].
    """
    p = self.params

    with tf.name_scope('HRR_emb_lookup'):
      emb_weights = self._Emb2Weight(theta, role_anneal=role_anneal)


      emb = tf.nn.embedding_lookup(emb_weights.e, ids, partition_strategy=p.s.partition_strategy)
      s_cat = None

    # distribution constraint
    # mean, variance = tf.nn.moments(emb, axes=[2]) # size: l x bs, l x bs
    # mean = tf.expand_dims(mean, axis=2)
    # variance = tf.expand_dims(variance, axis=2)
    # d = tf.shape(emb)[2]
    # (emb - mean) / tf.sqrt(variance * d)


    return emb, s_cat, emb_weights

  def _Emb2Weight(self, theta, role_anneal=False):
    p = self.params
    e_weights = list()
    rf_weights = list()
    f_weights = list()

    if p.mode == 'rs':
      bases = self._circular_conv(tf.expand_dims(theta.rR, axis=1), theta.F) # size: nr x nf x d
      for rs_shard, s_shard in zip(theta.rs.wm, theta.s.wm):
        rs_shard = tf.reshape(rs_shard, [-1, p.num_roles, 2])
        s_shard = tf.reshape(s_shard, [-1, p.num_roles, p.num_fillers_per_role])
        coeffs = tf.matmul(tf.transpose(rs_shard, perm=[0, 2, 1]), s_shard) # size: V/n_shards x nr x nf
        coeffs_t = tf.transpose(coeffs, [1, 0, 2])
        rf_shard = tf.matmul(coeffs_t, bases) # size: nr x V/n_shards x d
        e_shard = tf.reduce_sum(rf_shard, axis=0)
        # old
        # rf_shard = self._circular_conv(hid_r_shard, hid_f_shard)
        # e_shard = tf.reduce_sum(rf_shard, axis=1) # size: V/n_shards x d
        e_weights.append(e_shard)
        rf_weights.append(rf_shard)
        # real f shard
        f_shard = self._circular_corr(theta.rR, tf.expand_dims(e_shard, axis=1))
        f_weights.append(f_shard)
        # f_weights.append(hid_f_shard)
        r_weights = theta.rR
    elif p.mode == 'basic':
      for s_shard in theta.s.wm:
        s_shard = tf.reshape(s_shard, [-1, p.num_roles, p.num_fillers_per_role])
        f_shard_list = list()
        for role_ind in xrange(p.num_roles):
          f_shard_i = tf.matmul(s_shard[:, role_ind], theta.F[role_ind]) # size: V/n_shards x d
          f_shard_list.append(f_shard_i)
        f_shard = tf.stack(f_shard_list, axis=1) # size: V/n_shards x nr x d
        # TODO(jmluo) revert this
        # if role_anneal:
        #   prob_1 = tf.ones(shape=tf.shape(f_shard_list[0]))
        #   global_step = tf.to_float(py_utils.GetOrCreateGlobalStep())
        #   temperature = tf.minimum(tf.constant(3000.0), global_step) / 3000
        #   probs = tf.stack([prob_1, prob_1 * temperature], axis=1)
        #   f_shard = f_shard * probs

        # f_shard = tf.transpose(tf.matmul(tf.transpose(s_shard, perm=[1, 0, 2]), theta.F), perm=[1, 0, 2]) # |V|/n_shards x nr x d
        # f_shard = tf.reduce_sum(s_shard * theta.F, axis=2) # size: V/n_shards x nr x d
        rf_shard = self._circular_conv(theta.r, f_shard)
        e_shard = tf.reduce_sum(rf_shard, axis=1)
        e_weights.append(e_shard)
        rf_weights.append(rf_shard)
        f_weights.append(f_shard)
        # noisy_f_shard = self._circular_corr(theta.r, tf.expand_dims(e_shard, axis=1))
        # f_weights.append(noisy_f_shard)
        r_weights = theta.r
    else:
      e_weights = list()
      f_weights = list()
      r_weights = theta.r
      for e_shard in theta.e_l.wm:
        e_weights.append(e_shard)
        e_shard = tf.reshape(e_shard, [-1, 1, p.embedding_dim])
        f_shard = self._circular_corr(theta.r, e_shard) # size: V/n_shard x nr x d
        f_weights.append(f_shard)

    # NOTE all following weights are sharded along the |V| axis, except r_weights which are
    # not sharded.
    return py_utils.NestedMap(e=e_weights,
                              # rf=rf_weights,
                              r=r_weights,
                              f=f_weights)
                                  