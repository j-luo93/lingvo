"""Train word-level LMs on PTB data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import layers
from lingvo.core import lr_schedule
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import tokenizers
from lingvo.tasks.lm import input_generator as lm_inp
from lingvo.tasks.lm import layers as lm_layers
from lingvo.tasks.lm import model

def iter_iter(p, pattern, scale):
  for name, param in p.IterParams():
    if hasattr(param, 'IterParams'):
      if pattern in name:
        d = {name: py_utils.WeightInit.Uniform(scale=scale)}
        p.Set(**d)
      else:
        iter_iter(param, pattern, scale)
    elif isinstance(param, list):
      for cell_p in param:
        if hasattr(cell_p, 'IterParams'):
          cell_p.Set(params_init=py_utils.WeightInit.Uniform(scale=scale))

'''
PTB baseline with tied embeddings
'''
@model_registry.RegisterSingleTaskModel
class PennBaseline(base_model_params.SingleTaskModelParams):
  """Params for training a word-level LM on PTB."""

  # One Billion Words benchmark corpus is available in iq, li and ok.
  CORPUS_DIR = os.path.join('/tmp/lingvo/HRR/',
                            'data/ptb/')
  EMBEDDING_DIM = 650
  MAX_TOKENS = 512
  NUM_EMBEDDING_SHARDS = 1
  NUM_SAMPLED = 0
  NUM_SOFTMAX_SHARDS = 1
  RNN_STATE_DIM = 650
  VOCAB_SIZE = 10000  # includes <epsilon>, vocabulary in fst symtable format
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'vocab.txt')

  @classmethod
  def Train(cls):
    p = lm_inp.LmInput.Params()
    #p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    #p.bucket_batch_limit = [1024, 512, 256, 256, 128, 128, 64, 32, 16]
    p.bucket_upper_bound = [10, 20, 30, 40, 50]#, 100]
    p.bucket_batch_limit = [128] * len(p.bucket_upper_bound)
    #p.bucket_batch_limit = [64] * len(p.bucket_upper_bound) # [1024, 512, 256, 256, 128, 128, 64, 32, 16]
    p.file_buffer_size = 10000000
    p.file_parallelism = 10
    p.file_pattern = 'text:' + os.path.join(
        cls.CORPUS_DIR, 'train.txt')
    p.name = 'ptb_train_set'
    p.tokenizer = tokenizers.VocabFileTokenizer.Params()
    p.tokenizer.normalization = ''
    p.num_batcher_threads = 16
    p.target_max_length = cls.MAX_TOKENS
    p.tokenizer.target_sos_id = 1
    p.tokenizer.target_eos_id = 2
    p.tokenizer.target_unk_id = 3
    p.tokenizer.token_vocab_filepath = cls.WORD_VOCAB
    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    return p

  @classmethod
  def Dev(cls):
    p = cls.Train()
    # Use small batches for eval.
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [128, 64, 32, 32, 16, 16, 4, 2, 1]
    p.file_buffer_size = 1
    p.file_parallelism = 1
    p.file_pattern = 'text:' + os.path.join(
        cls.CORPUS_DIR, 'dev.txt')
    p.name = 'ptb_dev_set'
    p.num_batcher_threads = 1
    p.num_samples = 3370  # Number of sentences to evaluate on.
    return p

  @classmethod
  def Test(cls):
    p = cls.Train()
    # Use small batches for eval.
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [128, 64, 32, 32, 16, 16, 4, 2, 1]
    p.file_buffer_size = 1
    p.file_parallelism = 1
    p.file_pattern = 'text:' + os.path.join(
        cls.CORPUS_DIR, 'test.txt')
    p.name = 'ptb_test_set'
    p.num_batcher_threads = 1
    p.num_samples = 3761  # Number of sentences to evaluate on.
    return p

  @classmethod
  def get_lm_params(cls, num_layers, keep_prob):
    lm = lm_layers.RnnLm.CommonParams(
        vocab_size=cls.VOCAB_SIZE,
        emb_dim=cls.EMBEDDING_DIM,
        num_layers=num_layers,
        residual_start=3,  # disable residuals
        rnn_dims=cls.EMBEDDING_DIM,
        rnn_hidden_dims=0)#cls.RNN_STATE_DIM)

    # Input embedding needs to be sharded.
    lm.emb.max_num_shards = cls.NUM_EMBEDDING_SHARDS
    lm.embedding_dropout_keep_prob = keep_prob
    # Match the initialization in third_party code.
    lm.emb.params_init = py_utils.WeightInit.UniformUnitScaling(
        1.0 * cls.NUM_EMBEDDING_SHARDS)

    # We also want dropout after each of the RNN layers.
    lm.rnns.dropout.keep_prob = keep_prob
    
    num_input_dim = lm.softmax.input_dim
    # Tie input and output embeddings
    lm.softmax = layers.SimpleFullSoftmax.Params()
    lm.softmax.input_dim = num_input_dim
    lm.softmax.num_classes = cls.VOCAB_SIZE
    lm.softmax.num_sampled = cls.NUM_SAMPLED
    lm.softmax.num_shards = cls.NUM_SOFTMAX_SHARDS
    lm.tie = True
    lm.softmax.tie = True
    # NOTE this makes tying input and output embeddings much easier
    lm.emb.partition_strategy = 'div'
    assert lm.softmax.num_classes % lm.softmax.num_shards == 0
    return lm
    
  @classmethod
  def Task(cls):
    p = model.LanguageModel.Params()
    p.name = 'ptb_word_level_lm'
    p.eval.samples_per_summary = 10000

    p.lm = cls.get_lm_params(2, 0.5)
    
    # Adjusts training params.
    tp = p.train
    # Use raw loss: sum logP across tokens in a batch but average across splits.
    tp.sum_loss_across_tokens_in_batch = True
    # Disable any so called "clipping" (gradient scaling really).
    # tp.clip_gradient_norm_to_value = 0.0
    # tp.grad_norm_to_clip_to_zero = 0.0
    # # Do clip the LSTM gradients.
    # tp.max_lstm_gradient_norm = 16
    # # Straight Adagrad; very sensitive to initial accumulator value, the default
    # # 0.1 value is far from adequate.
    # # TODO(ciprianchelba): tune accumulator value, learning rate, clipping
    # # threshold.
    #tp.lr_schedule = (
    #    lr_schedule.PiecewiseConstantLearningRateSchedule.Params().Set(
    #        boundaries=[], values=[1.0]))
    # tp.lr_schedule = (
    #    lr_schedule.DevBasedSchedule.Params().Set(decay=0.9, window=100))
    p.train.lr_schedule = (
        lr_schedule.DevBasedSchedule.Params().Set(decay=0.8, window=100))
    tp.lr_schedule.metric_history.local_filesystem = True

        # lr_schedule.ExponentialLearningRateSchedule.Params().Set(start=(0, 1.0), limit=(10000, 0.01)))
    # tp.learning_rate = 0.02
    # p.train.optimizer = optimizer.SGD.Params()
    p.train.lr_schedule.metric_history.local_filesystem = True
    # p.train.learning_rate = 1.0
    
    tp.l2_regularizer_weight = None  # No regularization.
    # tp.optimizer = optimizer.Adagrad.Params()
    #tp.save_interval_seconds = 20
    #tp.summary_interval_steps = 20
    
    ######################### stuff I added to the base ########################

    p.train.optimizer = optimizer.SGD.Params()
    p.train.learning_rate = 1.0
    #p.train.optimizer = optimizer.Adam.Params()
    #p.train.learning_rate = 5e-4#2e-3

    # HACK
    # use uniform initializer (-scale, scale)
    iter_iter(p, 'params_init', 0.05)

    # forget gate bias set to 1.0
    for param in p.lm.rnns.cell_tpl:
      # make it compatible with cudnnlstm
      param.forget_gate_bias = 0
      param.cell_value_cap = None 
      #param.zo_prob = 0.15

    ## gradient norm clipping
    p.train.clip_gradient_norm_to_value = 5.0#10.0
    p.train.grad_norm_to_clip_to_zero = 0.0
    p.train.max_lstm_gradient_norm = 0

    p.train.save_interval_seconds = 100
    p.train.summary_interval_steps = 100
    return p

@model_registry.RegisterSingleTaskModel
class PennBaselineCont(PennBaseline):
  
  @classmethod
  def Train(cls):
    p = super(PennBaselineCont, cls).Train()
    p.contiguous = True
    p.ckpt = 'HRR/data/ptb/train.ckpt'
    p.data = 'ids_train'
    p.num_steps = 35
    p.data_shape = (929564,)
    return p
    
  @classmethod
  def Dev(cls):
    p = super(PennBaselineCont, cls).Dev()
    p.contiguous = True
    p.eval = True
    p.ckpt = 'HRR/data/ptb/dev.ckpt'
    p.data = 'ids_dev'
    p.num_steps = 35
    p.data_shape = (73760,)
    p.num_samples = int(math.ceil((p.data_shape[0] // p.batch_size) / p.num_steps) * p.batch_size)
    return p
    
  @classmethod
  def Test(cls):
    p = super(PennBaselineCont, cls).Test()
    p.contiguous = True
    p.eval = True
    p.ckpt = 'HRR/data/ptb/test.ckpt'
    p.data = 'ids_test'
    p.num_steps = 35
    p.data_shape = (82430,)
    p.num_samples = int(math.ceil((p.data_shape[0] // p.batch_size) / p.num_steps) * p.batch_size)
    return p
  
  @classmethod
  def Task(cls):
    p = super(PennBaselineCont, cls).Task()
    p.contiguous = True
    p.train.save_interval_seconds = 200
    p.train.summary_interval_steps = 200
    p.train.lr_schedule = (
        lr_schedule.DevBasedSchedule.Params().Set(decay=0.8, window=1000))
    p.train.lr_schedule.metric_history.local_filesystem = True
    return p

# helper function: add some HRR-specific options
def HRRify(p, cls):
  old_params = p.lm.emb
  hrr = lm_layers.HRREmbeddingLayer.Params()
  hrr.s = old_params.Copy()
  hrr.e_l = old_params.Copy()
  hrr.vocab_size = hrr.e_l.vocab_size = cls.VOCAB_SIZE
  hrr.s.vocab_size = cls.VOCAB_SIZE
  hrr.embedding_dim = hrr.e_l.embedding_dim = cls.EMBEDDING_DIM
  hrr.num_roles = cls.NUM_ROLES
  hrr.num_fillers_per_role = cls.NUM_FILLERS_PER_ROLE
  hrr.s.embedding_dim = cls.NUM_FILLERS_PER_ROLE * cls.NUM_ROLES
  hrr.lazy = False
  p.lm.emb = hrr
  p.lm.num_word_roles = cls.NUM_ROLES
  p.lm.softmax.num_roles = cls.NUM_ROLES
  p.lm.softmax.input_dim *= cls.NUM_ROLES # size: |V| x nr*d
  # dropout for f_noisy
  p.lm.decoded_filler_keep_prob = 0.9#5
  #p.lm.rnns.drop_last = False
  # annealing for second role
  p.lm.softmax.role_anneal_steps = [15000]#[3000]
  #p.lm.softmax.role_anneal_steps = [15000]
  # isometric loss
  p.train.isometric = 1e2
  return p

'''
Word level HRR, with num_fillers = 50
'''
@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNF50(PennBaseline):
  """Use sampled soft-max in training."""

  NUM_ROLES = 2
  NUM_FILLERS_PER_ROLE = 50

  @classmethod
  def Task(cls):
    p = super(PennHRRWordLevelNF50, cls).Task()
    p = HRRify(p, cls)
    p.lm.softmax.role_anneal_steps = [3000]
    return p

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNF100(PennHRRWordLevelNF50): 

  NUM_FILLERS_PER_ROLE = 100

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNF250(PennHRRWordLevelNF50): 

  NUM_FILLERS_PER_ROLE = 250

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNF250FixedBases(PennHRRWordLevelNF250): 

  @classmethod
  def Task(cls):
    p = super(PennHRRWordLevelNF250FixedBases, cls).Task()  
    p.lm.emb.trainable_basis = False
    p.train.isometric = 0.
    return p

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNF320FixedBases(PennHRRWordLevelNF250FixedBases): 
  NUM_FILLERS_PER_ROLE = 320

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNF50FixedBases(PennHRRWordLevelNF320FixedBases): 
  NUM_FILLERS_PER_ROLE = 50

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNF200FixedBases(PennHRRWordLevelNF320FixedBases): 
  NUM_FILLERS_PER_ROLE = 200

'''
Word level HRR with contiguous input, with num_fillers = 50
'''
@model_registry.RegisterSingleTaskModel
class PennHRRContWordLevelNF320FixedBases(PennBaselineCont):
  """Use sampled soft-max in training."""

  NUM_ROLES = 2
  NUM_FILLERS_PER_ROLE = 320

  @classmethod
  def Task(cls):
    p = super(PennHRRContWordLevelNF320FixedBases, cls).Task()
    p = HRRify(p, cls)
    p.lm.emb.trainable_basis = False
    p.train.isometric = 0.
    return p

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNR4NF125(PennHRRWordLevelNF250):
  NUM_FILLERS_PER_ROLE = 125
  NUM_ROLES = 4
  
  @classmethod
  def Task(cls):
    p = super(PennHRRWordLevelNR4NF125, cls).Task()
    p.lm.softmax.role_anneal_steps = [3000, 4000, 5000]
    return p

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNR4NF125FixedBases(PennHRRWordLevelNF250FixedBases):
  NUM_FILLERS_PER_ROLE = 125
  NUM_ROLES = 4
  
  @classmethod
  def Task(cls):
    p = super(PennHRRWordLevelNR4NF125FixedBases, cls).Task()
    p.lm.softmax.role_anneal_steps = [3000, 4000, 5000]
    return p

@model_registry.RegisterSingleTaskModel
class PennHRRWordLevelNR4NF160FixedBases(PennHRRWordLevelNR4NF125FixedBases):
  NUM_FILLERS_PER_ROLE = 160
  NUM_ROLES = 4
  
'''
PTB baseline with tied embeddings on tagged data (chunks).
'''
@model_registry.RegisterSingleTaskModel
class PennTaggedBaseline(PennBaseline):

  CORPUS_DIR = os.path.join('/tmp/lingvo/HRR/',
                            'data/ptb-chunk')
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'vocab.txt')
  NUM_SAMPLED = 0#9999
  VOCAB_SIZE = 10000  # includes <epsilon>, vocabulary in fst symtable format

  @classmethod
  def Train(cls):
    p = super(PennTaggedBaseline, cls).Train()
    p.use_chunks = True
    p.bucket_upper_bound = [b * 2 for b in p.bucket_upper_bound ]
    return p

#  @classmethod
#  def Dev(cls):
#    p = super(PennTaggedBaseline, cls).Dev()
#    p.num_samples = 1006
#    return p
#
#  @classmethod
#  def Test(cls):
#    p = super(PennTaggedBaseline, cls).Test()
#    p.num_samples = 1006
#    return p

'''
word-level HRR on chunk data
'''
@model_registry.RegisterSingleTaskModel
class PennTaggedHRRWordLevelNF50(PennHRRWordLevelNF50):

  CORPUS_DIR = os.path.join('/tmp/lingvo/HRR/',
                            'data/ptb-chunk')
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'vocab.txt')
  NUM_SAMPLED = 0#9999
  VOCAB_SIZE = 10000  # includes <epsilon>, vocabulary in fst symtable format


  @classmethod
  def Train(cls):
    p = super(PennTaggedHRRWordLevelNF50, cls).Train()
    p.use_chunks = True
    p.bucket_upper_bound = [b * 2 for b in p.bucket_upper_bound ]
    return p

#  @classmethod
#  def Dev(cls):
#    p = super(PennTaggedHRRWordLevelNF50, cls).Dev()
#    p.num_samples = 1006
#    return p
#
#  @classmethod
#  def Test(cls):
#    p = super(PennTaggedHRRWordLevelNF50, cls).Test()
#    p.num_samples = 1006
#    return p


def chunkify(p, cls):
  p.train.chunk_loss_anneal = 1.0#3000.0
  p.lm.use_chunks = True
  p.lm.num_sent_roles = 2
  p.lm.sent_role_anneal_steps = [1.0]#3000.0]
  p.lm.num_word_roles = cls.NUM_ROLES
  #for tpl in p.lm.rnns.cell_tpl:
  #p.lm.rnns.cell_tpl[-1].num_output_nodes = 2 * cls.EMBEDDING_DIM
  
  p.lm.pred_proj.input_dim = 2 * cls.EMBEDDING_DIM
  p.lm.pred_proj.output_dim = cls.EMBEDDING_DIM
  p.lm.pred_proj.activation = 'TANH'
  p.lm.pred_proj.batch_norm = False
  return p

'''
Full model. Chunk-level HRR. 
'''
@model_registry.RegisterSingleTaskModel
class PennTaggedHRRChunkLevelNF50(PennTaggedHRRWordLevelNF50):

  @classmethod
  def Task(cls):
    p = super(PennTaggedHRRChunkLevelNF50, cls).Task()
    p = chunkify(p, cls)
    # TODO(jmluo) need to rename this -- I'm still using chunk loss but there is no r_o prediction.
    return p

@model_registry.RegisterSingleTaskModel
class PennTaggedHRRChunkLevelNF320FixedBases(PennTaggedHRRChunkLevelNF50):
  NUM_FILLERS_PER_ROLE = 320

  @classmethod
  def Task(cls):
    p = super(PennTaggedHRRChunkLevelNF320FixedBases, cls).Task()
    p.lm.emb.trainable_basis = False
    p.lm.trainable_basis = False
    p.train.isometric = 0.
    p.partial_restore = True
    return p

@model_registry.RegisterSingleTaskModel
class PennTaggedHRRChunkLevelNF50RNN(PennTaggedHRRChunkLevelNF50):
  
  @classmethod
  def Task(cls):
    p = super(PennTaggedHRRChunkLevelNF50RNN, cls).Task()
    p.lm.pred_mode = 'rnn'
    
    p.lm.pred_rnn = PennTaggedHRRChunkLevelNF50RNN.get_lm_params(1, 0.5).rnns
    p.lm.pred_rnn.dropout.keep_prob = 0.5
    iter_iter(p, 'params_init', 0.05)
    return p

@model_registry.RegisterSingleTaskModel
class PennTaggedHRRChunkLevelNF320RNN(PennTaggedHRRChunkLevelNF50):
  NUM_FILLERS_PER_ROLE = 320

@model_registry.RegisterSingleTaskModel
class PennTaggedHRRChunkLevelNF320RNNFixedBases(PennTaggedHRRChunkLevelNF320RNN):

  @classmethod
  def Task(cls):
    p = super(PennTaggedHRRChunkLevelNF320RNNFixedBases, cls).Task()
    p.lm.emb.trainable_basis = False
    p.lm.trainable_basis = False
    p.train.isometric = 0.
    return p
    

@model_registry.RegisterSingleTaskModel
class PennTaggedHRRChunkLevelNR4NF160RNNFixedBases(PennTaggedHRRChunkLevelNF320RNNFixedBases):
  NUM_FILLERS_PER_ROLE = 125
  NUM_ROLES = 4
  
  @classmethod
  def Task(cls):
    p = super(PennTaggedHRRChunkLevelNR4NF160RNNFixedBases, cls).Task()
    p.lm.softmax.role_anneal_steps = [3000, 4000, 5000]
    return p
                                                                                                                                                                                                                                                                                       
                                                                                                                                   
