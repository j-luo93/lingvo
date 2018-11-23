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
"""Trainer.

To run locally:
$ bazel build -c opt lingvo:trainer
$ bazel-bin/lingvo/trainer --logtostderr --model=image.mnist.LeNet5 --mode=sync
--logdir=/tmp/lenet5 --run_locally=cpu

To use GPU, add --config=cuda to build command and set --run_locally=gpu.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import re
import threading
import time

import numpy as np
import six
import tensorflow as tf

from lingvo import base_runner
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from lingvo import base_trial
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import cluster_factory
from lingvo.core import metrics
from lingvo.core import py_utils

tf.flags.DEFINE_string(
    'model', '', 'Name of the model class to train. Must be one of those'
    ' defined in models.py.')
tf.flags.DEFINE_string(
    'model_task_name', '', 'For multitask models: '
    'select task to train/evaluate/decode. '
    'Empty means to sample a task (training only).')
tf.flags.DEFINE_string('logdir', '', 'Log directory.')
tf.flags.DEFINE_bool(
    'interactive', False,
    'If True, enter interactive IPython for the controller job.')

tf.flags.DEFINE_string(
    'run_locally', None,
    'If True, ignores flags below and runs controller and trainer '
    'in the single process.')

tf.flags.DEFINE_string('tf_master', '', 'TF runtime.')
tf.flags.DEFINE_string(
    'cluster_spec', '', 'A tf.train.ClusterSpec to override the master. '
    'The dict is specified as: job=host1:port1,host2:port2,'
    'host3:port3@job2=host3:port4,...')

tf.flags.DEFINE_string(
    'mode', 'async', 'How this trainer binary is used. '
    'async: used in an async training setup; '
    'sync: used in a sync training setup; '
    'shell: an interactive shell for development; '
    'inspect_evaler: print evaler dataset names; '
    'inspect_decoder: print decoder dataset names.')
tf.flags.DEFINE_string('job', None, 'trainer/controller/eval, etc.')
tf.flags.DEFINE_integer('task', 0, 'Task id within the job.')

tf.flags.DEFINE_string('controller_job', '/job:controller', 'Job name.')
tf.flags.DEFINE_integer('controller_gpus', 0, 'Number of controller GPUs.')

tf.flags.DEFINE_string('worker_job', '/job:trainer', 'Job name.')
tf.flags.DEFINE_integer('worker_replicas', 1, 'Number of replicas.')
tf.flags.DEFINE_integer('worker_gpus', 0, 'Number of gpus to use per replica.')
tf.flags.DEFINE_integer('worker_split_size', 1,
                        'Number of devices for one split.')

tf.flags.DEFINE_string('ps_job', '/job:ps', 'Job name')
tf.flags.DEFINE_integer('ps_replicas', 1, 'Number of replicas.')
tf.flags.DEFINE_integer('ps_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_string('input_job', '/job:input', 'Job name')
tf.flags.DEFINE_integer('input_replicas', 0, 'Number of replicas.')

tf.flags.DEFINE_string('evaler_job', '/job:evaler', 'Job name')
tf.flags.DEFINE_integer('evaler_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_integer('evaler_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_string('decoder_job', '/job:decoder', 'Job name')
tf.flags.DEFINE_integer('decoder_replicas', 0, 'Number of replicas.')
tf.flags.DEFINE_integer('decoder_gpus', 0, 'Number of gpus to use per replica.')

tf.flags.DEFINE_bool(
    'evaler_in_same_address_as_controller', False,
    'Whether or not evaler is in the same address space as '
    ' controller. This flag is meant for unittest only.')

FLAGS = tf.flags.FLAGS


def get_tensors_in_checkpoint_file(file_name,
                                   tensor_name,
                                   all_tensors,
                                   all_tensor_names=False):
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
      var_to_shape_map = reader.get_variable_to_shape_map()
      key_tensor_map = dict()
      for key in sorted(var_to_shape_map):
        # print("tensor_name: ", key)
        if all_tensors:
          key_tensor_map[key] = reader.get_tensor(key)
      return key_tensor_map
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      return reader.get_tensor(tensor_name)
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        (any([e in file_name for e in [".index", ".meta", ".data"]]))):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))
      
def get_initializer(model):
  try:
    word_level_ckpt_path = model.params.task.train.word_level_ckpt_path
  except AttributeError:
    word_level_ckpt_path = ''
  if word_level_ckpt_path:
    all_saved_tensors = get_tensors_in_checkpoint_file(word_level_ckpt_path, '', True)
    all_model_vars = {t.name.rstrip(':0'): t for t in model.vars.Flatten()}
    
    # insert chunk tag embeddings:
    prefix = model.params.task.name
    emb_names = sorted(filter(lambda n: n.startswith('%s/lm/emb/s/var_' %prefix), all_saved_tensors.keys()))
    n_shards = len(emb_names)
    expected_n_shards = 8 if prefix.startswith('1b') else 1
    assert n_shards == expected_n_shards
    concat = np.concatenate([all_saved_tensors[name] for name in emb_names], axis=0)
    chunk_tag_emb = np.random.uniform(low=-0.05, high=0.05, size=(3, 640))
    if prefix.startswith('1b'):
      garbage_emb = np.random.uniform(low=-0.05, high=0.05, size=(5, 640)) # garbage appended at the end to make sure it is divided by 8
      adjusted_emb = np.concatenate([concat[:4], chunk_tag_emb, concat[4:], garbage_emb], axis=0)
    else:
      adjusted_emb = np.concatenate([concat[:4], chunk_tag_emb, concat[4:-3]], axis=0)
    assert adjusted_emb.shape[0] % n_shards == 0
    adjusted_emb_shards = np.split(adjusted_emb, n_shards, axis=0)
    
    for i, name in enumerate(emb_names):
      all_saved_tensors[name] = adjusted_emb_shards[i]
    
    # add init ops
    init_ops = list()
    for name, tensor in all_saved_tensors.iteritems():
      if name.startswith(prefix):
        if name.startswith('%s/lm/softmax/bias' %prefix):
          tf.logging.warning('Skipped saved tensor %s' %name)
        elif name.startswith('%s/lm/rnns' %prefix):
          tf.logging.warning('Skipped saved tensor %s' %name)
        elif name in all_model_vars:
          tf.logging.warning('Initialize %s from saved ckpt' %name)
          init_ops.append(tf.assign(all_model_vars[name], all_saved_tensors[name]))
        else:
          tf.logging.warning('%s not found in model vars' %name)
      else:
        tf.logging.warning('Skipped saved tensor %s' %name)
        
    for name in all_model_vars.keys():
      if name not in all_saved_tensors:
        tf.logging.warning('%s not found in saved model' %name)
    
    # change params_init for cudnn lstm
    #wb = all_saved_tensors['%s/lm/rnns_0/wb/var' %prefix]
    #w_pc = py_utils.WeightParams(
    #  shape=wb.shape,
    #  init=py_utils.WeightInit.Constant(scale=wb),
    #  dtype=tf.float32)
    #model.params.task.lm.rnns.params_init = w_pc
    #tf.logging.warning('Changing lower level cudnn lstm params_init')
    
    generic_init_op = tf.tables_initializer()
    init_ops.insert(0, generic_init_op)
    return tf.group(*init_ops)
  else:
    return tf.tables_initializer()

# useful for debugging.
def _StartShell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython  # pylint: disable=g-import-not-at-top

  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def _ModelAnalysis(model):
  """Returns a text showing variable sizes and their total size."""

  class Analyzer(object):

    def __init__(self):
      self._seen_var = {}
      self.total = 0

    def __call__(self, v):
      assert isinstance(v, tf.Variable)
      # pylint: disable=protected-access
      if not v.shape.is_fully_defined():
        # Only Cudnn RNN params lack static shapes.
        if hasattr(v, 'approx_size'):
          size = v.approx_size
        else:
          return '%-20s %10s %s' % (v.shape, 'n/a', v._shared_name)
      else:
        size = v.shape.num_elements()
      if v._shared_name not in self._seen_var:
        self._seen_var[v._shared_name] = size
        self.total += size
      return '%-20s %10d %s' % (v.shape, size, v._shared_name)

  analyzer = Analyzer()
  output = '\n'
  output += model.vars.Transform(analyzer).DebugString()
  output += '\n'
  output += '=' * 100
  output += '\ntotal #params: %10d\n' % (analyzer.total)
  return output, analyzer.total


class Controller(base_runner.BaseRunner):
  """Controller for a training cluster."""

  def __init__(self, *args, **kwargs):
    super(Controller, self).__init__(*args, **kwargs)
    assert not self._model_task_name, 'Controller needs all tasks!'
    self._save_path = os.path.join(self._train_dir, 'ckpt')
    tf.gfile.MakeDirs(self._train_dir)
    self._control_dir = os.path.join(self._logdir, 'control')
    tf.gfile.MakeDirs(self._control_dir)
    self._summary_writer = self._CreateSummaryWriter(self._control_dir)
    self._time_steps = []  # A short history of (timestamp, global_step)

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._model = self.params.cls(self.params)
        self._params = self._model.params
        self._model.ConstructFPropBPropGraph()
        self._saver = self._GetSaver()
        self._summary_op = tf.summary.merge_all()
        self._vars = tf.get_collection(tf.GraphKeys.VARIABLES)
        self._uninitialized = tf.report_uninitialized_variables(self._vars)
        self._initialize_all = tf.global_variables_initializer()
        self.initialize_tables = get_initializer(self._model)
        self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
        self.close_queue_ops = tf.get_collection(py_utils.CLOSE_QUEUE_OPS)

    self._ExportMetrics(params=self.params)
    self._model_analysis, self._total_num_params = _ModelAnalysis(self._model)
    tf.logging.error(self._model_analysis)
    self._WriteToLog(self._model_analysis, self._control_dir,
                     'model_analysis.txt')
    self._WriteToLog(self.params.ToText(), self._control_dir, 'params.txt')
    tf.train.write_graph(self._graph.as_graph_def(), self._control_dir,
                         'train.pbtxt')

  def Start(self):
    self._RunLoop('controller', self._Loop)

  def StartEnqueueOp(self, op):
    self._RunLoop('controller/enqueue_op/%s' % op.name, self._LoopEnqueue, op)

  def _Loop(self):
    self._summary_writer.add_graph(self._graph)
    with tf.container(self._container_id), self._GetSession() as sess:
      gsteps = self._model.global_step
      examples = self._model.total_examples

      if FLAGS.interactive:
        # Into interactive debugging mode.
        _StartShell(locals())
        return

      # This initializes local tables
      sess.run(self.initialize_tables)

      # TODO(zhifengc): Moves these options into params.
      tp = self.params.train
      save_interval_seconds = tp.save_interval_seconds
      summary_interval_steps = tp.summary_interval_steps

      next_checkpoint_seconds = 0
      next_summary_step = 1

      while True:
        now = time.time()
        next_iteration_seconds = now + 10  # 10 seconds

        # Init/restore variable if needed.
        self._RestoreIfNeeded(sess)

        global_step, total_examples = sess.run([gsteps, examples])
        step_rate, example_rate = self._RecordStepRate(global_step,
                                                       total_examples)
        if self._trial.ShouldStop() or self._ShouldStop(sess, global_step):
          tf.logging.info('Training finished.')
          self._saver.save(sess, self._save_path, gsteps)
          # Close all the queues so the enqueue threads can also finish.
          for close_op in self.close_queue_ops:
            sess.run(close_op)
          sess.close()
          return

        # Checkpoint.
        if now >= next_checkpoint_seconds:
          tf.logging.info('Save checkpoint')
          path = self._saver.save(sess, self._save_path, gsteps)
          tf.logging.info('Save checkpoint done: %s', path)
          next_checkpoint_seconds = now + save_interval_seconds

        # Summary.
        if self._summary_op is not None and global_step >= next_summary_step:
          tf.logging.info('Write summary @%s', global_step)
          summary_str = sess.run(self._summary_op)
          if isinstance(summary_str, np.ndarray) and summary_str.size == 0:
            tf.logging.info('Skipping summary: %s', summary_str)
          else:
            self._summary_writer.add_summary(summary_str, global_step)
          self._SummarizeValue(global_step, 'total_num_params',
                               self._total_num_params)
          next_summary_step = global_step + summary_interval_steps
          tf.logging.info('Write summary done: step %d', global_step)
          self._SetStatusMessage(
              'step:%6d, steps/sec: %0.2f, examples/sec: %0.2f' %
              (global_step, step_rate, example_rate))
          self._ExportMetrics(
              global_step=global_step,
              step_rate=step_rate,
              example_rate=example_rate)

        now = time.time()
        if now < next_iteration_seconds:
          time.sleep(next_iteration_seconds - now)

  def _RestoreIfNeeded(self, sess):
    uninitialized_var_names = list(sess.run(self._uninitialized))
    if not uninitialized_var_names:
      return

    tf.logging.info('Uninitialized var list: %s ', uninitialized_var_names)
    path = tf.train.latest_checkpoint(self._train_dir)
    if path:
      tf.logging.info('Load from checkpoint %s.', path)
      self._saver.restore(sess, path)
      tf.logging.info('Load checkpoint done.')
      return

    if (not any(task.params.train.init_from_checkpoint_rules
                for task in self._model.tasks) and
        not self._params.train.init_from_checkpoint_rules):
      tf.logging.info('Initialize ALL variables: %s', uninitialized_var_names)
      sess.run([self._initialize_all])
      tf.logging.info('Initialize variables done.')
      return

    # There was a race in local run. Another thread will get unblocked once
    # _initialize_all is called. OverrideVarsFromCheckpoints
    # might not happen at the right time.
    for task in self._model.tasks:
      tp = task.params.train
      if tp.init_from_checkpoint_rules:
        tf.logging.info('OverrideVarsFromCheckpoints %s',
                        tp.init_from_checkpoint_rules)
        py_utils.OverrideVarsFromCheckpoints(sess, self._vars,
                                             tp.init_from_checkpoint_rules)

    if self._params.train.init_from_checkpoint_rules:
      tp = self._params.train
      tf.logging.info('OverrideVarsFromCheckpoints %s',
                      tp.init_from_checkpoint_rules)
      py_utils.OverrideVarsFromCheckpoints(sess, self._vars,
                                           tp.init_from_checkpoint_rules)

    uninitialized_var_names = list(sess.run(self._uninitialized))
    if not uninitialized_var_names:
      return

    # uninitialized_var_names is a list of strings without ":0" suffix.
    assert all(isinstance(s, str) for s in uninitialized_var_names)

    # Need to retrieve vars, removing ":0" suffix from names.
    uninitialized_vars = [
        v for v in self._vars if v.name[:-2] in uninitialized_var_names
    ]
    tf.logging.info('Initialize variables: %s',
                    [v.name for v in uninitialized_vars])
    sess.run(tf.variables_initializer(uninitialized_vars))

  def _SummarizeValue(self, steps, tag, value):
    self._summary_writer.add_summary(
        metrics.CreateScalarSummary(tag, value), steps)

  def _RecordStepRate(self, current_steps, total_examples):
    """Computes the overall step rate and adds a summary."""
    self._time_steps.append((time.time(), current_steps, total_examples))
    # Keeps a relative long history to compute a smooth steps/second.
    while self._time_steps[-1][1] - self._time_steps[0][1] > 10000:
      del self._time_steps[0]
    (t0, s0, e0), (t1, s1, e1) = self._time_steps[0], self._time_steps[-1]
    rate = 0.0
    example_rate = 0.0
    if t1 > t0 + 1:
      elapsed_secs = t1 - t0
      rate = (s1 - s0) / elapsed_secs
      example_rate = (e1 - e0) / elapsed_secs
    tf.logging.info('Steps/second: %f, Examples/second: %f', rate, example_rate)
    self._SummarizeValue(current_steps,
                         '%s/sec' % self._model.global_step.op.name, rate)
    self._SummarizeValue(current_steps, 'examples/sec', example_rate)
    return rate, example_rate


class Trainer(base_runner.BaseRunner):
  """Trainer on non-TPU."""

  def __init__(self, *args, **kwargs):
    super(Trainer, self).__init__(*args, **kwargs)
    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._model = self.params.cls(self.params)
        self._params = self._model.params
        self._model.ConstructFPropBPropGraph()
      self.initialize_tables = get_initializer(self._model)
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      self.close_queue_ops = tf.get_collection(py_utils.CLOSE_QUEUE_OPS)
      tf.logging.info('Trainer number of enqueue ops: %d',
                      len(self.enqueue_ops))

    try:
      self._task_probs_summary_writers = []
      for task in self._model.task_schedule.tasks:
        path = os.path.join(os.path.join(self._train_dir, task))
        tf.gfile.MakeDirs(path)
        self._task_probs_summary_writers.append(self._CreateSummaryWriter(path))
    except AttributeError:
      tf.logging.info('AttributeError. Expected for single task models.')
      self._task_probs_summary_writers = []

    # Saves the graph def.
    if self.params.cluster.task > 0:
      self._summary_writer = None
    else:
      self._summary_writer = self._CreateSummaryWriter(self._train_dir)
      tf.train.write_graph(self._graph.as_graph_def(), self._train_dir,
                           'train.pbtxt')
    worker_id = self.params.cluster.task
    self._start_up_delay_steps = (((worker_id + 1) * worker_id / 2) *
                                  self.params.train.start_up_delay_steps)

  def _SummarizeValue(self, steps, tag, value, writer):
    if writer:
      writer.add_summary(metrics.CreateScalarSummary(tag, value), steps)

  def Start(self):
    self._RunLoop('trainer', self._Loop)

  def StartEnqueueOp(self, op):
    self._RunLoop('trainer/enqueue_op/%s' % op.name, self._LoopEnqueue, op)

  def _Loop(self):
    # builder = tf.profiler.ProfileOptionBuilder
    # opts = builder(builder.time_and_memory()).order_by('micros').build()
    # Create a profiling context, set constructor argument `trace_steps`,
    # `dump_steps` to empty for explicit control.
    # with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
    #                                       trace_steps=[],
    #                                       dump_steps=[]) as pctx:
    with tf.container(self._container_id), self._GetSession() as sess:
      # This initializes local tables
      import ipdb; ipdb.set_trace()
      sess.run(self.initialize_tables)
      global_step = None

      @py_utils.Retry(retry_value=(tf.errors.FailedPreconditionError,))
      def _WaitTillInit():
        """Wait until the model is ready."""
        try:
          global_step = sess.run(self._model.global_step)
        except tf.errors.FailedPreconditionError as e:
          tf.logging.info('Probably the expected race on global_step: %s', e)
          raise
        msg = 'step:%6d' % global_step
        self._SetStatusMessage(msg)
        if global_step < self._start_up_delay_steps:
          msg = 'global step (%d) has not reached start up delay steps (%d)' % (
              global_step, self._start_up_delay_steps)
          tf.logging.info('%s', msg)
          raise tf.errors.FailedPreconditionError(
              node_def=None, op=None, message=msg)
        return global_step

      global_step = _WaitTillInit()

      status_interval_steps = 100
      next_status_step = 1
      eval_metrics = None
      # 
      # from tensorflow.python import debug as tf_debug
      # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

      while True:
        if (self._trial.ShouldStopAndMaybeReport(global_step, eval_metrics) or
            self._ShouldStop(sess, global_step)):
          tf.logging.info('Training finished.')
          # Close all the queues so the enque threads can also finish.
          for close_op in self.close_queue_ops:
            sess.run(close_op)
          if self._early_stop:
            time.sleep(300)  # controller hangs if it doesn't finish first
          return

        # If a task is explicitly specified, only train that task.
        if self._model_task_name:
          model_task = self._model.GetTask(self._model_task_name)
        else:
          # Note: This is a slightly stale global_step value from the previous
          # sess.run() call.
          # For multi-task models, `self._model.task_schedule.cur_probs` will
          # be updated.
          model_task = self._model.SampleTask(global_step)
          if self._task_probs_summary_writers:
            for index, prob in enumerate(self._model.task_schedule.cur_probs):
              self._SummarizeValue(global_step, 'task_probability', prob,
                                   self._task_probs_summary_writers[index])
            try:
              for index, task in enumerate(self._model.tasks):
                self._SummarizeValue(global_step, 'task_weight',
                                     sess.run(task.vars.task_weight),
                                     self._task_probs_summary_writers[index])
            except AttributeError:
              pass
              
        # pctx.trace_next_step()
        # pctx.dump_next_step()

        _, global_step, eval_metrics = sess.run([
            model_task.train_op,
            self._model.global_step,
            model_task.eval_metrics,
        ])
        # pctx.profiler.profile_operations(options=opts)
        msg = 'step:%6d' % (global_step)
        for key, (val, _) in sorted(six.iteritems(eval_metrics)):
          msg += ' %s:%.8g' % (key, val)
          self._SummarizeValue(global_step, key, val, self._summary_writer)
        if global_step >= next_status_step:
          self._SetStatusMessage(msg)
          next_status_step = global_step + status_interval_steps
        else:
          tf.logging.info(msg)
        self._model.ModelPostUpdate(sess, global_step, eval_metrics)


class Evaler(base_runner.BaseRunner):
  """Evaler."""

  def __init__(self, eval_type, *args, **kwargs):
    super(Evaler, self).__init__(*args, **kwargs)
    self._eval_type = 'eval_' + eval_type
    self.params.is_eval = True
    self._eval_dir = os.path.join(self._logdir, self._eval_type)
    if self._model_task_name:
      self._eval_dir += '_' + str(self._model_task_name)
    tf.gfile.MakeDirs(self._eval_dir)
    self._summary_writer = self._CreateSummaryWriter(self._eval_dir)

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._model = self.params.cls(self.params)
        self._params = self._model.params
        # Always create the same graph to make sure node names are always
        # exactly the same.
        self._model.ConstructFPropGraph()
        self._model_task = self._model.GetTask(self._model_task_name)
        self._saver = self._GetSaver()
        self._summary_op = None#tf.summary.merge_all()
      self.initialize_tables = get_initializer(self._model)
      # No queues are allowed for eval models.
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      assert not self.enqueue_ops

    # Saves the graph def.
    self._WriteToLog(self.params.ToText(), self._eval_dir, 'params.txt')
    if self.params.cluster.task == 0:
      tf.train.write_graph(self._graph.as_graph_def(), self._eval_dir,
                           '%s.pbtxt' % self._eval_type)

  def Start(self):
    self._RunLoop(self._eval_type, self._Loop)

  def _Loop(self):
    """The main loop."""
    with tf.container(self._container_id), self._GetSession() as sess:
      # This initializes local tables
      sess.run(self.initialize_tables)
      path = None
      while True:
        path = self._FindNewCheckpoint(path, sess)
        if not path or self._EvalOnce(path, sess):
          break

    self.EvalLatestCheckpoint()
    tf.logging.info('Evaluation finished.')

  def EvalLatestCheckpoint(self):
    """Runs eval once on the latest checkpoint."""
    with tf.container(self._container_id), self._GetSession() as sess:
      # This initializes local tables
      sess.run(self.initialize_tables)
      path = tf.train.latest_checkpoint(self._train_dir)
      if not path:
        tf.logging.info('No checkpoint available.')
        return
      self._EvalOnce(path, sess)

  def _EvalOnce(self, path, sess):
    """Runs evaluation for a batch of samples.

    Args:
      path: checkpoint path.
      sess: the tf Session.

    Returns:
      should_stop.
    """
    if not FLAGS.evaler_in_same_address_as_controller:
      self._LoadCheckpointForEval(sess, path)

    global_step = sess.run(self._model.global_step)
    metrics_dict = {
        name: metrics.AverageMetric() for name in self._model_task.eval_metrics
    }
    num_samples_metric = metrics_dict['num_samples_in_batch']
    
    #from tensorflow.python import debug as tf_debug
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    while (num_samples_metric.total_value <
           self._model_task.params.eval.samples_per_summary):
      all_ops = [self._model_task.eval_metrics]
      all_keys = ['eval']
      if hasattr(self._model_task, 'last_state_group_op'):
        all_ops.append(self._model_task.last_state_group_op)
        all_keys.append('update')
      if self._summary_op is not None:
        all_ops.append(self._summary_op)
        all_keys.append('summary')

      ret = sess.run(all_ops) 
      #  # No summaries were collected.
      #  ans = sess.run([self._model_task.eval_metrics)
      #else:
      #  ans, summary = sess.run(
      #      [self._model_task.eval_metrics, self._summary_op])
      #  self._summary_writer.add_summary(summary, global_step)
      if 'summary' in all_keys:
        summary = ret[-1]
        self._summary_writer.add_summary(summary, global_step)
      ans = ret[0]
      for name, (value, weight) in six.iteritems(ans):
        metrics_dict[name].Update(value, weight)
      tf.logging.info('Total examples done: %d/%d',
                      num_samples_metric.total_value,
                      self._model_task.params.eval.samples_per_summary)

    # Replace average values with total values for certain metrics.
    if 'num_predictions' in metrics_dict:
      metrics_dict['num_predictions'].total_weight = 1.0
    if 'num_words' in metrics_dict:
      metrics_dict['num_words'].total_weight = 1.0

    # When we have evaluated so many samples, generate a summary.
    self._WriteSummaries(
        self._summary_writer,
        os.path.basename(self._eval_dir),
        global_step, {k: v.Summary(k) for k, v in six.iteritems(metrics_dict)},
        text_filename=os.path.join(self._eval_dir, 'score.txt'))

    is_final = global_step >= self.params.train.max_steps
    should_stop = self._trial.ReportEvalMeasure(global_step, is_final,
                                                metrics_dict)
    return should_stop or is_final


def GetDecoderDir(logdir, decoder_type, model_task_name):
  if model_task_name:
    decoder_dir = '%s_%s' % (decoder_type, model_task_name)
  else:
    decoder_dir = decoder_type
  return os.path.join(logdir, decoder_dir)


def _GetCheckpointIdForDecodeOut(checkpoint_path, global_step):
  """Retrieve the checkpoint id for the decoder out file.

  Finds the checkpoint id in the checkpoint file name and compares to global
  step. If they diverge, uses the retrieved id and prints a warning.

  Args:
   checkpoint_path: path to checkpoint file.
   global_step: int specifying the global step of the model.
  Returns:
   Checkpoint id as int.
  """
  ckpt_id_from_file = int(re.sub(r'.*ckpt-', '', checkpoint_path))
  tf.logging.info('Loaded checkpoint is at global step: %d', global_step)
  tf.logging.info('Checkpoint path: %s', checkpoint_path)
  tf.logging.info('Checkpoint id according to checkpoint path: %d',
                  ckpt_id_from_file)
  if global_step != ckpt_id_from_file:
    tf.logging.warning(
        'Checkpoint id %d != global step %d. '
        'Will use checkpoint id from checkpoint file for '
        'writing decoder output.', ckpt_id_from_file, global_step)
  return ckpt_id_from_file


class Decoder(base_runner.BaseRunner):
  """Decoder."""

  def __init__(self, decoder_type, *args, **kwargs):
    super(Decoder, self).__init__(*args, **kwargs)
    self._decoder_type = 'decoder_' + decoder_type
    self.params.is_eval = True
    self._decoder_dir = GetDecoderDir(self._logdir, self._decoder_type,
                                      self._model_task_name)
    tf.gfile.MakeDirs(self._decoder_dir)
    self._summary_writer = self._CreateSummaryWriter(self._decoder_dir)

    with self._graph.as_default(), tf.container(self._container_id):
      with self._cluster, tf.device(self._cluster.GetPlacer()):
        self._model = self.params.cls(self.params)
        self._params = self._model.params
        self._model_task = self._model.GetTask(self._model_task_name)
        # Note, different graphs are being constructed for different model
        # tasks, which may result in different node names being chosen.
        # Obviously, variable names has to be stay the same between train and
        # decode.
        self._dec_output = self._model_task.Decode()
        self._saver = self._GetSaver()
        self._summary_op = tf.summary.merge_all()
      self.initialize_tables = get_initializer(self._model)
      # No queues are allowed for decoder models.
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      assert not self.enqueue_ops

    # Saves the graph def.
    self._WriteToLog(self.params.ToText(), self._decoder_dir, 'params.txt')
    if self.params.cluster.task == 0:
      tf.train.write_graph(self._graph.as_graph_def(), self._decoder_dir,
                           '%s.pbtxt' % self._decoder_type)

  def Start(self):
    self._RunLoop(self._decoder_type, self._Loop)

  def _Loop(self):
    with tf.container(
        self._container_id), self._GetSession(inline=False) as sess:
      # This initializes local tables
      sess.run(self.initialize_tables)

      path = None
      while True:
        path = self._FindNewCheckpoint(path, sess)
        if not path or self.DecodeCheckpoint(sess, path):
          break
      tf.logging.info('Decoding finished.')

  @classmethod
  def GetDecodeOutPath(cls, decoder_dir, checkpoint_id):
    """Gets the path to decode out file."""
    out_dir = cls._GetTtlDir(decoder_dir, duration='7d')
    return os.path.join(out_dir, 'decoder_out_%09d' % checkpoint_id)

  def DecodeCheckpoint(self, sess, checkpoint_path):
    """Decodes samples_per_summary examples using params in checkpoint_path."""
    p = self._model_task.params
    samples_per_summary = p.eval.decoder_samples_per_summary
    if not samples_per_summary:
      samples_per_summary = p.eval.samples_per_summary
    self._LoadCheckpointForEval(sess, checkpoint_path)

    global_step = sess.run(self._model.global_step)
    dec_metrics = self._model_task.CreateDecoderMetrics()
    buffered_decode_out = []
    num_examples_metric = dec_metrics['num_samples_in_batch']
    start_time = time.time()
    while num_examples_metric.total_value < samples_per_summary:
      tf.logging.info('Fetching dec_output.')
      run_options = config_pb2.RunOptions(
          report_tensor_allocations_upon_oom=True)
      if self._summary_op is None:
        # No summaries were collected.
        dec_out = sess.run(self._dec_output, options=run_options)
      else:
        dec_out, summary = sess.run(
            [self._dec_output, self._summary_op], options=run_options)
        self._summary_writer.add_summary(summary, global_step)
      tf.logging.info('Done fetching.')
      decode_out = self._model_task.PostProcessDecodeOut(dec_out, dec_metrics)
      if decode_out:
        buffered_decode_out.extend(decode_out)
      tf.logging.info('Total examples done: %d/%d',
                      num_examples_metric.total_value, samples_per_summary)

    summaries = {k: v.Summary(k) for k, v in six.iteritems(dec_metrics)}
    elapsed_secs = time.time() - start_time
    example_rate = num_examples_metric.total_value / elapsed_secs
    summaries['examples/sec'] = metrics.CreateScalarSummary(
        'examples/sec', example_rate)
    self._WriteSummaries(
        self._summary_writer,
        os.path.basename(self._decoder_dir),
        global_step,
        summaries,
        text_filename=os.path.join(self._decoder_dir, 'score.txt'))
    self._ExportMetrics(
        decode_checkpoint=global_step,
        dec_metrics=dec_metrics,
        example_rate=example_rate)
    if buffered_decode_out:
      # global_step and the checkpoint id from the checkpoint file might be
      # different. For consistency of checkpoint filename and decoder_out
      # file, use the checkpoint id as derived from the checkpoint filename.
      checkpoint_id = _GetCheckpointIdForDecodeOut(checkpoint_path, global_step)
      decode_out_path = self.GetDecodeOutPath(self._decoder_dir, checkpoint_id)
      self._WriteKeyValuePairs(decode_out_path, buffered_decode_out)

    return global_step >= self.params.train.max_steps


class RunnerManager(object):
  """Helper class for managing runners."""

  # This is a hack so these classes can be overridded with internal
  # non-public implementations.
  Controller = Controller
  Trainer = Trainer
  Evaler = Evaler
  Decoder = Decoder

  @classmethod
  def LaunchTensorFlow(cls):
    """Starts TF machinary in this process."""
    tf.logging.info('Launching tensorflow.')

    target = FLAGS.tf_master
    if not target.startswith('localhost'):
      # E.g., train_client is configured w/ FLAGS.tf_master pointing to
      # another job. In that case, start a local server.
      target = tf.train.Server.create_local_server().target
    with tf.Session(target).as_default():
      value = (tf.constant(1.) + tf.constant(1.)).eval()
    assert value == 2.0, 'Something is really wrong.'
    tf.logging.info('Launched tensorflow.')

  @classmethod
  def GetParamsForDataset(cls, model_name, job_name, dataset_name):
    """Returns params for `model_name` on the dataset `dataset_name`."""
    try:
      cfg = base_runner.GetParams(model_name, dataset_name)
    except AttributeError:
      cfg = base_runner.GetParams(model_name, dataset_name.title())
    cls.UpdateClusterParamsFromFlags(cfg, job_name)
    return cfg

  @classmethod
  def MaybeConfigRunDistributed(cls):
    """If given a FLAGS.cluster_spec, create a server and update flags."""
    if not FLAGS.cluster_spec:
      return
    job_specs = FLAGS.cluster_spec.split('@')
    cluster_spec_dict = {}
    for job_spec in job_specs:
      # ps_host=worker1:1231,worker2:1234
      job_machines = job_spec.split('=')
      if len(job_machines) != 2:
        raise ValueError('Invalid job specification: %s', job_spec)
      cluster_spec_dict[job_machines[0]] = job_machines[1].split(',')
    start_server = True
    for role in sorted(cluster_spec_dict.keys()):
      if role.startswith('decoder_'):
        assert len(job_specs) == 1, 'Decoder jobs must run on their own'
        assert ',' not in job_specs[0], 'Only single machine supported'
        FLAGS.decoder_job = '/job:localhost'
        FLAGS.decoder_replicas = 1
        start_server = False
      elif role.startswith('evaler_'):
        assert len(job_specs) == 1, 'Evaler jobs must run on their own'
        assert ',' not in job_specs[0], 'Only single machine supported'
        FLAGS.evaler_job = '/job:localhost'
        FLAGS.evaler_replicas = 1
        start_server = False
      elif role == 'worker':
        assert FLAGS.mode == 'sync', (
            '\'worker\' jobs should only be in sync mode')
        FLAGS.worker_job = '/job:worker'
        FLAGS.worker_replicas = len(cluster_spec_dict['worker'])
        FLAGS.ps_job = '/job:worker'
        FLAGS.ps_replicas = FLAGS.worker_replicas
    if start_server:
      cluster = tf.train.ClusterSpec(cluster_spec_dict)
      server = tf.train.Server(
          cluster, job_name=FLAGS.job, task_index=FLAGS.task)
      FLAGS.tf_master = server.target
      # TODO(drpng): properly join all processes.
      cls._server = server

  @classmethod
  def UpdateClusterParamsFromFlags(cls, cfg, job_name):
    """Update Params with a training cluster configuration from flags."""
    cfg.cluster.mode = FLAGS.mode
    cfg.cluster.job = job_name
    cfg.cluster.task = FLAGS.task

    cfg.cluster.controller.name = FLAGS.controller_job
    cfg.cluster.controller.gpus_per_replica = FLAGS.controller_gpus

    cfg.cluster.worker.name = FLAGS.worker_job
    cfg.cluster.worker.replicas = FLAGS.worker_replicas
    cfg.cluster.worker.gpus_per_replica = FLAGS.worker_gpus
    cfg.cluster.worker.devices_per_split = FLAGS.worker_split_size

    cfg.cluster.ps.name = FLAGS.ps_job
    cfg.cluster.ps.replicas = FLAGS.ps_replicas
    cfg.cluster.ps.gpus_per_replica = FLAGS.ps_gpus

    cfg.cluster.input.name = FLAGS.input_job
    cfg.cluster.input.replicas = FLAGS.input_replicas

    cfg.cluster.evaler.name = FLAGS.evaler_job
    cfg.cluster.evaler.replicas = FLAGS.evaler_replicas
    cfg.cluster.evaler.gpus_per_replica = FLAGS.evaler_gpus

    cfg.cluster.decoder.name = FLAGS.decoder_job
    cfg.cluster.decoder.replicas = FLAGS.decoder_replicas
    cfg.cluster.decoder.gpus_per_replica = FLAGS.decoder_gpus

  @classmethod
  def _CreateRunner(cls, job, model_name, model_task_name, logdir, tf_master,
                    trial):
    """Create a runner."""
    evaler_job_name_prefix = 'evaler_'
    decoder_job_name_prefix = 'decoder_'

    tf.logging.info('Job %s start', job)
    common_args = (model_task_name, logdir, tf_master, trial)
    if job == 'controller':
      cfg = cls.GetParamsForDataset(model_name, 'controller', 'Train')
      return cls.Controller(cfg, *common_args)
    elif job == 'trainer':
      cfg = cls.GetParamsForDataset(model_name, 'trainer', 'Train')
      return cls.Trainer(cfg, *common_args)
    elif job == 'trainer_client':
      cfg = cls.GetParamsForDataset(model_name, 'trainer_client', 'Train')
      if py_utils.use_tpu():
        raise ValueError('TPU training is not supported.')
      else:
        return cls.Trainer(cfg, *common_args)
    elif job.startswith(evaler_job_name_prefix):
      dataset_name = job[len(evaler_job_name_prefix):]
      cfg = cls.GetParamsForDataset(model_name, 'evaler', dataset_name)
      return cls.Evaler(dataset_name.lower(), cfg, *common_args)
    elif job.startswith(decoder_job_name_prefix):
      dataset_name = job[len(decoder_job_name_prefix):]
      cfg = cls.GetParamsForDataset(model_name, 'decoder', dataset_name)
      return cls.Decoder(dataset_name.lower(), cfg, *common_args)
    # TODO(drpng): make a tf_server.
    elif job == 'ps' or 'worker' or 'input':
      if cls._server:
        cls._server.join()
    else:
      raise ValueError('job %s is not supported' % job)

  @classmethod
  def CreateRunners(cls, jobs, model_name, logdir,
                    trial=base_trial.NoOpTrial()):
    """Creates a list of runners based on FLAGS.mode.

    Args:
      jobs: a list of runner jobs.
      model_name: name of a registered ModelParams class.
      logdir: the directory used for logging, usually on CNS.
      trial: optional Trial object, used for reporting measures and early
          stopping.

    Returns:
      A list of BaseRunners, one per job in 'jobs'.
    """

    runners = []
    for j in jobs:
      tf_master = FLAGS.tf_master
      # Ensure that decoder or evaler threads do not clobber variables being
      # updated by trainer by forcing them to use independent sessions.
      if ('trainer' in jobs and
          (j.startswith('decoder') or j.startswith('evaler'))):
        tf_master = ''

      runner = cls._CreateRunner(j, model_name, FLAGS.model_task_name, logdir,
                                 tf_master, trial)
      runners.append(runner)
    return runners

  @classmethod
  def StartRunners(cls, runners):
    """Runs 'runners' in parallel threads. Returns when all of them finish.

    Args:
      runners: a list of BaseRunners.

    Returns:
      None.
    """
    threads = []
    tf.logging.info('Starting runners')
    for runner in runners:
      t = threading.Thread(target=runner.Start)
      t.daemon = True
      t.start()
      threads.append(t)
      tf.logging.info('Total num runner.enqueue_ops: %d',
                      len(runner.enqueue_ops))
      for enqueue_op in runner.enqueue_ops:

        def StartEnqueue(runner, op):
          tf.logging.info('Starting enqueue op %s', op.name)
          return lambda: runner.StartEnqueueOp(op)

        tq = threading.Thread(target=StartEnqueue(runner, enqueue_op))
        tq.start()
        threads.append(tq)
    tf.logging.info('Waiting for runners to finish...')
    for t in threads:
      while True:
        t.join(1)
        if not t.isAlive():
          break
    tf.logging.info('All runners done.')

  @classmethod
  def RunTrial(cls, job, model_name, logdir, trial):
    """A wrapper function for running a trial."""
    if job == 'all':
      # For async mode: Run controller, trainer, evaler jobs in one process,
      # multiple threads.
      cls.StartRunners(
          cls.CreateRunners(['controller', 'trainer'], model_name, logdir,
                            trial))
      evaler = cls._CreateRunner('evaler_dev', model_name,
                                 FLAGS.model_task_name, logdir, FLAGS.tf_master,
                                 trial)
      evaler.EvalLatestCheckpoint()
    elif job == 'all_sync':
      # For sync mode: Run controller, trainer_client, evaler jobs in one
      # process, multiple threads.
      cls.StartRunners(
          cls.CreateRunners(['controller', 'trainer_client'], model_name,
                            logdir, trial))
      evaler = cls._CreateRunner('evaler_dev', model_name,
                                 FLAGS.model_task_name, logdir, FLAGS.tf_master,
                                 trial)
      evaler.EvalLatestCheckpoint()
    else:
      # Run each job in separate process/task
      # TODO(rpang): add support for running evaler_test and decoder.
      cls.StartRunners(cls.CreateRunners([job], model_name, logdir, trial))

  @classmethod
  def MaybeConfigRunLocally(cls):
    """Update flags if configured to run locally."""
    if not FLAGS.run_locally:
      # Do nothing
      return
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    FLAGS.tf_master = tf.train.Server.create_local_server(config=config).target

    if not FLAGS.mode:
      FLAGS.mode = 'sync'

    if not FLAGS.job:
      FLAGS.job = 'controller,trainer_client'

    FLAGS.task = 0

    FLAGS.controller_job = '/job:local'

    FLAGS.worker_job = '/job:local'
    FLAGS.worker_replicas = 1
    if FLAGS.run_locally == 'gpu':
      if not FLAGS.worker_gpus:
        FLAGS.worker_gpus = 1
    else:
      FLAGS.worker_gpus = 0

    if not FLAGS.worker_split_size:
      FLAGS.worker_split_size = 1

    FLAGS.ps_job = '/job:local'
    FLAGS.ps_replicas = 1
    FLAGS.ps_gpus = 0

    FLAGS.input_job = '/job:local'
    FLAGS.input_replicas = 0

    FLAGS.evaler_job = '/job:local'
    FLAGS.evaler_replicas = 1
    # if FLAGS.run_locally == 'gpu':
    #   FLAGS.evaler_gpus = 1
    # else:
    #   FLAGS.evaler_gpus = 0

    FLAGS.decoder_job = '/job:local'
    FLAGS.decoder_replicas = 1
    # if FLAGS.run_locally == 'gpu':
    #   FLAGS.decoder_gpus = 1
    # else:
    #   FLAGS.decoder_gpus = 0


def InspectModel():
  """Prints out model analysis for the model."""
  p = RunnerManager.GetParamsForDataset(FLAGS.model, 'controller', 'Train')
  p.cluster.mode = 'sync'
  c = cluster_factory.Cluster(p.cluster)
  with tf.Graph().as_default(), c, tf.device(c.GetPlacer()):
    analysis, _ = _ModelAnalysis(p.cls(p))
  print(analysis)


def InspectDatasets():
  """Prints out datasets configured for the model."""
  cls = base_runner.GetClass(FLAGS.model)
  datasets = []
  for name, _ in inspect.getmembers(cls, inspect.ismethod):
    if name not in ['GetDatasetParams', 'Model', 'Task'
                   ] and not name.startswith('_'):
      datasets += [name]
  print(','.join([_.lower() for _ in datasets]))


def InspectDecoder():
  """Prints out datasets configured for the decoder."""
  cls = base_runner.GetClass(FLAGS.model)

  has_decoder = False
  if issubclass(cls, base_model_params.SingleTaskModelParams):
    has_decoder = cls.Task(
    ).cls.CreateDecoderMetrics != base_model.BaseTask.CreateDecoderMetrics
  else:
    for _, task_param in cls.Model().task_params.IterParams():
      has_decoder |= (
          task_param.cls.CreateDecoderMetrics !=
          base_model.BaseTask.CreateDecoderMetrics)
  if has_decoder:
    # We assume that the proper decoder is implemented.
    InspectDatasets()
  else:
    print('')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  RunnerManager.MaybeConfigRunLocally()
  RunnerManager.MaybeConfigRunDistributed()

  if not (FLAGS.run_locally or FLAGS.mode == 'inspect_evaler' or
          FLAGS.mode == 'inspect_decoder'):
    RunnerManager.LaunchTensorFlow()

  # pylint: disable=g-import-not-at-top
  # pylint: disable=unused-variable
  from lingvo import model_imports

  if FLAGS.mode == 'shell':
    _StartShell(locals())
    return

  assert base_runner.GetClass(
      FLAGS.model), ('Model %s is not found.' % FLAGS.model)

  if FLAGS.mode == 'inspect_model':
    InspectModel()
    return

  if FLAGS.mode == 'inspect_evaler':
    InspectDatasets()
    return

  if FLAGS.mode == 'inspect_decoder':
    InspectDecoder()
    return

  assert FLAGS.mode in ['sync', 'async']

  RunnerManager.StartRunners(
      RunnerManager.CreateRunners(
          FLAGS.job.split(','), FLAGS.model, FLAGS.logdir))


if __name__ == '__main__':
  tf.app.run(main)
        
