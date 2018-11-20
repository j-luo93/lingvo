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
"""Tests for inference_graph_exporter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo import model_registry
from lingvo.core import base_input_generator
from lingvo.core import base_model
from lingvo.core import base_model_params
from lingvo.core import inference_graph_exporter
from lingvo.core import inference_graph_pb2
from lingvo.core import py_utils


class DummyLegacyModel(base_model.BaseTask):

  @classmethod
  def Params(cls):
    p = super(DummyLegacyModel, cls).Params()
    return p

  def Inference(self):
    with tf.name_scope('inference'):
      feed1 = tf.placeholder(name='feed1_node', dtype=tf.float32, shape=[1])
      fetch1 = tf.identity(feed1, name='fetch1_node')
      return {
          'default': (
              py_utils.NestedMap({
                  'fetch1': fetch1,
                  'fetch_op': fetch1.op,  # Tests that ops are supported.
              }),
              py_utils.NestedMap({
                  'feed1': feed1,
              })),
          'unused': (py_utils.NestedMap({}), py_utils.NestedMap({})),
      }


@model_registry.RegisterSingleTaskModel
class DummyLegacyModelParams(base_model_params.SingleTaskModelParams):

  @classmethod
  def Test(cls):
    p = base_input_generator.BaseSequenceInputGenerator.Params()
    p.name = 'input'
    return p

  @classmethod
  def Task(cls):
    p = DummyLegacyModel.Params()
    p.name = 'testing'
    return p


class DummyModel(base_model.BaseTask):

  @classmethod
  def Params(cls):
    p = super(DummyModel, cls).Params()
    return p

  def Inference(self):
    with tf.name_scope('inference'):
      feed1 = tf.placeholder(name='feed1_node', dtype=tf.float32, shape=[1])
      fetch1 = tf.identity(feed1, name='fetch1_node')
      inference_graph = inference_graph_pb2.InferenceGraph()
      subgraph = inference_graph.subgraphs['default']
      subgraph.feeds['feed1'] = feed1.name
      subgraph.fetches['fetch1'] = fetch1.name
      # Tests that ops are supported.
      subgraph.fetches['fetch_op'] = fetch1.op.name
      return inference_graph


@model_registry.RegisterSingleTaskModel
class DummyModelParams(base_model_params.SingleTaskModelParams):

  @classmethod
  def Test(cls):
    p = base_input_generator.BaseSequenceInputGenerator.Params()
    p.name = 'input'
    return p

  @classmethod
  def Task(cls):
    p = DummyModel.Params()
    p.name = 'testing'
    return p


class InferenceGraphExporterTest(tf.test.TestCase):

  def testExportModelParamsWithSubgraphDict(self):
    params = model_registry.GetParams('test.DummyLegacyModelParams', 'Test')
    inference_graph = inference_graph_exporter.InferenceGraphExporter.Export(
        params, subgraph_filter=['default'])

    # Should populate subgraphs.
    self.assertIn('default', inference_graph.subgraphs)
    self.assertNotIn('unused', inference_graph.subgraphs)
    subgraph = inference_graph.subgraphs['default']
    self.assertIn('feed1', subgraph.feeds)
    self.assertIn('fetch1', subgraph.fetches)

    self.assertEqual(subgraph.feeds['feed1'], 'inference/feed1_node:0')
    self.assertEqual(subgraph.fetches['fetch1'], 'inference/fetch1_node:0')
    self.assertEqual(subgraph.fetches['fetch_op'], 'inference/fetch1_node')

  def testExportModelParamsWithInferenceGraph(self):
    params = model_registry.GetParams('test.DummyModelParams', 'Test')
    inference_graph = inference_graph_exporter.InferenceGraphExporter.Export(
        params)

    # Should populate subgraphs.
    self.assertIn('default', inference_graph.subgraphs)
    subgraph = inference_graph.subgraphs['default']
    self.assertIn('feed1', subgraph.feeds)
    self.assertIn('fetch1', subgraph.fetches)

    self.assertEqual(subgraph.feeds['feed1'], 'inference/feed1_node:0')
    self.assertEqual(subgraph.fetches['fetch1'], 'inference/fetch1_node:0')
    self.assertEqual(subgraph.fetches['fetch_op'], 'inference/fetch1_node')


class NoConstGuaranteeScopeTest(tf.test.TestCase):

  def testNoConsting(self):
    with inference_graph_exporter.ConstGuaranteeScope():
      wp = py_utils.WeightParams(
          shape=[1],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=tf.float32,
          collections=['v'])
      v = py_utils.CreateVariable('v', wp)[1]
      self.assertEqual(tf.Tensor, type(v))
      with inference_graph_exporter.NoConstGuaranteeScope():
        v = py_utils.CreateVariable('v', wp, reuse=True)[1]
        self.assertIsInstance(v, tf.Variable)


class LinearModel(base_model.BaseTask):
  """A basic linear model."""

  @classmethod
  def Params(cls):
    p = super(LinearModel, cls).Params()
    p.name = 'linear_model'
    return p

  def __init__(self, params):
    super(LinearModel, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      w = py_utils.WeightParams(
          shape=[3],
          init=py_utils.WeightInit.Gaussian(scale=1.0, seed=123456),
          dtype=p.dtype)
      b = py_utils.WeightParams(
          shape=[],
          init=py_utils.WeightInit.Gaussian(scale=1.0, seed=234567),
          dtype=p.dtype)
      self._w, _ = py_utils.CreateVariable('w', w)
      self._b, _ = py_utils.CreateVariable('b', b)

  def Inference(self):
    """Computes y = w^T x + b. Returns y and x, as outputs and inputs."""
    with tf.variable_scope('inference'):
      x = tf.placeholder(dtype=tf.float32, name='input')
      y = tf.reduce_sum(self._w * x) + self._b
      return {'default': ({'output': y}, {'input': x})}


class LinearModelTpu(LinearModel):
  """A basic linear model that runs inference on the TPU."""

  def Inference(self):
    """Computes y = w^T x + b. Returns y and x, as outputs and inputs."""
    with tf.variable_scope('inference'):
      x = tf.placeholder(dtype=tf.bfloat16, name='input')

      def InferenceFn(x):
        return tf.reduce_sum(self._w * x) + self._b

      y = tf.contrib.tpu.rewrite(InferenceFn, [x])
      return {'tpu': ({'output': y[0]}, {'input': x})}


@model_registry.RegisterSingleTaskModel
class LinearModelParams(base_model_params.SingleTaskModelParams):

  @classmethod
  def Test(cls):
    p = base_input_generator.BaseSequenceInputGenerator.Params()
    p.name = 'input'
    return p

  @classmethod
  def Task(cls):
    p = LinearModel.Params()
    p.name = 'testing'
    return p


@model_registry.RegisterSingleTaskModel
class LinearModelTpuParams(base_model_params.SingleTaskModelParams):

  @classmethod
  def Test(cls):
    p = base_input_generator.BaseSequenceInputGenerator.Params()
    p.name = 'input'
    return p

  @classmethod
  def Task(cls):
    p = LinearModelTpu.Params()
    p.name = 'testing'
    return p


class InferenceGraphExporterLinearModelTest(tf.test.TestCase):

  def testExport(self):
    """Test basic export."""
    params = model_registry.GetParams('test.LinearModelParams', 'Test')
    inference_graph = inference_graph_exporter.InferenceGraphExporter.Export(
        params, subgraph_filter=['default'])
    self.assertIn('default', inference_graph.subgraphs)

  def testTpuBfloat16OverrideExport(self):
    """Test that we can export with tf.bfloat16 dtype."""
    params = model_registry.GetParams('test.LinearModelTpuParams', 'Test')
    inference_graph = inference_graph_exporter.InferenceGraphExporter.Export(
        params,
        subgraph_filter=['tpu'],
        device_options=inference_graph_exporter.InferenceDeviceOptions(
            device='tpu',
            retain_device_placement=True,
            var_options='ON_DEVICE',
            gen_init_op=True,
            dtype_override=tf.bfloat16))
    self.assertIn('tpu', inference_graph.subgraphs)


if __name__ == '__main__':
  tf.test.main()
