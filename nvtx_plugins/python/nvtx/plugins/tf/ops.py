# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wrapt
import tensorflow as tf

from tensorflow.python.framework import ops

from nvtx.plugins.tf.ext_utils import load_library
from nvtx.plugins.tf.ext_utils import get_ext_suffix
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


__all__ = ['nvtx_tf_ops', 'start', 'end', 'trace']


nvtx_tf_ops = load_library('lib/nvtx_ops' + get_ext_suffix())

def _maybe_convert_list_to_tensor(inputs):

    inputs_were_processed = False

    if isinstance(inputs, (list, tuple)) and \
        all([isinstance(x, tf.Tensor) for x in inputs]):
        inputs = tf.stack(inputs, axis=0, name="nvtx_trace_inputs")
        inputs_were_processed = True
    elif isinstance(inputs, (list, tuple)) and \
        all([isinstance(x, tf.RaggedTensor) for x in inputs]):
        inputs = tf.stack(inputs, axis=0, name="nvtx_trace_inputs")
        inputs_were_processed = True

    assert isinstance(inputs, (tf.Tensor, tf.RaggedTensor))

    return inputs, inputs_were_processed

def _unstack(value, axis=0):
  if isinstance(value, tf.RaggedTensor):
    leading_slices = (Ellipsis,) * axis
    return [value.__getitem__(leading_slices+(i,)) for i in range(value.shape[axis])]
  elif isinstance(value, tf.Tensor):
    return tf.unstack(value, axis=axis)
  return value

@ops.RegisterGradient('NvtxStart')
def _nvtx_start_grad(*grad_pack):
    #print(grad_pack)
    # grad_message and grad_domain_name are not used
    op = grad_pack[0]
    grad = []
    idx = []
    for i in range(1,len(grad_pack)-2):
      if grad_pack[i] is not None:
        grad.append(grad_pack[i])
        idx.append(True)
      else:
        idx.append(False)
    #grad = [grad_pack[i] if grad_pack[i] is not None for i in range(1,len(grad_pack)-2)]
    #= grad_pack[1:-2]
    marker_id = grad_pack[-2]
    domain_handle = grad_pack[-1]
    #print('Start grad marker id = ', marker_id)
    #print(grad)
    if not isinstance(marker_id, tf.Tensor) and marker_id is None:
        raise RuntimeError('Error in nvtx range %s. '
                           'Make sure all nvtx ranges are closed' % op.name)

    if not grad:
      grad = [tf.constant(0, dtype=tf.int64)]
    grad, null_grad = nvtx_tf_ops.nvtx_end(inputs=grad,
        marker_id=marker_id, domain_handle=domain_handle,
        grad_message=op.inputs[-2], grad_domain_name=op.inputs[-1])
    final_grad = []
    curr = 0
    for i in range(len(grad_pack)-3):
      if idx[i]:
        final_grad.append(grad[curr])
        curr += 1
      else:
        final_grad.append(None)
    #print(final_grad)
    #grad.append(null_grad)
    #grad.append(None)
    #grad.append(None)
    #[null_grad, None, None]
    #return grad
    return [*final_grad, null_grad, None, None]

@ops.RegisterGradient('NvtxEnd')
def _nvtx_end_grad(op, grad, null_grad):
    #print(grad)
    added_list = False
    if not isinstance(grad, (list, tuple)):
      grad = [grad]
      added_list = True
    grad, marker_id, domain_handle = nvtx_tf_ops.nvtx_start(
        inputs=grad, null_input=1.,
        message=op.inputs[3], domain_name=op.inputs[4])
    #print('End grad marker id = ', marker_id)
    if added_list:
      grad = grad[0]
    return [grad, marker_id, domain_handle, None, None]

def start(inputs, message, domain_name=None,
          grad_message=None, grad_domain_name=None,
          trainable=False, enabled=True, name=None):
    """An identity operation with a side effect of opening an NVTX marker.

    Note:
        The :func:`ops.start <start>` and :func:`ops.end <end>` operations
        must be used in pairs.

    Example:
        .. highlight:: python
        .. code-block:: python

            x, nvtx_context = nvtx.plugins.tf.ops.start(x, message='Dense 1-3',
                domain_name='Forward', grad_domain_name='Gradient')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_1')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_2')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_3')
            x = nvtx.plugins.tf.ops.end(x, nvtx_context)

    Arguments:
        inputs: A ``Tensor`` object that is passed to ``output``.
        message: A ``string`` message to be associated with this marker.
        domain_name: An optional ``string`` domain name to be associated with
            this marker. If not provided the default NVTX domain will be used.
        grad_message: An optional ``string`` message to be associated with
            the op gradient. If not provided ``message`` will be used.
        grad_domain_name: An optional ``string`` domain name to be associated
            with this marker gradient. If not provided ``domain_name`` will
            be used.
        trainable: ``bool``, if ``True`` will make this op
            trainable. Used when this is the first operation in the graph to
            prevent an open ended marker during gradient calculation.
        enabled: ``bool``, if ``False`` the nvtx marker will be disabled.
        name: An optional `string` name for the operation.

    Returns:
        ``tuple``:
        - output: The inputs ``Tensor``.
        - nvtx_context: ``list``, NVTX context associated with this op and passed to :func:`ops.end <end>`. ``None``  if ``enabled=False``.

    """
    if not enabled:
        return inputs, None

    domain_name = domain_name or ''
    grad_message = grad_message or message
    grad_domain_name = grad_domain_name or domain_name or ''

    null_input = 1.

    if trainable:
        #null_input = tf.Variable(1., shape=(), dtype=tf.float32, name='nvtx_null_input', experimental_enable_variable_lifting=False, trainable=True)
        with tf.compat.v1.variable_scope("nvtx", reuse=tf.compat.v1.AUTO_REUSE):
            #null_input = tf.Variable(1. shape=(), dtype=tf.float32, name='null_input')
            null_input = tf.compat.v1.get_variable('null_input', shape=(),
                                                   dtype=tf.float32,
                                                   initializer=tf.zeros_initializer,
                                                   trainable=True)

    added_list = False
    if not isinstance(inputs, (list,tuple)):
      added_list = True
      inputs = [inputs]

    if isinstance(inputs[0], tf.Tensor):
      #inputs, should_unstack = _maybe_convert_list_to_tensor(inputs)
      outputs, marker_id, domain_handle = nvtx_tf_ops.nvtx_start(inputs=inputs, null_input=null_input,
          message=message, domain_name=domain_name, name=name)
      #if should_unstack:
      #  outputs = tf.unstack(inputs, axis=0)
    elif isinstance(inputs[0], tf.RaggedTensor):
      outputs = []
      tensors = []
      for elem in inputs:
        tensors.append(elem.values)
        tensors.append(elem.row_splits)
      print('Tensors len', len(tensors))
      tensors, marker_id, domain_handle = nvtx_tf_ops.nvtx_start(inputs=tensors, null_input=null_input,
          message=message, domain_name=domain_name, name=name)
      print('Fprop marker id generated: ', marker_id)
      ''' 
      row_splits, values, marker_id, domain_handle = nvtx_tf_ops.nvtx_start_ragged(
        inputs_splits=row_splits, inputs_values=values, null_input=null_input,
          message=message, domain_name=domain_name, name=name)
      outputs.append(tf.RaggedTensor.from_row_splits(values, row_splits, validate=False))
      for i in range(1, len(inputs)):
        outputs.append(inputs[i])
      '''
      for i in range(0,len(tensors),2):
        outputs.append(tf.RaggedTensor.from_row_splits(tensors[i], tensors[i+1], validate=False))
      print('Tensors len', len(tensors))
      print('Inputs len', len(inputs))
      print('Outputs len', len(outputs))

    #if ragged:
    #  inputs = tf.RaggedTensor.from_tensor(inputs)

    #if should_unstack:
    #    inputs = _unstack(inputs)
        #inputs = tf.unstack(inputs, axis=0)
    if added_list:
      outputs = outputs[0]

    return outputs, (marker_id, domain_handle, grad_message, grad_domain_name)


def end(inputs, nvtx_context, name=None):
    """An identity operation with a side effect of closing an NVTX marker.

    Note:
        The :func:`ops.start <start>` and :func:`ops.end <end>` operations
        must be used in pairs.

    Example:
        .. highlight:: python
        .. code-block:: python

            x, nvtx_context = nvtx.plugins.tf.ops.start(x, message='Dense 1-3',
                domain_name='Forward', grad_domain_name='Gradient')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_1')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_2')
            x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='dense_3')
            x = nvtx.plugins.tf.ops.end(x, nvtx_context)

    Arguments:
        inputs: A ``Tensor`` object that will be passed to ``output``.
        nvtx_context: ``list``, NVTX context received from
            :func:`ops.start <start>` If `None` the marker will be disabled.
        name: An optional ``string`` name for the operation.

    Returns:
        The inputs ``Tensor``.

    """
    if nvtx_context is None:
        return inputs

    marker_id, domain_handle, grad_message, grad_domain_name = nvtx_context
    added_list = False
    if not isinstance(inputs, (list,tuple)):
      added_list = True
      inputs = [inputs]

    if isinstance(inputs, tf.Tensor) or (isinstance(inputs, list) and isinstance(inputs[0], tf.Tensor)):
      #inputs, should_unstack = _maybe_convert_list_to_tensor(inputs)
      outputs, null_output = nvtx_tf_ops.nvtx_end(inputs=inputs,
          marker_id=marker_id, domain_handle=domain_handle,
          grad_message=grad_message, grad_domain_name=grad_domain_name, name=name
      )
      #if should_unstack:
      #  outputs = tf.unstack(outputs, axis=0)
    elif isinstance(inputs[0], tf.RaggedTensor):
      outputs = []
      tensors = []
      for elem in inputs:
        tensors.append(elem.values)
        tensors.append(elem.row_splits)
      print('Tensors len', len(tensors))
      output, null_output = nvtx_tf_ops.nvtx_end(inputs=tensors,
          marker_id=marker_id, domain_handle=domain_handle,
          grad_message=grad_message, grad_domain_name=grad_domain_name, name=name
      )
      for i in range(0,len(tensors),2):
        outputs.append(tf.RaggedTensor.from_row_splits(tensors[i], tensors[i+1], validate=False))
      print('Tensors len', len(tensors))
      print('Inputs len', len(inputs))
      print('Outputs len', len(outputs))
      '''
      outputs = []
      values = inputs[0].values
      row_splits = inputs[0].row_splits
      print(values.shape)
      print(row_splits.shape)
      row_splits, values, null_output = nvtx_tf_ops.nvtx_end_ragged(inputs_splits=row_splits, inputs_values=values,
      #output, null_output = nvtx_tf_ops.nvtx_end(inputs=tf.constant(0, dtype=tf.int32),
          marker_id=marker_id, domain_handle=domain_handle,
          grad_message=grad_message, grad_domain_name=grad_domain_name, name=name
      )
      print(values.shape)
      print(row_splits.shape)
      outputs.append(tf.RaggedTensor.from_row_splits(values, row_splits, validate=False))
      for i in range(1, len(inputs)):
        outputs.append(inputs[i])
      #inputs[0] = tf.RaggedTensor.from_row_splits(values, row_splits, validate=False)
      '''

    if added_list:
      outputs = outputs[0]
    return outputs


def trace(message, domain_name=None,
          grad_message=None, grad_domain_name=None,
          trainable=False, enabled=True, name=None):
    """An identity function decorator with a side effect of adding NVTX marker.

    Note:
        The decorator expects the wrapped function to take the input ``Tensor``
        as the first argument or to be named ``inputs``, and to return a single
        ``Tensor``.

    Arguments:
        message: A ``string`` message to be associated with this marker.
        domain_name: An optional ``string`` domain name to be associated with
            this marker. If not provided the default NVTX domain will be used.
        grad_message: An optional ``string`` message to be associated with
            the op gradient. If not provided `message` will be used.
        grad_domain_name: An optional ``string`` domain name to be associated
            with this marker gradient. If not provided ``domain_name`` will
            be used.
        trainable: ``bool``, if ``True`` will make this op
            trainable. Used when this is the first operation in the graph to
            prevent an open ended marker during gradient calculation.
        enabled: ``bool``, if ``False`` the nvtx marker will be disabled.
        name: An optional ``string`` name for the operation.

    """
    @wrapt.decorator
    def func_wrapper(wrapped, instance, args, kwargs):
        try:
            inputs = kwargs["inputs"] if "inputs" in kwargs else args[0]
        except:
            raise ValueError("The input tensor must be the first argument"
                             " or named `inputs`")

        inputs, should_unstack = _maybe_convert_list_to_tensor(inputs)

        start_name = '{}_start'.format(name) if name else None
        end_name = '{}_end'.format(name) if name else None

        inputs, nvtx_context = start(inputs=inputs,
            message=message, domain_name=domain_name,
            grad_message=grad_message, grad_domain_name=grad_domain_name,
            enabled=enabled, trainable=trainable, name=start_name
        )

        if should_unstack:
            inputs = tf.unstack(inputs, axis=0)

        if "inputs" in kwargs:
            kwargs["inputs"] = inputs
        else:
            args = [inputs] + list(args[1:])
            
        output = wrapped(*args, **kwargs)
        output = end(inputs=output, nvtx_context=nvtx_context, name=end_name)

        return output

    return func_wrapper
