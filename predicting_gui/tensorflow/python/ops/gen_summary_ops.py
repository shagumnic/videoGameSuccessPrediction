"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: summary_ops.cc
"""

import collections

from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export


def close_summary_writer(writer, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "CloseSummaryWriter", name,
        tld.op_callbacks, writer)
      return _result
    except _core._FallbackException:
      try:
        return close_summary_writer_eager_fallback(
            writer, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CloseSummaryWriter", writer=writer, name=name)
  return _op
CloseSummaryWriter = tf_export("raw_ops.CloseSummaryWriter")(_ops.to_raw_op(close_summary_writer))


def close_summary_writer_eager_fallback(writer, name, ctx):
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  _inputs_flat = [writer]
  _attrs = None
  _result = _execute.execute(b"CloseSummaryWriter", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def create_summary_db_writer(writer, db_uri, experiment_name, run_name, user_name, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    db_uri: A `Tensor` of type `string`.
    experiment_name: A `Tensor` of type `string`.
    run_name: A `Tensor` of type `string`.
    user_name: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "CreateSummaryDbWriter", name,
        tld.op_callbacks, writer, db_uri, experiment_name, run_name,
        user_name)
      return _result
    except _core._FallbackException:
      try:
        return create_summary_db_writer_eager_fallback(
            writer, db_uri, experiment_name, run_name, user_name, name=name,
            ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CreateSummaryDbWriter", writer=writer, db_uri=db_uri,
                                 experiment_name=experiment_name,
                                 run_name=run_name, user_name=user_name,
                                 name=name)
  return _op
CreateSummaryDbWriter = tf_export("raw_ops.CreateSummaryDbWriter")(_ops.to_raw_op(create_summary_db_writer))


def create_summary_db_writer_eager_fallback(writer, db_uri, experiment_name, run_name, user_name, name, ctx):
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  db_uri = _ops.convert_to_tensor(db_uri, _dtypes.string)
  experiment_name = _ops.convert_to_tensor(experiment_name, _dtypes.string)
  run_name = _ops.convert_to_tensor(run_name, _dtypes.string)
  user_name = _ops.convert_to_tensor(user_name, _dtypes.string)
  _inputs_flat = [writer, db_uri, experiment_name, run_name, user_name]
  _attrs = None
  _result = _execute.execute(b"CreateSummaryDbWriter", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def create_summary_file_writer(writer, logdir, max_queue, flush_millis, filename_suffix, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    logdir: A `Tensor` of type `string`.
    max_queue: A `Tensor` of type `int32`.
    flush_millis: A `Tensor` of type `int32`.
    filename_suffix: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "CreateSummaryFileWriter",
        name, tld.op_callbacks, writer, logdir, max_queue, flush_millis,
        filename_suffix)
      return _result
    except _core._FallbackException:
      try:
        return create_summary_file_writer_eager_fallback(
            writer, logdir, max_queue, flush_millis, filename_suffix,
            name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "CreateSummaryFileWriter", writer=writer, logdir=logdir,
                                   max_queue=max_queue,
                                   flush_millis=flush_millis,
                                   filename_suffix=filename_suffix, name=name)
  return _op
CreateSummaryFileWriter = tf_export("raw_ops.CreateSummaryFileWriter")(_ops.to_raw_op(create_summary_file_writer))


def create_summary_file_writer_eager_fallback(writer, logdir, max_queue, flush_millis, filename_suffix, name, ctx):
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  logdir = _ops.convert_to_tensor(logdir, _dtypes.string)
  max_queue = _ops.convert_to_tensor(max_queue, _dtypes.int32)
  flush_millis = _ops.convert_to_tensor(flush_millis, _dtypes.int32)
  filename_suffix = _ops.convert_to_tensor(filename_suffix, _dtypes.string)
  _inputs_flat = [writer, logdir, max_queue, flush_millis, filename_suffix]
  _attrs = None
  _result = _execute.execute(b"CreateSummaryFileWriter", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=ctx,
                             name=name)
  _result = None
  return _result


def flush_summary_writer(writer, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "FlushSummaryWriter", name,
        tld.op_callbacks, writer)
      return _result
    except _core._FallbackException:
      try:
        return flush_summary_writer_eager_fallback(
            writer, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "FlushSummaryWriter", writer=writer, name=name)
  return _op
FlushSummaryWriter = tf_export("raw_ops.FlushSummaryWriter")(_ops.to_raw_op(flush_summary_writer))


def flush_summary_writer_eager_fallback(writer, name, ctx):
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  _inputs_flat = [writer]
  _attrs = None
  _result = _execute.execute(b"FlushSummaryWriter", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def import_event(writer, event, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    event: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "ImportEvent", name,
        tld.op_callbacks, writer, event)
      return _result
    except _core._FallbackException:
      try:
        return import_event_eager_fallback(
            writer, event, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "ImportEvent", writer=writer, event=event, name=name)
  return _op
ImportEvent = tf_export("raw_ops.ImportEvent")(_ops.to_raw_op(import_event))


def import_event_eager_fallback(writer, event, name, ctx):
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  event = _ops.convert_to_tensor(event, _dtypes.string)
  _inputs_flat = [writer, event]
  _attrs = None
  _result = _execute.execute(b"ImportEvent", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def summary_writer(shared_name="", container="", name=None):
  r"""TODO: add doc.

  Args:
    shared_name: An optional `string`. Defaults to `""`.
    container: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "SummaryWriter", name,
        tld.op_callbacks, "shared_name", shared_name, "container", container)
      return _result
    except _core._FallbackException:
      try:
        return summary_writer_eager_fallback(
            shared_name=shared_name, container=container, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "SummaryWriter", shared_name=shared_name, container=container,
                         name=name)
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("shared_name", _op.get_attr("shared_name"), "container",
              _op.get_attr("container"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "SummaryWriter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

SummaryWriter = tf_export("raw_ops.SummaryWriter")(_ops.to_raw_op(summary_writer))


def summary_writer_eager_fallback(shared_name, container, name, ctx):
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  _inputs_flat = []
  _attrs = ("shared_name", shared_name, "container", container)
  _result = _execute.execute(b"SummaryWriter", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "SummaryWriter", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result


def write_audio_summary(writer, step, tag, tensor, sample_rate, max_outputs=3, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tag: A `Tensor` of type `string`.
    tensor: A `Tensor` of type `float32`.
    sample_rate: A `Tensor` of type `float32`.
    max_outputs: An optional `int` that is `>= 1`. Defaults to `3`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "WriteAudioSummary", name,
        tld.op_callbacks, writer, step, tag, tensor, sample_rate,
        "max_outputs", max_outputs)
      return _result
    except _core._FallbackException:
      try:
        return write_audio_summary_eager_fallback(
            writer, step, tag, tensor, sample_rate, max_outputs=max_outputs,
            name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if max_outputs is None:
    max_outputs = 3
  max_outputs = _execute.make_int(max_outputs, "max_outputs")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WriteAudioSummary", writer=writer, step=step, tag=tag, tensor=tensor,
                             sample_rate=sample_rate, max_outputs=max_outputs,
                             name=name)
  return _op
WriteAudioSummary = tf_export("raw_ops.WriteAudioSummary")(_ops.to_raw_op(write_audio_summary))


def write_audio_summary_eager_fallback(writer, step, tag, tensor, sample_rate, max_outputs, name, ctx):
  if max_outputs is None:
    max_outputs = 3
  max_outputs = _execute.make_int(max_outputs, "max_outputs")
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  tensor = _ops.convert_to_tensor(tensor, _dtypes.float32)
  sample_rate = _ops.convert_to_tensor(sample_rate, _dtypes.float32)
  _inputs_flat = [writer, step, tag, tensor, sample_rate]
  _attrs = ("max_outputs", max_outputs)
  _result = _execute.execute(b"WriteAudioSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def write_graph_summary(writer, step, tensor, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tensor: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "WriteGraphSummary", name,
        tld.op_callbacks, writer, step, tensor)
      return _result
    except _core._FallbackException:
      try:
        return write_graph_summary_eager_fallback(
            writer, step, tensor, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WriteGraphSummary", writer=writer, step=step, tensor=tensor,
                             name=name)
  return _op
WriteGraphSummary = tf_export("raw_ops.WriteGraphSummary")(_ops.to_raw_op(write_graph_summary))


def write_graph_summary_eager_fallback(writer, step, tensor, name, ctx):
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tensor = _ops.convert_to_tensor(tensor, _dtypes.string)
  _inputs_flat = [writer, step, tensor]
  _attrs = None
  _result = _execute.execute(b"WriteGraphSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def write_histogram_summary(writer, step, tag, values, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tag: A `Tensor` of type `string`.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "WriteHistogramSummary", name,
        tld.op_callbacks, writer, step, tag, values)
      return _result
    except _core._FallbackException:
      try:
        return write_histogram_summary_eager_fallback(
            writer, step, tag, values, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WriteHistogramSummary", writer=writer, step=step, tag=tag,
                                 values=values, name=name)
  return _op
WriteHistogramSummary = tf_export("raw_ops.WriteHistogramSummary")(_ops.to_raw_op(write_histogram_summary))


def write_histogram_summary_eager_fallback(writer, step, tag, values, name, ctx):
  _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, _dtypes.float32)
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [writer, step, tag, values]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"WriteHistogramSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def write_image_summary(writer, step, tag, tensor, bad_color, max_images=3, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tag: A `Tensor` of type `string`.
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `float32`, `half`.
    bad_color: A `Tensor` of type `uint8`.
    max_images: An optional `int` that is `>= 1`. Defaults to `3`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "WriteImageSummary", name,
        tld.op_callbacks, writer, step, tag, tensor, bad_color, "max_images",
        max_images)
      return _result
    except _core._FallbackException:
      try:
        return write_image_summary_eager_fallback(
            writer, step, tag, tensor, bad_color, max_images=max_images,
            name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  if max_images is None:
    max_images = 3
  max_images = _execute.make_int(max_images, "max_images")
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WriteImageSummary", writer=writer, step=step, tag=tag, tensor=tensor,
                             bad_color=bad_color, max_images=max_images,
                             name=name)
  return _op
WriteImageSummary = tf_export("raw_ops.WriteImageSummary")(_ops.to_raw_op(write_image_summary))


def write_image_summary_eager_fallback(writer, step, tag, tensor, bad_color, max_images, name, ctx):
  if max_images is None:
    max_images = 3
  max_images = _execute.make_int(max_images, "max_images")
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, _dtypes.float32)
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  bad_color = _ops.convert_to_tensor(bad_color, _dtypes.uint8)
  _inputs_flat = [writer, step, tag, tensor, bad_color]
  _attrs = ("max_images", max_images, "T", _attr_T)
  _result = _execute.execute(b"WriteImageSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def write_raw_proto_summary(writer, step, tensor, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tensor: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "WriteRawProtoSummary", name,
        tld.op_callbacks, writer, step, tensor)
      return _result
    except _core._FallbackException:
      try:
        return write_raw_proto_summary_eager_fallback(
            writer, step, tensor, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WriteRawProtoSummary", writer=writer, step=step, tensor=tensor,
                                name=name)
  return _op
WriteRawProtoSummary = tf_export("raw_ops.WriteRawProtoSummary")(_ops.to_raw_op(write_raw_proto_summary))


def write_raw_proto_summary_eager_fallback(writer, step, tensor, name, ctx):
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tensor = _ops.convert_to_tensor(tensor, _dtypes.string)
  _inputs_flat = [writer, step, tensor]
  _attrs = None
  _result = _execute.execute(b"WriteRawProtoSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def write_scalar_summary(writer, step, tag, value, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tag: A `Tensor` of type `string`.
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "WriteScalarSummary", name,
        tld.op_callbacks, writer, step, tag, value)
      return _result
    except _core._FallbackException:
      try:
        return write_scalar_summary_eager_fallback(
            writer, step, tag, value, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WriteScalarSummary", writer=writer, step=step, tag=tag, value=value,
                              name=name)
  return _op
WriteScalarSummary = tf_export("raw_ops.WriteScalarSummary")(_ops.to_raw_op(write_scalar_summary))


def write_scalar_summary_eager_fallback(writer, step, tag, value, name, ctx):
  _attr_T, (value,) = _execute.args_to_matching_eager([value], ctx)
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  _inputs_flat = [writer, step, tag, value]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"WriteScalarSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result


def write_summary(writer, step, tensor, tag, summary_metadata, name=None):
  r"""TODO: add doc.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tensor: A `Tensor`.
    tag: A `Tensor` of type `string`.
    summary_metadata: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx._context_handle, tld.device_name, "WriteSummary", name,
        tld.op_callbacks, writer, step, tensor, tag, summary_metadata)
      return _result
    except _core._FallbackException:
      try:
        return write_summary_eager_fallback(
            writer, step, tensor, tag, summary_metadata, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
  # Add nodes to the TensorFlow graph.
  _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "WriteSummary", writer=writer, step=step, tensor=tensor, tag=tag,
                        summary_metadata=summary_metadata, name=name)
  return _op
WriteSummary = tf_export("raw_ops.WriteSummary")(_ops.to_raw_op(write_summary))


def write_summary_eager_fallback(writer, step, tensor, tag, summary_metadata, name, ctx):
  _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx)
  writer = _ops.convert_to_tensor(writer, _dtypes.resource)
  step = _ops.convert_to_tensor(step, _dtypes.int64)
  tag = _ops.convert_to_tensor(tag, _dtypes.string)
  summary_metadata = _ops.convert_to_tensor(summary_metadata, _dtypes.string)
  _inputs_flat = [writer, step, tensor, tag, summary_metadata]
  _attrs = ("T", _attr_T)
  _result = _execute.execute(b"WriteSummary", 0, inputs=_inputs_flat,
                             attrs=_attrs, ctx=ctx, name=name)
  _result = None
  return _result

