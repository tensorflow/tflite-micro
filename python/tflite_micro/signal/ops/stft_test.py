# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Multi-op STFT analysis/synthesis pipeline test across TF, TFLite, and TFLM."""

import numpy as np
import tensorflow as tf

from tflite_micro.python.tflite_micro.signal.ops import fft_ops
from tflite_micro.python.tflite_micro.signal.ops import framer_op
from tflite_micro.python.tflite_micro.signal.ops import overlap_add_op
from tflite_micro.python.tflite_micro.signal.ops import window_op
from tflite_micro.python.tflite_micro.signal.utils import util


def _build_stft_reconstruction(config):
  """Builds an STFT analysis + synthesis reconstruction tf.function."""
  frame_size = config['frame_size']
  frame_step = config['frame_step']
  fft_length = config['fft_length']
  fft_dtype = tf.float32 if config['fft_type'] == 'float32' else tf.int32
  weight_shift = 14

  if config['window_type'] == 'hann':
    analysis_window_float = tf.signal.hann_window(frame_size, dtype=tf.float32)
    analysis_window_int = np.int16(
      np.round(analysis_window_float * 2**weight_shift)
    )
    synthesis_window_fn = tf.signal.inverse_stft_window_fn(
      frame_step, forward_window_fn=tf.signal.hann_window
    )
    synthesis_window_float = synthesis_window_fn(frame_size, dtype=tf.float32)
    synthesis_window_int = np.int16(
      np.round(synthesis_window_float * 2**weight_shift)
    )
  elif config['window_type'] == 'square_root_hann_cwola':
    analysis_window_float = window_op.square_root_hann_cwola_window_weights(
      frame_size, frame_step, weight_shift, dtype=float
    )
    analysis_window_int = window_op.square_root_hann_cwola_window_weights(
      frame_size, frame_step, weight_shift, dtype=np.int16
    )
    synthesis_window_int = analysis_window_int
    synthesis_window_float = analysis_window_float
  else:
    raise ValueError(f"Unsupported window type: {config['window_type']}")

  @tf.function
  def stft_analysis(signal_in):
    framer_out, _ = framer_op.framer(
      signal_in, frame_size, frame_step, prefill=True
    )
    if fft_dtype == tf.float32:
      window_in = tf.cast(framer_out, dtype=tf.float32)
      window_out = tf.math.multiply(window_in, analysis_window_float)
      window_out = tf.math.multiply(window_out, 1 / 32768)
    else:
      window_out = window_op.window(
        framer_out, analysis_window_int, weight_shift
      )
      window_out = tf.cast(window_out, dtype=tf.int32) * 65536
    return fft_ops.rfft(window_out, fft_length)

  @tf.function
  def stft_synthesis(subband_signal):
    ifft_out = fft_ops.irfft(subband_signal, fft_length)
    if fft_dtype == tf.float32:
      ifft_out_scaled = ifft_out * 32768
      window_out_float = tf.math.multiply(
        ifft_out_scaled, synthesis_window_float
      )
      window_out_float = tf.maximum(window_out_float, -32768)
      window_out_float = tf.minimum(window_out_float, 32767)
      window_out = tf.cast(window_out_float, tf.int16)
    else:
      ifft_out_float = tf.cast(ifft_out, dtype=tf.float32)
      ifft_out_scaled = ifft_out_float * (fft_length / 65536)
      ifft_out_int16 = tf.cast(ifft_out_scaled, tf.int16)
      window_out = window_op.window(
        ifft_out_int16, synthesis_window_int, weight_shift
      )
    return overlap_add_op.overlap_add(window_out, frame_step)

  @tf.function
  def stft_reconstruction(signal_in):
    return stft_synthesis(stft_analysis(signal_in))

  return stft_reconstruction


def _make_chirp_signal(num_channels, num_samples, sample_rate=16000):
  """Generates deterministic int16 linear chirp signals."""
  t = np.arange(num_samples, dtype=np.float64) / sample_rate
  duration = num_samples / sample_rate
  channels = []
  for ch in range(num_channels):
    f0 = 150.0 + 110.0 * ch
    f1 = 3500.0 - 300.0 * ch
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t * t)
    amplitude = 18000.0 - 1500.0 * ch
    channels.append(np.round(amplitude * np.sin(phase)).astype(np.int16))
  return np.stack(channels, axis=0)


class StftPipelineTest(tf.test.TestCase):
  def _run_stft_check(
    self,
    config,
    input_signal,
    block_size,
    error_rms_threshold,
    error_no_dc_rms_threshold,
  ):
    self.assertEqual(block_size % config['frame_step'], 0)
    num_channels, num_samples = input_signal.shape
    delay = config['frame_size'] - config['frame_step']
    num_blocks = int(np.ceil((num_samples + delay) / block_size))
    num_samples_padded = num_blocks * block_size
    pad_amount = num_samples_padded - num_samples

    audio_input = np.pad(input_signal, ((0, 0), (0, pad_amount)))
    audio_input = np.array(
      np.split(audio_input, num_blocks, axis=1), dtype=np.int16
    )
    audio_output = np.zeros(audio_input.shape, dtype=np.int16)

    graph = _build_stft_reconstruction(config)
    concrete_function = graph.get_concrete_function(
      tf.TensorSpec((num_channels, block_size), dtype=tf.int16)
    )
    interpreter = util.get_tflm_interpreter(concrete_function, graph)

    for block_index in range(num_blocks):
      in_frame = audio_input[block_index]

      # TF
      out_tf = concrete_function(in_frame).numpy()
      audio_output[block_index] = out_tf

      # TFLM or TFLite interpreter
      interpreter.set_input(in_frame, 0)
      interpreter.invoke()
      out_interp = interpreter.get_output(0)
      self.assertAllLessEqual(np.abs(out_interp - out_tf), 1)

    audio_output = np.reshape(
      np.swapaxes(audio_output, 0, 1), [num_channels, num_samples_padded]
    )
    error = audio_output[:, delay : delay + num_samples] - input_signal
    error_float = error / 2**15
    error_dc_float = np.mean(error) / 2**15
    error_power = np.mean(np.square(error_float))
    error_power_no_dc = np.mean(np.square(error_float - error_dc_float))

    rms_dbfs = 10 * np.log10(error_power) if error_power != 0 else float('-inf')
    rms_no_dc_dbfs = (
      10 * np.log10(error_power_no_dc)
      if error_power_no_dc != 0
      else float('-inf')
    )
    self.assertLess(rms_dbfs, error_rms_threshold)
    self.assertLess(rms_no_dc_dbfs, error_no_dc_rms_threshold)

  def testSingleChannelFloat32(self):
    config = {
      'frame_size': 320,
      'frame_step': 160,
      'fft_type': 'float32',
      'fft_length': 320,
      'window_type': 'hann',
    }
    signal = _make_chirp_signal(num_channels=1, num_samples=8000)
    self._run_stft_check(config, signal, 160, -84, -89)

  def testSingleChannelInt32(self):
    config = {
      'frame_size': 320,
      'frame_step': 160,
      'fft_type': 'int32',
      'fft_length': 320,
      'window_type': 'hann',
    }
    signal = _make_chirp_signal(num_channels=1, num_samples=8000)
    self._run_stft_check(config, signal, 160, -84, -89)

  def testMultiChannelFloat32(self):
    config = {
      'frame_size': 512,
      'frame_step': 256,
      'fft_type': 'float32',
      'fft_length': 512,
      'window_type': 'square_root_hann_cwola',
    }
    signal = _make_chirp_signal(num_channels=4, num_samples=8192)
    self._run_stft_check(config, signal, 256, -84, -89)

  def testMultiChannelInt32(self):
    config = {
      'frame_size': 512,
      'frame_step': 256,
      'fft_type': 'int32',
      'fft_length': 512,
      'window_type': 'square_root_hann_cwola',
    }
    signal = _make_chirp_signal(num_channels=4, num_samples=8192)
    self._run_stft_check(config, signal, 256, -84, -89)


if __name__ == '__main__':
  tf.test.main()
