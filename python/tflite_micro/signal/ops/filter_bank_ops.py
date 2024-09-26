# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Use filter bank ops in python."""

import numpy as np
import tensorflow as tf

from tflite_micro.python.tflite_micro.signal.utils import util
from tflite_micro.python.tflite_micro.signal.utils.freq_to_mel_wrapper import freq_to_mel

gen_filter_bank_ops = util.load_custom_op('filter_bank_ops.so')

# A note about precision:
# The code to calculate center frequencies and weights uses floating point
# extensively. The original speech micro code is written in C and uses
# 32-bit 'float' C types. Python's floating point type is 64-bit by default,
# which resulted in slight differences that made verification harder.
# In order to establish parity with speech micro, and recognizing the slight
# loss in precision, numpy.float32 was used throughout this code instead of
# the default Python 'float' type. For the same reason, the function freq_to_mel
# wraps the same FreqToMel() C function used by Speech Micro.

FILTER_BANK_ALIGNMENT = 1
FILTER_BANK_CHANNEL_BLOCK_SIZE = 1
FILTER_BANK_WEIGHT_SCALING_BITS = 12


def _calc_center_freq(channel_num, lower_freq_limit, upper_freq_limit):
  """Calculate the center frequencies of mel spectrum filter banks."""
  if lower_freq_limit < 0:
    raise ValueError("Lower frequency limit must be non negative")
  if lower_freq_limit > upper_freq_limit:
    raise ValueError("Lower frequency limit can't be larger than upper limit")
  mel_lower = np.float32(freq_to_mel(lower_freq_limit))
  mel_upper = np.float32(freq_to_mel(upper_freq_limit))
  mel_span = mel_upper - mel_lower
  mel_spacing = mel_span / np.float32(channel_num)
  channels = np.arange(1, channel_num + 1, dtype=np.float32)
  return mel_lower + (mel_spacing * channels)


def _quantize_filterbank_weight(float_weight, scale_bits):
  """Scale float filterbank weights return the integer weights and unweights."""
  weight = int(float_weight * (1 << scale_bits))
  unweight = int((1 - float_weight) * (1 << scale_bits))
  return weight, unweight


def _init_filter_bank_weights(spectrum_size, sample_rate, alignment,
                              channel_block_size, num_channels,
                              lower_band_limit, upper_band_limit):
  """Initialize mel-spectrum filter bank weights."""
  # How should we align things to index counts given the byte alignment?
  item_size = np.dtype("int16").itemsize
  if alignment < item_size:
    index_alignment = 1
  else:
    index_alignment = int(alignment / item_size)

  channel_frequency_starts = np.zeros(num_channels + 1, dtype=np.int16)
  channel_weight_starts = np.zeros(num_channels + 1, dtype=np.int16)
  channel_widths = np.zeros(num_channels + 1, dtype=np.int16)

  actual_channel_starts = np.zeros(num_channels + 1, dtype=np.int16)
  actual_channel_widths = np.zeros(num_channels + 1, dtype=np.int16)

  center_mel_freqs = _calc_center_freq(num_channels + 1, lower_band_limit,
                                       upper_band_limit)

  # (spectrum_size - 1) to exclude DC. Emulate Hidden Markov Model Toolkit (HTK)
  hz_per_sbin = (sample_rate / 2) / (spectrum_size - 1)
  # (1 + ...) to exclude DC.
  start_index = round(1 + (lower_band_limit / hz_per_sbin))

  # For each channel, we need to figure out what frequencies belong to it, and
  # how much padding we need to add so that we can efficiently multiply the
  # weights and unweights for accumulation. To simplify the multiplication
  # logic, all channels will have some multiplication to do (even if there are
  # no frequencies that accumulate to that channel) - they will be directed to
  # a set of zero weights.
  chan_freq_index_start = start_index
  weight_index_start = 0
  needs_zeros = 0

  for chan in range(num_channels + 1):
    # Keep jumping frequencies until we overshoot the bound on this channel.
    freq_index = chan_freq_index_start
    while freq_to_mel(freq_index * hz_per_sbin) <= center_mel_freqs[chan]:
      freq_index += 1

    width = freq_index - chan_freq_index_start
    actual_channel_starts[chan] = chan_freq_index_start
    actual_channel_widths[chan] = width

    if width == 0:
      # This channel doesn't actually get anything from the frequencies, it's
      # always zero. We need then to insert some 'zero' weights into the
      # output, and just redirect this channel to do a single multiplication at
      # this point. For simplicity, the zeros are placed at the beginning of
      # the weights arrays, so we have to go and update all the other
      # weight_starts to reflect this shift (but only once).
      channel_frequency_starts[chan] = 0
      channel_weight_starts[chan] = 0
      channel_widths[chan] = channel_block_size
      if needs_zeros == 0:
        needs_zeros = 1
        for j in range(chan):
          channel_weight_starts[j] += channel_block_size
        weight_index_start += channel_block_size
    else:
      # How far back do we need to go to ensure that we have the proper
      # alignment?
      aligned_start = int(
          chan_freq_index_start / index_alignment) * index_alignment
      aligned_width = (chan_freq_index_start - aligned_start + width)
      padded_width = (int(
          (aligned_width - 1) / channel_block_size) + 1) * channel_block_size

      channel_frequency_starts[chan] = aligned_start
      channel_weight_starts[chan] = weight_index_start
      channel_widths[chan] = padded_width
      weight_index_start += padded_width
    chan_freq_index_start = freq_index

  # Allocate the two arrays to store the weights - weight_index_start contains
  # the index of what would be the next set of weights that we would need to
  # add, so that's how many weights we need to allocate.
  num_weights = weight_index_start
  weights = np.zeros(num_weights, dtype=np.int16)
  unweights = np.zeros(num_weights, dtype=np.int16)

  # Next pass, compute all the weights. Since everything has been memset to
  # zero, we only need to fill in the weights that correspond to some frequency
  # for a channel.
  end_index = 0
  mel_low = freq_to_mel(lower_band_limit)
  for chan in range(num_channels + 1):
    frequency = actual_channel_starts[chan]
    num_frequencies = actual_channel_widths[chan]
    frequency_offset = frequency - channel_frequency_starts[chan]
    weight_start = channel_weight_starts[chan]
    if chan == 0:
      denom_val = mel_low
    else:
      denom_val = center_mel_freqs[chan - 1]
    for j in range(num_frequencies):
      num = np.float32(center_mel_freqs[chan] -
                       freq_to_mel(frequency * hz_per_sbin))
      den = np.float32(center_mel_freqs[chan] - denom_val)
      weight = num / den
      # Make the float into an integer for the weights (and unweights).
      # Explicetly cast to int64. Numpy 2.0 introduces downcasting if we don't
      weight_index = weight_start + np.int64(frequency_offset) + j
      weights[weight_index], unweights[
          weight_index] = _quantize_filterbank_weight(
              weight, FILTER_BANK_WEIGHT_SCALING_BITS)
      # Explicetly cast to int64. Numpy 2.0 introduces downcasting if we don't
      frequency = np.int64(frequency) + 1
    if frequency > end_index:
      end_index = frequency

  if end_index >= spectrum_size:
    raise ValueError("Lower frequency limit can't be larger than upper limit")

  return (start_index, end_index, weights, unweights, channel_frequency_starts,
          channel_weight_starts, channel_widths)


def calc_start_end_indices(fft_length, sample_rate, num_channels,
                           lower_band_limit, upper_band_limit):
  """Returns the range of FFT indices needed by mel-spectrum filter bank.

  The caller can use the indices to avoid calculating the energy of FFT bins
  that won't be used.

  Args:
    fft_length: Length of FFT, in bins.
    sample_rate: Sample rate, in Hz.
    num_channels: Number of mel-spectrum filter bank channels.
    lower_band_limit: lower limit of mel-spectrum filterbank, in Hz.
    upper_band_limit: upper limit of mel-spectrum filterbank, in Hz.

  Returns:
    A pair: start and end indices, in the range [0, fft_length)

  Raises:
    ValueError: If fft_length isn't a power of 2
  """
  if fft_length % 2 != 0:
    raise ValueError("FFT length must be an even number")
  spectrum_size = fft_length / 2 + 1
  (start_index, end_index, _, _, _, _,
   _) = _init_filter_bank_weights(spectrum_size, sample_rate,
                                  FILTER_BANK_ALIGNMENT,
                                  FILTER_BANK_CHANNEL_BLOCK_SIZE, num_channels,
                                  lower_band_limit, upper_band_limit)
  return start_index, end_index


def _filter_bank_wrapper(filter_bank_fn, default_name):
  """Wrapper around gen_filter_bank_ops.filter_bank*."""

  def _filter_bank(input_tensor,
                   sample_rate,
                   num_channels,
                   lower_band_limit,
                   upper_band_limit,
                   name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint32)
      dim_list = input_tensor.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Input tensor must have a rank of 1")
      spectrum_size = dim_list[0]

      (_, _, weights, unweights, channel_frequency_starts,
       channel_weight_starts, channel_widths) = _init_filter_bank_weights(
           spectrum_size, sample_rate, FILTER_BANK_ALIGNMENT,
           FILTER_BANK_CHANNEL_BLOCK_SIZE, num_channels, lower_band_limit,
           upper_band_limit)
      weights_tensor = tf.convert_to_tensor(weights, dtype=tf.int16)
      unweights_tensor = tf.convert_to_tensor(unweights, dtype=tf.int16)
      channel_frequency_starts_tensor = tf.convert_to_tensor(
          channel_frequency_starts, dtype=tf.int16)
      channel_weight_starts_tensor = tf.convert_to_tensor(
          channel_weight_starts, dtype=tf.int16)
      channel_widths_tensor = tf.convert_to_tensor(channel_widths,
                                                   dtype=tf.int16)

      return filter_bank_fn(input_tensor,
                            weights_tensor,
                            unweights_tensor,
                            channel_frequency_starts_tensor,
                            channel_weight_starts_tensor,
                            channel_widths_tensor,
                            num_channels=num_channels,
                            name=name)

  return _filter_bank


def _filter_bank_square_root_wrapper(filter_bank_square_root_fn, default_name):
  """Wrapper around gen_filter_bank_ops.filter_bank_square_root*."""

  def _filter_bank_square_root(input_tensor, scale_bits, name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint64)
      dim_list = input_tensor.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Input tensor must have a rank of 1")
      scale_bits_tensor = tf.convert_to_tensor(scale_bits, dtype=tf.int32)
      return filter_bank_square_root_fn(input_tensor,
                                        scale_bits_tensor,
                                        name=name)

  return _filter_bank_square_root


def _filter_bank_spectral_subtraction_wrapper(
    filter_bank_spectral_subtraction_fn, default_name):
  """Wrapper around gen_filter_bank_ops.filter_bank_spectral_subtraction*."""

  def _filter_bank_spectral_subtraction(input_tensor,
                                        num_channels,
                                        smoothing,
                                        alternate_smoothing,
                                        smoothing_bits,
                                        min_signal_remaining,
                                        clamping,
                                        spectral_subtraction_bits=14,
                                        name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint32)
      dim_list = input_tensor.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Input tensor must have a rank of 1")

      min_signal_remaining = int(min_signal_remaining *
                                 (1 << spectral_subtraction_bits))
      # Alternate smoothing may be disabled
      if alternate_smoothing == 0:
        alternate_smoothing = smoothing

      smoothing = int(smoothing * (1 << spectral_subtraction_bits))
      one_minus_smoothing = int((1 << spectral_subtraction_bits) - smoothing)
      alternate_smoothing = int(alternate_smoothing *
                                (1 << spectral_subtraction_bits))
      alternate_one_minus_smoothing = int((1 << spectral_subtraction_bits) -
                                          alternate_smoothing)
      return filter_bank_spectral_subtraction_fn(
          input_tensor,
          num_channels=num_channels,
          smoothing=smoothing,
          one_minus_smoothing=one_minus_smoothing,
          alternate_smoothing=alternate_smoothing,
          alternate_one_minus_smoothing=alternate_one_minus_smoothing,
          smoothing_bits=smoothing_bits,
          min_signal_remaining=min_signal_remaining,
          clamping=clamping,
          spectral_subtraction_bits=spectral_subtraction_bits,
          name=name)

  return _filter_bank_spectral_subtraction


def _filter_bank_log_wrapper(filter_bank_log_fn, default_name):
  """Wrapper around gen_filter_bank_ops.filter_bank_log*."""

  def _filter_bank_log(input_tensor,
                       output_scale,
                       input_correction_bits,
                       name=default_name):
    with tf.name_scope(name) as name:
      input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint32)
      dim_list = input_tensor.shape.as_list()
      if len(dim_list) != 1:
        raise ValueError("Input tensor must have a rank of 1")

      return filter_bank_log_fn(input_tensor,
                                output_scale=output_scale,
                                input_correction_bits=input_correction_bits,
                                name=name)

  return _filter_bank_log


filter_bank = _filter_bank_wrapper(gen_filter_bank_ops.signal_filter_bank,
                                   "signal_filter_bank")
filter_bank_square_root = _filter_bank_square_root_wrapper(
    gen_filter_bank_ops.signal_filter_bank_square_root,
    "signal_filter_bank_square_root")
filter_bank_spectral_subtraction = _filter_bank_spectral_subtraction_wrapper(
    gen_filter_bank_ops.signal_filter_bank_spectral_subtraction,
    "signal_filter_bank_spectral_subtraction")
filter_bank_log = _filter_bank_log_wrapper(
    gen_filter_bank_ops.signal_filter_bank_log, "signal_filter_bank_log")

tf.no_gradient("signal_filter_bank")
tf.no_gradient("signal_filter_bank_square_root")
tf.no_gradient("signal_filter_bank_spectral_subtraction")
tf.no_gradient("signal_filter_bank_log")
