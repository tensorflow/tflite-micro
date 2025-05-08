# Copyright 2016 The Gemmlowp Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""."""

import common


def _ReadParams(emitter, registers, input_address, elements, min_register):
  registers_count = (elements + 3) / 4
  registers = [
      registers.QuadRegister(min_register)
      for unused_i in range(registers_count)
  ]
  emitter.EmitVLoadAE(registers_count * 4, 32, registers, input_address, 64)
  return registers


def _Duplicate(emitter, registers, rows, values):
  """Populate a grid of registers duplicating provided values."""
  duplicated = []
  for i in range(rows):
    if i is rows - 1:
      duplicated.append(values[0])
    else:
      duplicated.append(registers.QuadRegister())

    emitter.EmitVDup('32', duplicated[i],
                     emitter.Lane(32, values[i / 4], i % 4))

  return duplicated


def _DuplicateGeneralRegister(emitter, registers, value, min_register):
  register = registers.QuadRegister(min_register)
  emitter.EmitVDup('32', register, value)
  return register


class _StaticQuantizationUInt8Transformation(object):
  """Calculate quantized values and cast back to uint8."""

  def Prepare(self, emitter, registers, kernel_m, kernel_n, lhs, rhs):
    """Load parameters and prepare duplicated registers."""
    emitter.EmitNewline()
    emitter.EmitComment('StaticQuantization::Prepare')

    lhs_offset = _ReadParams(emitter, registers, lhs, kernel_m, 4)
    self.rhs_offsets = _ReadParams(emitter, registers, rhs, kernel_n, 4)
    self.multiplicative_offset = _DuplicateGeneralRegister(
        emitter, registers,
        registers.MapParameter('multiplicative_offset',
                               'params.kernel.multiplicative_offset'), 4)
    self.rounding_offset = _DuplicateGeneralRegister(
        emitter, registers,
        registers.MapParameter('rounding_offset',
                               'params.kernel.rounding_offset'), 4)
    self.shift = _DuplicateGeneralRegister(
        emitter, registers,
        registers.MapParameter('shift', 'params.kernel.shift'), 4)
    self.lhs_offsets = _Duplicate(emitter, registers, kernel_m, lhs_offset)

  def Transform(self, emitter, registers, data, unused_kernel_m,
                unused_kernel_n):
    """Quantize the data."""
    emitter.EmitNewline()
    emitter.EmitComment('StaticQuantization::Transform')

    for (row, lhs_offset) in zip(data, self.lhs_offsets):
      for row_register in row:
        emitter.EmitVAdd('s32', row_register, row_register, lhs_offset)

    for row in data:
      for (row_register, rhs_offset_register) in zip(row, self.rhs_offsets):
        emitter.EmitVAdd('s32', row_register, row_register, rhs_offset_register)

    for row in data:
      for row_register in row:
        emitter.EmitVMul('i32', row_register, row_register,
                         self.multiplicative_offset)

    for row in data:
      for row_register in row:
        emitter.EmitVAdd('i32', row_register, row_register,
                         self.rounding_offset)

    for row in data:
      for row_register in row:
        emitter.EmitVShl('s32', row_register, row_register, self.shift)

    if len(data[0]) is 1:
      for row in data:
        emitter.EmitVQmovn('s32', row[0], row[0])

      for row in data:
        emitter.EmitVQmovun('s16', row[0], row[0])

      return data
    elif len(data[0]) is 2:
      results = []
      for row in data:
        emitter.EmitVQmovn2('s32', row[0], row[0], row[1])
        registers.FreeRegister(row[1])
        results.append([row[0]])

      for row in results:
        emitter.EmitVQmovun('s16', row[0], row[0])

      return results
    else:
      assert False

  def Type(self):
    return 8


class _StaticQuantizationInt32Transformation(object):
  """."""

  def Prepare(self, emitter, registers, kernel_m, kernel_n, lhs, rhs):
    emitter.EmitNewline()
    emitter.EmitComment('StaticQuantizationInt32::Prepare')

    lhs_offset = _ReadParams(emitter, registers, lhs, kernel_m, 4)
    self.rhs_offsets = _ReadParams(emitter, registers, rhs, kernel_n, 4)
    self.lhs_offsets = _Duplicate(emitter, registers, kernel_m, lhs_offset)

  def Transform(self, emitter, unused_registers, data, unused_kernel_m,
                unused_kernel_n):
    """Quantize data and output as int32."""
    emitter.EmitNewline()
    emitter.EmitComment('StaticQuantizationInt32::Transform')

    for (row, lhs_offset) in zip(data, self.lhs_offsets):
      for row_register in row:
        emitter.EmitVAdd('s32', row_register, row_register, lhs_offset)

    for row in data:
      for (row_register, rhs_offsets_register) in zip(row, self.rhs_offsets):
        emitter.EmitVAdd('s32', row_register, row_register,
                         rhs_offsets_register)

    return data

  def Type(self):
    return 32


class _StaticQuantizationFloatTransformation(object):
  """."""

  def Prepare(self, emitter, registers, kernel_m, kernel_n, lhs, rhs):
    emitter.EmitNewline()
    emitter.EmitComment('StaticQuantizationFloat::Prepare')

    lhs_offset = _ReadParams(emitter, registers, lhs, kernel_m, 4)
    self.rhs_offsets = _ReadParams(emitter, registers, rhs, kernel_n, 4)
    self.scale = _DuplicateGeneralRegister(
        emitter, registers,
        registers.MapParameter('scale', 'params.kernel.scale'), 4)
    self.lhs_offsets = _Duplicate(emitter, registers, kernel_m, lhs_offset)

  def Transform(self, emitter, unused_registers, data, unused_kernel_m,
                unused_kernel_n):
    """Quantize data and output as float."""
    emitter.EmitNewline()
    emitter.EmitComment('StaticQuantizationFloat::Transform')

    for (row, lhs_offset) in zip(data, self.lhs_offsets):
      for row_register in row:
        emitter.EmitVAdd('s32', row_register, row_register, lhs_offset)

    for row in data:
      for (row_register, rhs_offsets_register) in zip(row, self.rhs_offsets):
        emitter.EmitVAdd('s32', row_register, row_register,
                         rhs_offsets_register)

    for row in data:
      for row_register in row:
        emitter.EmitVCvt('f32', 's32', row_register, row_register)

    for row in data:
      for row_register in row:
        emitter.EmitVMul('f32', row_register, row_register, self.scale)

    return data

  def Type(self):
    return 32


class _RowMajorOutput(object):
  """Output data in row major layout."""

  def Prepare(self, emitter, registers, kernel_m, unused_kernel_n,
              unused_data_type):
    """Prepare strided load addresses."""
    emitter.EmitNewline()
    emitter.EmitComment('RowMajorOutput::Prepare')

    stride = registers.MapParameter('stride', 'params.output_stream.stride')

    self.outputs = []
    self.outputs.append(registers.MapOutputParameter('result'))

    for unused_i in range(kernel_m - 1):
      register = registers.GeneralRegister()
      emitter.EmitAdd(register, self.outputs[-1], stride)
      self.outputs.append(register)

  def Output(self, emitter, unused_registers, data, data_type, unused_kernel_m,
             kernel_n):
    emitter.EmitNewline()
    emitter.EmitComment('RowMajorOutput::Output')

    for (datum, output) in zip(data, self.outputs):
      emitter.EmitVStoreAE(data_type, kernel_n, datum, output, None)


def _GenerateAndClearAggregators(emitter, registers, count):
  """Prepare aggregators and emit aggregator clear code."""
  emitter.EmitNewline()
  emitter.EmitComment('Clear aggregators.')
  aggregators = [registers.QuadRegister() for unused_i in range(count)]
  for i in range(count):
    if i < 3:
      emitter.EmitVMov('i32', aggregators[i], emitter.ImmediateConstant(0))
    else:
      emitter.EmitVMov('i32', aggregators[i], aggregators[i - 3])
  return aggregators


def _Generate3x3LoadMultiplyAggregate(emitter, registers, aggregators, lhs, rhs,
                                      count):
  """Emit inner loop for 3 rows x 3 cols multiplication."""
  emitter.EmitNewline()
  emitter.EmitComment('3x3 lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()

  lhs_load = [registers.DoubleRegister() for unused_i in range(3)]
  rhs_load = [registers.DoubleRegister() for unused_i in range(3)]
  temp = [registers.QuadRegister() for unused_i in range(4)]

  emitter.EmitVLoadA(1, 8, rhs_load, emitter.DereferenceIncrement(rhs, 64))
  emitter.EmitVLoad(1, 8, lhs_load[0], emitter.DereferenceIncrement(lhs, 64))

  emitter.EmitVMull('u8', temp[0], lhs_load[0], rhs_load[0])
  emitter.EmitVLoad(1, 8, lhs_load[1], emitter.DereferenceIncrement(lhs, 64))

  emitter.EmitVMull('u8', temp[1], lhs_load[0], rhs_load[1])
  emitter.EmitVLoad(1, 8, lhs_load[2], emitter.DereferenceIncrement(lhs, 64))

  emitter.EmitVMull('u8', temp[2], lhs_load[0], rhs_load[2])
  emitter.EmitPldOffset(lhs, emitter.ImmediateConstant(64))

  emitter.EmitVMull('u8', temp[3], lhs_load[1], rhs_load[0])
  emitter.EmitPldOffset(rhs, emitter.ImmediateConstant(64))

  emitter.EmitVPadal('u16', aggregators[0], temp[0])
  emitter.EmitVPadal('u16', aggregators[1], temp[1])
  emitter.EmitVPadal('u16', aggregators[2], temp[2])
  emitter.EmitVPadal('u16', aggregators[3], temp[3])

  emitter.EmitVMull('u8', temp[0], lhs_load[1], rhs_load[1])
  emitter.EmitVMull('u8', temp[1], lhs_load[1], rhs_load[2])

  registers.FreeRegisters([lhs_load[0], lhs_load[1]])
  temp.append(registers.QuadRegister())

  emitter.EmitVMull('u8', temp[2], lhs_load[2], rhs_load[0])
  emitter.EmitVMull('u8', temp[3], lhs_load[2], rhs_load[1])

  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))
  emitter.EmitNewline()

  emitter.EmitVMull('u8', temp[4], lhs_load[2], rhs_load[2])

  emitter.EmitVPadal('u16', aggregators[4], temp[0])
  emitter.EmitVPadal('u16', aggregators[5], temp[1])
  emitter.EmitVPadal('u16', aggregators[6], temp[2])
  emitter.EmitVPadal('u16', aggregators[7], temp[3])
  emitter.EmitVPadal('u16', aggregators[8], temp[4])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBgtBack(1)

  registers.FreeRegisters(temp + [lhs_load[2]] + rhs_load)


def _Generate2x4LoadMultiplyAggregate(emitter, registers, aggregators, lhs, rhs,
                                      count):
  """Emit inner loop for 2 rows x 4 cols multiplication."""
  emitter.EmitNewline()
  emitter.EmitComment('2x4 lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()

  lhs_load = [registers.DoubleRegister() for unused_i in range(2)]
  rhs_load = [registers.DoubleRegister() for unused_i in range(4)]
  temp = [registers.QuadRegister() for unused_i in range(5)]

  emitter.EmitVLoadA(1, 8, rhs_load, emitter.DereferenceIncrement(rhs, 256))
  emitter.EmitVLoad(1, 8, lhs_load[0], emitter.DereferenceIncrement(lhs, 64))

  emitter.EmitVMull('u8', temp[0], lhs_load[0], rhs_load[0])
  emitter.EmitVLoad(1, 8, lhs_load[1], emitter.DereferenceIncrement(lhs, 64))

  emitter.EmitVMull('u8', temp[1], lhs_load[0], rhs_load[1])
  emitter.EmitPldOffset(rhs, emitter.ImmediateConstant(64))

  emitter.EmitVMull('u8', temp[2], lhs_load[0], rhs_load[2])
  emitter.EmitPldOffset(lhs, emitter.ImmediateConstant(64))

  emitter.EmitVMull('u8', temp[3], lhs_load[0], rhs_load[3])
  emitter.EmitVMull('u8', temp[4], lhs_load[1], rhs_load[0])

  emitter.EmitVPadal('u16', aggregators[0], temp[0])
  emitter.EmitVPadal('u16', aggregators[1], temp[1])
  emitter.EmitVPadal('u16', aggregators[2], temp[2])

  emitter.EmitVMull('u8', temp[0], lhs_load[1], rhs_load[1])
  emitter.EmitVMull('u8', temp[1], lhs_load[1], rhs_load[2])
  emitter.EmitVMull('u8', temp[2], lhs_load[1], rhs_load[3])

  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))

  emitter.EmitNewline()
  emitter.EmitVPadal('u16', aggregators[3], temp[3])
  emitter.EmitVPadal('u16', aggregators[4], temp[4])
  emitter.EmitVPadal('u16', aggregators[5], temp[0])
  emitter.EmitVPadal('u16', aggregators[6], temp[1])
  emitter.EmitVPadal('u16', aggregators[7], temp[2])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBgtBack(1)

  registers.FreeRegisters(temp + lhs_load + rhs_load)


def _Generate1x8LoadMultiplyAggregate(emitter, registers, aggregators, lhs, rhs,
                                      count):
  """Emit inner loop for 1 rows x 8 cols multiplication."""
  emitter.EmitNewline()
  emitter.EmitComment('1x8 lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()

  lhs_load = registers.DoubleRegister()
  rhs_load = [registers.DoubleRegister() for unused_i in range(4)]
  temp = [registers.QuadRegister() for unused_i in range(5)]

  emitter.EmitVLoadAE(4 * 8, 8, rhs_load, rhs, 256)
  emitter.EmitVLoadE(8, 8, lhs_load, lhs, 64)

  emitter.EmitVMull('u8', temp[0], lhs_load, rhs_load[0])
  emitter.EmitVMull('u8', temp[1], lhs_load, rhs_load[1])
  emitter.EmitVMull('u8', temp[2], lhs_load, rhs_load[2])
  emitter.EmitVMull('u8', temp[3], lhs_load, rhs_load[3])

  emitter.EmitVLoadAE(4 * 8, 8, rhs_load, rhs, 256)

  emitter.EmitVPadal('u16', aggregators[0], temp[0])
  emitter.EmitVPadal('u16', aggregators[1], temp[1])
  emitter.EmitVPadal('u16', aggregators[2], temp[2])
  emitter.EmitVPadal('u16', aggregators[3], temp[3])

  emitter.EmitPldOffset(rhs, emitter.ImmediateConstant(256))

  emitter.EmitVMull('u8', temp[4], lhs_load, rhs_load[0])
  emitter.EmitVMull('u8', temp[0], lhs_load, rhs_load[1])
  emitter.EmitVMull('u8', temp[1], lhs_load, rhs_load[2])
  emitter.EmitVMull('u8', temp[2], lhs_load, rhs_load[3])

  emitter.EmitPldOffset(lhs, emitter.ImmediateConstant(32))

  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))

  emitter.EmitNewline()
  emitter.EmitVPadal('u16', aggregators[4], temp[4])
  emitter.EmitVPadal('u16', aggregators[5], temp[0])
  emitter.EmitVPadal('u16', aggregators[6], temp[1])
  emitter.EmitVPadal('u16', aggregators[7], temp[2])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBgtBack(1)

  registers.FreeRegisters(temp + [lhs_load] + rhs_load)


def _GenerateNxMLoadMultiplyAggregate(emitter, registers, kernel_m, kernel_n,
                                      aggregators, lhs, rhs, count):
  """Emit inner loop for N rows x M cols multiplication."""
  emitter.EmitNewline()
  emitter.EmitComment('General NxM lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))
  emitter.EmitNewline()

  lhs_load = [registers.DoubleRegister() for unused_i in range(kernel_m)]
  rhs_load = [registers.DoubleRegister() for unused_i in range(kernel_n)]

  emitter.EmitVLoadAE(8 * kernel_m, 8, lhs_load, lhs, 64)
  emitter.EmitVLoadAE(8 * kernel_n, 8, rhs_load, rhs, 64)

  emitter.EmitPldOffset(lhs, emitter.ImmediateConstant(64))
  emitter.EmitPldOffset(rhs, emitter.ImmediateConstant(64))

  results = [
      registers.QuadRegister() for unused_i in range(kernel_m * kernel_n)
  ]

  for row in range(kernel_m):
    for col in range(kernel_n):
      index = row * kernel_n + col
      emitter.EmitVMull('u8', results[index], rhs_load[col], lhs_load[row])

  for i in range(kernel_m * kernel_n):
    emitter.EmitVPadal('u16', aggregators[i], results[i])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBgtBack(1)

  registers.FreeRegisters(lhs_load + rhs_load + results)


def _Generate1xNLoadMultiplyAggregate(emitter, registers, kernel_n, aggregators,
                                      lhs, rhs, count):
  """Emit inner loop for 1 row x M cols multiplication."""
  assert kernel_n in [5, 6, 7, 8]
  emitter.EmitNewline()
  emitter.EmitComment('General 1xM lanes loop.')
  emitter.EmitNumericalLabel(1)
  emitter.EmitNewline()
  emitter.EmitComment('Subtract counter.')
  emitter.EmitSubs(count, count, emitter.ImmediateConstant(8))
  emitter.EmitNewline()

  leftover = kernel_n - 4

  rhs_load = [registers.DoubleRegister() for unused_i in range(4)]
  lhs_load = registers.DoubleRegister()

  emitter.EmitVLoadAE(8 * 4, 8, rhs_load, rhs, 64)
  emitter.EmitVLoadE(8, 8, lhs_load, lhs, 64)

  emitter.EmitPldOffset(lhs, emitter.ImmediateConstant(64))

  results = [registers.QuadRegister() for unused_i in range(4)]

  for i in range(4):
    emitter.EmitVMull('u8', results[i], rhs_load[i], lhs_load)

  emitter.EmitVLoadAE(8 * leftover, 8, rhs_load, rhs, 64)
  emitter.EmitPldOffset(rhs, emitter.ImmediateConstant(128))

  for i in range(4):
    emitter.EmitVPadal('u16', aggregators[i], results[i])

  for i in range(leftover):
    emitter.EmitVMull('u8', results[i], rhs_load[i], lhs_load)

  for i in range(leftover):
    emitter.EmitVPadal('u16', aggregators[i + 4], results[i])

  emitter.EmitNewline()
  emitter.EmitComment('Loop break.')
  emitter.EmitBgtBack(1)

  registers.FreeRegisters([lhs_load] + rhs_load + results)


def _GenerateMultiplyKernel(emitter, registers, kernel_m, kernel_n, lhs, rhs):
  """Main muliply loop. Pick best implementation for given kernel shape."""
  count = registers.MapParameter('count', 'params.kernel.count')

  aggregators = _GenerateAndClearAggregators(emitter, registers,
                                             kernel_m * kernel_n)
  if kernel_m is 3 and kernel_n is 3:
    _Generate3x3LoadMultiplyAggregate(emitter, registers, aggregators, lhs, rhs,
                                      count)
  elif kernel_m is 2 and kernel_n is 4:
    _Generate2x4LoadMultiplyAggregate(emitter, registers, aggregators, lhs, rhs,
                                      count)
  elif kernel_m is 1 and kernel_n is 8:
    _Generate1x8LoadMultiplyAggregate(emitter, registers, aggregators, lhs, rhs,
                                      count)
  elif kernel_m is 1 and kernel_n > 4:
    _Generate1xNLoadMultiplyAggregate(emitter, registers, kernel_n, aggregators,
                                      lhs, rhs, count)
  else:
    _GenerateNxMLoadMultiplyAggregate(emitter, registers, kernel_m, kernel_n,
                                      aggregators, lhs, rhs, count)
  return aggregators


def _ReduceAggregators(emitter, aggregators):
  reduced_count = (len(aggregators) + 3) / 4
  reduced = aggregators[:reduced_count]
  emitter.EmitVSumReduce('u32', len(aggregators), 4, reduced, aggregators)
  return reduced


def _GenerateAggregatorReduce(emitter, aggregators, kernel_m, kernel_n):
  emitter.EmitNewline()
  emitter.EmitComment('Reduce aggregators.')
  row_temps = []
  for i in range(kernel_m):
    row_temps.append(
        _ReduceAggregators(emitter, aggregators[i * kernel_n:(i + 1) *
                                                kernel_n]))
  return row_temps


class QuantizedMulKernel(common.MulKernelGenerator):
  """."""

  def __init__(self, cc_emitter, kernel_name, output_stream_name, asm_emitter,
               fused_transformation, output_strategy):
    common.MulKernelGenerator.__init__(self, cc_emitter, kernel_name,
                                       output_stream_name)
    self.asm_emitter = asm_emitter
    self.fused_transformation = fused_transformation
    self.output_strategy = output_strategy

  def EmitMultiply(self, in_type, out_type, kernel_m, kernel_n, pack_size):
    assert in_type is 'uint8_t'
    assert pack_size is 8
    assert kernel_m * kernel_n <= 9

    registers = self.asm_emitter.CreateRegisters()

    self.asm_emitter.PushIndent(self.emitter.indent)
    self.asm_emitter.EmitAsmBegin()

    lhs = registers.MapOutputParameter('lhs')
    rhs = registers.MapOutputParameter('rhs')
    self.asm_emitter.EmitPld(lhs)
    self.asm_emitter.EmitPld(rhs)

    aggregators = _GenerateMultiplyKernel(self.asm_emitter, registers, kernel_m,
                                          kernel_n, lhs, rhs)

    self.fused_transformation.Prepare(self.asm_emitter, registers, kernel_m,
                                      kernel_n, lhs, rhs)

    self.output_strategy.Prepare(self.asm_emitter, registers, kernel_m,
                                 kernel_n, self.fused_transformation.Type())

    reduced = _GenerateAggregatorReduce(self.asm_emitter, aggregators, kernel_m,
                                        kernel_n)

    transformed = self.fused_transformation.Transform(self.asm_emitter,
                                                      registers, reduced,
                                                      kernel_m, kernel_n)

    self.output_strategy.Output(self.asm_emitter, registers, transformed,
                                self.fused_transformation.Type(), kernel_m,
                                kernel_n)

    self.asm_emitter.EmitAsmEnd(registers)
    self.asm_emitter.PopIndent(len(self.emitter.indent))


class QuantizedMulStaticRowMajor(QuantizedMulKernel):
  """."""

  def __init__(self, cc_emitter, asm_emitter):
    QuantizedMulKernel.__init__(self, cc_emitter, 'QuantizedStaticPreprocessed',
                                'RowMajor', asm_emitter,
                                _StaticQuantizationUInt8Transformation(),
                                _RowMajorOutput())


class QuantizedMulStaticAsInt32RowMajor(QuantizedMulKernel):
  """."""

  def __init__(self, cc_emitter, asm_emitter):
    QuantizedMulKernel.__init__(self, cc_emitter,
                                'QuantizedStaticPreprocessedAsInt32',
                                'RowMajor', asm_emitter,
                                _StaticQuantizationInt32Transformation(),
                                _RowMajorOutput())


class QuantizedMulStaticAsFloatRowMajor(QuantizedMulKernel):
  """."""

  def __init__(self, cc_emitter, asm_emitter):
    QuantizedMulKernel.__init__(self, cc_emitter,
                                'QuantizedStaticPreprocessedAsFloat',
                                'RowMajor', asm_emitter,
                                _StaticQuantizationFloatTransformation(),
                                _RowMajorOutput())


def GenerateKernels(cc_emitter, asm_emitter, shapes):
  """Generate the quantized multiplication kernels for uint8 operands."""
  quantized_mul_static_row_major = QuantizedMulStaticRowMajor(cc_emitter,
                                                              asm_emitter)
  quantized_mul_static_int32_row_major = QuantizedMulStaticAsInt32RowMajor(
      cc_emitter, asm_emitter)

  quantized_mul_static_float_row_major = QuantizedMulStaticAsFloatRowMajor(
      cc_emitter, asm_emitter)

  for shape in shapes:
    quantized_mul_static_row_major.SpecializeMulKernel('uint8_t', 'uint8_t',
                                                       shape[0], shape[1], 8)
  for shape in shapes:
    quantized_mul_static_int32_row_major.SpecializeMulKernel('uint8_t',
                                                             'int32_t',
                                                             shape[0], shape[1],
                                                             8)

  for shape in shapes:
    quantized_mul_static_float_row_major.SpecializeMulKernel('uint8_t', 'float',
                                                             shape[0], shape[1],
                                                             8)
