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


def _AlignForLanes(lanes_count):
  if lanes_count is 8 or lanes_count is 4:
    return 256
  elif lanes_count is 6 or lanes_count is 2:
    return 128
  else:
    return 64


def _AlignForSums(lanes_count):
  if lanes_count is 8:
    return 256
  elif lanes_count in [2, 4, 6]:
    return 128
  else:
    return 64


def _GenerateInputs(emitter, registers, lanes_count, input_address, stride):
  """."""
  inputs = []
  last_address_register = input_address
  for i in range(lanes_count):
    if not i:
      inputs.append(input_address)
    else:
      address_register = registers.GeneralRegister()
      inputs.append(address_register)
      emitter.EmitAdd(address_register, last_address_register, stride)
      last_address_register = address_register
  return inputs


def _GenerateClear(emitter, clear_type, block):
  for row in block:
    emitter.EmitVMov(clear_type, row, emitter.ImmediateConstant(0))


def _GenerateLoadAggregateStore(emitter, registers, lanes_count, elements_count,
                                aggregators, inputs, output):
  """Emit inner loop code for reading N lanes and interweaving them."""
  emitter.EmitNewline()
  emitter.EmitComment('Load Aggregate Store: %dx%d.' % (lanes_count,
                                                        elements_count))

  block = [registers.DoubleRegister() for unused_i in range(lanes_count)]

  if elements_count is not 8:
    _GenerateClear(emitter, 'i8', block)

  for (row, input_address) in zip(block, inputs):
    emitter.EmitVLoadE(8, elements_count, row, input_address, None)

  for (aggregator, row) in zip(aggregators, block):
    emitter.EmitVAddw('u8', aggregator, aggregator, row)

  emitter.EmitVStoreAE(8, 8 * lanes_count, block, output,
                       _AlignForLanes(lanes_count))

  registers.FreeRegisters(block)


def _LoadMemoryParameter(emitter, registers, name, source):
  register = registers.GeneralRegister()
  emitter.EmitLdr(register, registers.MapMemoryParameter(name, source))
  return register


def _GenerateAggregatorReductionLowRegisters(emitter, registers,
                                             aggregators, output_address):
  emitter.EmitNewline()
  emitter.EmitComment('Aggregator Reduction.')
  _GenerateAggregatorReduction(
      emitter, registers, aggregators, output_address,
      _LoadMemoryParameter(emitter, registers, 'multiplicative_sum_offset',
                           'params.multiplicative_sum_offset'),
      _LoadMemoryParameter(emitter, registers, 'additive_sum_offset',
                           'params.additive_sum_offset'))


def _GenerateAggregatorReductionHighRegisters(emitter, registers,
                                              aggregators, output_address):
  emitter.EmitNewline()
  emitter.EmitComment('Aggregator Reduction.')
  _GenerateAggregatorReduction(
      emitter, registers, aggregators, output_address,
      registers.MapParameter('multiplicative_sum_offset',
                             'params.multiplicative_sum_offset'),
      registers.MapParameter('additive_sum_offset',
                             'params.additive_sum_offset'))


def _GenerateAggregatorReduction(emitter, registers, aggregators,
                                 output_address, multiplicative_sum_offset,
                                 additive_sum_offset):
  """Reduce 4 lane sum aggregators to 1 value and store the sums."""
  multiplier = registers.DoubleRegister()
  emitter.EmitVMov('32',
                   emitter.Lane(32, multiplier, 0), multiplicative_sum_offset)

  offset = registers.QuadRegister()
  emitter.EmitVDup('32', offset, additive_sum_offset)

  for aggregator in aggregators:
    emitter.EmitVPaddl('u16', aggregator, aggregator)

  reduced_count = (len(aggregators) + 3) / 4
  reduced = aggregators[:reduced_count]

  emitter.EmitVSumReduce('u32', len(aggregators), 4, reduced, aggregators)

  for temp in reduced:
    emitter.EmitVMulScalar('i32', temp, temp, emitter.Lane(32, multiplier, 0))

  for temp in reduced:
    emitter.EmitVAdd('i32', temp, temp, offset)

  emitter.EmitVStoreA(1, 32, reduced,
                      emitter.Dereference(output_address,
                                          _AlignForSums(len(aggregators))))


class RowMajorWithSumUInt8x8(common.StreamGenerator):
  """."""

  def __init__(self, emitter, asm_emitter):
    common.StreamGenerator.__init__(self, emitter, 'RowMajorWithSum')
    self.asm_emitter = asm_emitter

  def EmitPack(self, in_type, lanes_count, pack_size, leftovers):
    assert pack_size is 8
    assert in_type is 'uint8_t'

    registers = self.asm_emitter.CreateRegisters()

    self.emitter.EmitDeclare('int', 'params_count_copy', 'params.count')

    self.asm_emitter.PushIndent(self.emitter.indent)
    self.asm_emitter.EmitAsmBegin()

    count = registers.MapOutputParameter('count', 'params_count_copy')
    output = registers.MapOutputParameter('out')
    inputs = _GenerateInputs(self.asm_emitter, registers, lanes_count,
                             registers.MapOutputParameter('in'),
                             registers.MapParameter('stride', 'params.stride'))
    aggregators = [registers.QuadRegister(8) for unused_i in range(lanes_count)]

    _GenerateClear(self.asm_emitter, 'i16', aggregators)

    if leftovers:
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitComment('Reduce count by leftovers.')
      self.asm_emitter.EmitSubs(count, count,
                                self.asm_emitter.ImmediateConstant(leftovers))
      self.asm_emitter.EmitBeqFront(2)

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitNumericalLabel(1)
    self.asm_emitter.EmitSubs(count, count,
                              self.asm_emitter.ImmediateConstant(8))

    _GenerateLoadAggregateStore(self.asm_emitter, registers, lanes_count, 8,
                                aggregators, inputs, output)

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitBneBack(1)

    if leftovers:
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitNumericalLabel(2)
      _GenerateLoadAggregateStore(self.asm_emitter, registers, lanes_count,
                                  leftovers, aggregators, inputs, output)

    registers.FreeRegisters(inputs)

    if len(inputs) <= 6:
      _GenerateAggregatorReductionHighRegisters(
          self.asm_emitter, registers, aggregators, output)
    else:
      _GenerateAggregatorReductionLowRegisters(
          self.asm_emitter, registers, aggregators, output)

    self.asm_emitter.EmitAsmEnd(registers)
    self.asm_emitter.PopIndent(len(self.emitter.indent))


def _GenerateColLoadAggregateStore(emitter, registers, lanes_count,
                                   elements_count, aggregators, input_address,
                                   stride, output):
  """Emit inner loop code for reading N col lanes and interweaving them."""
  emitter.EmitNewline()
  emitter.EmitComment('Load Aggregate Store - column major %dx%d' %
                      (lanes_count, elements_count))

  block = [registers.DoubleRegister() for unused_i in range(lanes_count)]

  if elements_count is not 8:
    _GenerateClear(emitter, 'i8', block)

  block = emitter.EmitLoadColBlock(registers, 8, lanes_count, elements_count,
                                   block, input_address, stride)

  for (aggregator, row) in zip(aggregators, block):
    emitter.EmitVAddw('u8', aggregator, aggregator, row)

  emitter.EmitVStoreAE(8, 8 * lanes_count, block, output,
                       _AlignForLanes(lanes_count))

  registers.FreeRegisters(block)


class ColumnMajorWithSumUInt8x8(common.StreamGenerator):
  """."""

  def __init__(self, emitter, asm_emitter):
    common.StreamGenerator.__init__(self, emitter, 'ColumnMajorWithSum')
    self.asm_emitter = asm_emitter

  def EmitPack(self, in_type, lanes_count, pack_size, leftovers):
    assert pack_size is 8
    assert in_type is 'uint8_t'

    registers = self.asm_emitter.CreateRegisters()

    self.emitter.EmitDeclare('int', 'params_count_copy', 'params.count')
    self.emitter.EmitDeclare('int', 'params_stride_copy', 'params.stride')

    self.asm_emitter.PushIndent(self.emitter.indent)
    self.asm_emitter.EmitAsmBegin()

    count = registers.MapOutputParameter('count', 'params_count_copy')
    input_address = registers.MapOutputParameter('in')
    output_address = registers.MapOutputParameter('out')
    aggregators = [registers.QuadRegister(8) for unused_i in range(lanes_count)]
    stride = registers.MapOutputParameter('stride', 'params_stride_copy')

    self.asm_emitter.EmitColBlockStride(lanes_count, stride, stride)

    _GenerateClear(self.asm_emitter, 'i16', aggregators)

    if leftovers:
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitComment('Reduce count by leftovers.')
      self.asm_emitter.EmitSubs(count, count,
                                self.asm_emitter.ImmediateConstant(leftovers))
      self.asm_emitter.EmitBeqFront(2)

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitNumericalLabel(1)
    self.asm_emitter.EmitSubs(count, count,
                              self.asm_emitter.ImmediateConstant(8))

    _GenerateColLoadAggregateStore(self.asm_emitter, registers, lanes_count, 8,
                                   aggregators, input_address, stride,
                                   output_address)

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitBneBack(1)

    if leftovers:
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitNumericalLabel(2)
      _GenerateColLoadAggregateStore(self.asm_emitter, registers, lanes_count,
                                     leftovers, aggregators, input_address,
                                     stride, output_address)


    _GenerateAggregatorReductionHighRegisters(
        self.asm_emitter, registers, aggregators, output_address)

    self.asm_emitter.EmitAsmEnd(registers)
    self.asm_emitter.PopIndent(len(self.emitter.indent))


def GenerateUInt8x8Streams(cc_emitter, asm_emitter, lanes_count):
  row_major_with_sum = RowMajorWithSumUInt8x8(cc_emitter, asm_emitter)
  column_major_with_sum = ColumnMajorWithSumUInt8x8(cc_emitter, asm_emitter)

  for lanes_count in range(1, 1 + lanes_count):
    for leftovers in range(8):
      row_major_with_sum.SpecializeStream('uint8_t', lanes_count, 8, leftovers)

  for lanes_count in range(1, 1 + lanes_count):
    for leftovers in range(8):
      column_major_with_sum.SpecializeStream('uint8_t', lanes_count, 8,
                                             leftovers)
