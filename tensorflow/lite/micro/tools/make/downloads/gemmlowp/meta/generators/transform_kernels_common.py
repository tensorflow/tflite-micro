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


def _DuplicateGeneralRegister(size, emitter, registers, value, min_register):
  register = registers.QuadRegister(min_register)
  emitter.EmitVDup(size, register, value)
  return register


def _DuplicateGeneralMemoryRegister(size, emitter, registers, value,
                                    min_register):
  register = registers.QuadRegister(min_register)
  general = registers.GeneralRegister()
  emitter.EmitLdr(general, value)
  emitter.EmitVDup(size, register, general)
  registers.FreeRegister(general)
  return register


class MinMaxTransformation(object):
  """."""

  def Check(self, in_type, out_type, kernel_size, leftovers):
    assert in_type is 'uint8_t'
    assert out_type is 'uint8_t'
    assert kernel_size is 16
    assert leftovers < 16

  def Prepare(self, emitter, registers, unused_kernel_size):
    emitter.EmitNewline()
    emitter.EmitComment('MinMax::Prepare')

    self.min = _DuplicateGeneralRegister(8, emitter, registers,
                                         registers.MapParameter('min',
                                                                'params.min'),
                                         4)
    self.max = _DuplicateGeneralRegister(8, emitter, registers,
                                         registers.MapParameter('max',
                                                                'params.max'),
                                         4)

  def Transform(self, emitter, registers, input_address, elements,
                output_address):
    """Generate the MinMax transform inner loop code."""
    emitter.EmitNewline()
    emitter.EmitComment('MinMax::Transform')
    register_count = (elements + 15) / 16
    load = [registers.QuadRegister() for unused_i in range(register_count)]
    emitter.EmitVLoadAE(8, elements, load, input_address, None)
    emitter.EmitPldOffset(input_address, emitter.ImmediateConstant(16))

    for register in load:
      emitter.EmitVMax('u8', register, register, self.min)

    for register in load:
      emitter.EmitVMin('u8', register, register, self.max)

    emitter.EmitNewline()
    emitter.EmitVStoreAE(8, elements, load, output_address, None)
    emitter.EmitPld(output_address)
    registers.FreeRegisters(load)


class DequantizeTransformation(object):
  """."""

  def Check(self, in_type, out_type, kernel_size, leftovers):
    assert in_type is 'uint8_t'
    assert out_type is 'float'
    assert kernel_size is 16
    assert leftovers < 16

  def Prepare(self, emitter, registers, unused_kernel_size):
    """Duplicate quantization offsets to vector registers."""
    emitter.EmitNewline()
    emitter.EmitComment('Dequantize::Prepare')

    self.range_min = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_min', 'params.range_min'), 4)
    self.range_offset = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_offset', 'params.range_offset'), 4)
    self.range_scale = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_scale', 'params.range_scale'), 4)

  def Transform(self, emitter, registers, input_address, elements,
                output_address):
    """Emit the dequantization inner loop."""
    emitter.EmitNewline()
    emitter.EmitComment('Dequantize::Transform')
    register_count = (elements + 3) / 4
    load = [registers.QuadRegister() for unused_i in range(register_count)]
    emitter.EmitVLoadAE(8, elements, load, input_address, None)
    emitter.EmitPldOffset(input_address, emitter.ImmediateConstant(32))

    if len(load) is 1:
      emitter.EmitVMovl('u8', load[0], load[0])
      emitter.EmitVMovl('s16', load[0], load[0])
    elif len(load) is 2:
      emitter.EmitVMovl('u8', load[0], load[0])
      emitter.EmitVMovl2('s16', load[0], load[1], load[0])
    elif len(load) is 3:
      emitter.EmitVMovl2('u8', load[0], load[1], load[0])
      emitter.EmitVMovl('s16', load[2], load[1])
      emitter.EmitVMovl2('s16', load[0], load[1], load[0])
    elif len(load) is 4:
      emitter.EmitVMovl2('u8', load[0], load[1], load[0])
      emitter.EmitVMovl2('s16', load[2], load[3], load[1])
      emitter.EmitVMovl2('s16', load[0], load[1], load[0])
    else:
      assert False

    for register in load:
      emitter.EmitVCvt('f32', 's32', register, register)

    for register in load:
      emitter.EmitVSub('f32', register, register, self.range_offset)

    for register in load:
      emitter.EmitVMul('f32', register, register, self.range_scale)

    for register in load:
      emitter.EmitVAdd('f32', register, register, self.range_min)

    emitter.EmitNewline()
    emitter.EmitVStoreAE(32, elements, load, output_address, None)
    emitter.EmitPld(output_address)
    registers.FreeRegisters(load)


class QuantizeTransformation(object):
  """."""

  def Check(self, in_type, out_type, kernel_size, leftovers):
    assert in_type is 'float'
    assert out_type is 'uint8_t'
    assert kernel_size is 16
    assert leftovers < 16

  def Prepare(self, emitter, registers, unused_kernel_size):
    """Duplicate quantization offsets to vector registers."""
    emitter.EmitNewline()
    emitter.EmitComment('Quantize::Prepare')

    self.range_min = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_min', 'params.range_min'), 4)
    self.range_offset = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_offset', 'params.range_offset'), 4)
    self.range_scale = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('range_scale', 'params.range_scale'), 4)

  def Transform(self, emitter, registers, input_address, elements,
                output_address):
    """Emit quantization inner loop code."""
    emitter.EmitNewline()
    emitter.EmitComment('Quantize::Transform')
    register_count = (elements + 3) / 4
    load = [registers.QuadRegister() for unused_i in range(register_count)]
    emitter.EmitVLoadAE(32, elements, load, input_address, None)
    emitter.EmitPldOffset(input_address, emitter.ImmediateConstant(64))

    for register in load:
      emitter.EmitVSub('f32', register, register, self.range_min)

    for register in load:
      emitter.EmitVMul('f32', register, register, self.range_scale)

    for register in load:
      emitter.EmitVAdd('f32', register, register, self.range_offset)

    for register in load:
      emitter.EmitVCvt('s32', 'f32', register, register)

    if len(load) is 1:
      emitter.EmitVQmovn('s32', load[0], load[0])
      emitter.EmitVQmovun('s16', load[0], load[0])
    elif len(load) is 2:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovun('s16', load[0], load[0])
    elif len(load) is 3:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovn('s32', load[2], load[2])
      emitter.EmitVQmovun2('s16', load[0], load[0], load[2])
    elif len(load) is 4:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovn2('s32', load[2], load[2], load[3])
      emitter.EmitVQmovun2('s16', load[0], load[0], load[2])
    else:
      assert False

    emitter.EmitNewline()
    emitter.EmitVStoreAE(8, elements, load, output_address, None)
    emitter.EmitPld(output_address)
    registers.FreeRegisters(load)


class RequantizeTransformation(object):
  """."""

  def Check(self, in_type, out_type, kernel_size, leftovers):
    assert in_type is 'int32_t'
    assert out_type is 'uint8_t'
    assert kernel_size is 16
    assert leftovers < 16

  def Prepare(self, emitter, registers, unused_kernel_size):
    """Duplicate quantization parameters to vector registers."""
    emitter.EmitNewline()
    emitter.EmitComment('Requantize::Prepare')

    self.range_min_delta = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('input_range_min', 'params.input_range_min'), 4)
    self.output_range_min = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('output_range_min', 'params.output_range_min'),
        4)
    self.input_range_offset = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('input_range_offset',
                               'params.input_range_offset'), 4)
    self.input_range_scale = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('input_range_scale', 'params.input_range_scale'),
        4)
    self.one_over_output_range_scale = _DuplicateGeneralRegister(
        32, emitter, registers,
        registers.MapParameter('one_over_output_range_scale',
                               'params.one_over_output_range_scale'), 4)
    emitter.EmitVSub('f32', self.range_min_delta, self.range_min_delta,
                     self.output_range_min)

  def Transform(self, emitter, registers, input_address, elements,
                output_address):
    """Emit requantization inner loop code."""
    emitter.EmitNewline()
    emitter.EmitComment('Requantize::Transform')
    register_count = (elements + 3) / 4
    load = [registers.QuadRegister() for unused_i in range(register_count)]
    emitter.EmitVLoadAE(32, elements, load, input_address, None)
    emitter.EmitPldOffset(input_address, emitter.ImmediateConstant(64))

    for register in load:
      emitter.EmitVCvt('f32', 's32', register, register)

    for register in load:
      emitter.EmitVSub('f32', register, register, self.input_range_offset)

    for register in load:
      emitter.EmitVMul('f32', register, register, self.input_range_scale)

    for register in load:
      emitter.EmitVAdd('f32', register, register, self.range_min_delta)

    for register in load:
      emitter.EmitVMul('f32', register, register,
                       self.one_over_output_range_scale)

    for register in load:
      emitter.EmitVCvt('s32', 'f32', register, register)

    if len(load) is 1:
      emitter.EmitVQmovn('s32', load[0], load[0])
      emitter.EmitVQmovun('s16', load[0], load[0])
    elif len(load) is 2:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovun('s16', load[0], load[0])
    elif len(load) is 3:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovn('s32', load[2], load[2])
      emitter.EmitVQmovun2('s16', load[0], load[0], load[2])
    elif len(load) is 4:
      emitter.EmitVQmovn2('s32', load[0], load[0], load[1])
      emitter.EmitVQmovn2('s32', load[2], load[2], load[3])
      emitter.EmitVQmovun2('s16', load[0], load[0], load[2])
    else:
      assert False

    emitter.EmitNewline()
    emitter.EmitVStoreAE(8, elements, load, output_address, None)
    emitter.EmitPld(output_address)
    registers.FreeRegisters(load)


class BaseTransform(common.Transform1DKernelGenerator):
  """."""

  def __init__(self, cc_emitter, kernel_name, asm_emitter, transformation):
    common.Transform1DKernelGenerator.__init__(self, cc_emitter, kernel_name)
    self.asm_emitter = asm_emitter
    self.transformation = transformation

  def EmitTransform(self, in_type, out_type, kernel_size, leftovers):
    """."""
    self.transformation.Check(in_type, out_type, kernel_size, leftovers)

    registers = self.asm_emitter.CreateRegisters()

    self.emitter.EmitDeclare('int', 'params_count_copy', 'params.count')

    self.asm_emitter.PushIndent(self.emitter.indent)
    self.asm_emitter.EmitAsmBegin()

    count = registers.MapOutputParameter('count', 'params_count_copy')
    input_address = registers.MapOutputParameter('input')
    output_address = registers.MapOutputParameter('output')

    self.transformation.Prepare(self.asm_emitter, registers, kernel_size)

    if leftovers:
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitComment('Reduce count by leftovers.')
      self.asm_emitter.EmitSubs(count, count,
                                self.asm_emitter.ImmediateConstant(leftovers))
      self.asm_emitter.EmitBeqFront(2)

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitNumericalLabel(1)
    self.asm_emitter.EmitSubs(count, count,
                              self.asm_emitter.ImmediateConstant(kernel_size))

    self.transformation.Transform(self.asm_emitter, registers, input_address,
                                  kernel_size, output_address)

    self.asm_emitter.EmitNewline()
    self.asm_emitter.EmitBneBack(1)

    if leftovers:
      self.asm_emitter.EmitNumericalLabel(2)
      self.asm_emitter.EmitNewline()
      self.asm_emitter.EmitComment('Handle leftovers.')
      self.transformation.Transform(self.asm_emitter, registers, input_address,
                                    leftovers, output_address)

    self.asm_emitter.EmitAsmEnd(registers)
    self.asm_emitter.PopIndent(len(self.emitter.indent))


class Requantize(BaseTransform):
  """."""

  def __init__(self, cc_emitter, asm_emitter):
    BaseTransform.__init__(self, cc_emitter, 'Requantize', asm_emitter,
                           RequantizeTransformation())


class Quantize(BaseTransform):
  """."""

  def __init__(self, cc_emitter, asm_emitter):
    BaseTransform.__init__(self, cc_emitter, 'Quantize', asm_emitter,
                           QuantizeTransformation())


class Dequantize(BaseTransform):
  """."""

  def __init__(self, cc_emitter, asm_emitter):
    BaseTransform.__init__(self, cc_emitter, 'Dequantize', asm_emitter,
                           DequantizeTransformation())


class MinMax(BaseTransform):
  """."""

  def __init__(self, numerical_type, cc_emitter, asm_emitter):
    BaseTransform.__init__(self, cc_emitter, 'MinMax<%s>' % numerical_type,
                           asm_emitter, MinMaxTransformation())


class BiasAdd(common.Transform1DKernelGenerator):
  """."""

  def __init__(self, bias_type, cc_emitter, asm_emitter):
    common.Transform1DKernelGenerator.__init__(self, cc_emitter,
                                               'BiasAdd<%s>' % bias_type)
    self.asm_emitter = asm_emitter

  def EmitTransform(self, in_type, out_type, kernel_size, leftovers):
    """."""
    assert in_type is 'uint8_t'
    assert out_type is 'int32_t'
    assert kernel_size is 16
    assert leftovers < 16

    registers = self.asm_emitter.CreateRegisters()

    self.emitter.EmitDeclare('int', 'params_rows_copy', 'params.rows')

    self.asm_emitter.PushIndent(self.emitter.indent)
    self.asm_emitter.EmitAsmBegin()

    self._Prepare(self.asm_emitter, registers)

    rows = registers.MapParameter('rows', 'params_rows_copy')

    self.asm_emitter.EmitNumericalLabel(1)

    self._ProcessRow(self.asm_emitter, registers, kernel_size, leftovers)

    self.asm_emitter.EmitSubs(rows, rows, self.asm_emitter.ImmediateConstant(1))
    self.asm_emitter.EmitBneBack(1)

    self.asm_emitter.EmitAsmEnd(registers)
    self.asm_emitter.PopIndent(len(self.emitter.indent))

  def _Prepare(self, emitter, registers):
    self.input_range_min = _DuplicateGeneralMemoryRegister(
        32, emitter, registers,
        registers.MapMemoryParameter('input_range_min',
                                     'params.input_range_min'), 8)
    self.input_range_scale = _DuplicateGeneralMemoryRegister(
        32, emitter, registers,
        registers.MapMemoryParameter('input_range_scale',
                                     'params.input_range_scale'), 8)
    self.bias_range_min = _DuplicateGeneralMemoryRegister(
        32, emitter, registers,
        registers.MapMemoryParameter('bias_range_min', 'params.bias_range_min'),
        8)
    self.bias_range_scale = _DuplicateGeneralMemoryRegister(
        32, emitter, registers,
        registers.MapMemoryParameter('bias_range_scale',
                                     'params.bias_range_scale'), 8)
    self.output_range_min = _DuplicateGeneralMemoryRegister(
        32, emitter, registers,
        registers.MapMemoryParameter('output_range_min',
                                     'params.output_range_min'), 8)
    self.one_over_output_range_scale = _DuplicateGeneralMemoryRegister(
        32, emitter, registers,
        registers.MapMemoryParameter('one_over_output_range_scale',
                                     'params.one_over_output_range_scale'), 8)
    self.output_range_offset = _DuplicateGeneralMemoryRegister(
        32, emitter, registers,
        registers.MapMemoryParameter('output_range_offset',
                                     'params.output_range_offset'), 8)

  def _ProcessRow(self, emitter, registers, kernel_size, leftovers):
    const_count = registers.MapParameter('count', 'params.count')
    const_bias = registers.MapParameter('bias', 'params.bias')

    count = registers.GeneralRegister()
    bias = registers.GeneralRegister()

    input_address = registers.MapOutputParameter('input')
    output_address = registers.MapOutputParameter('output')

    emitter.EmitMov(count, const_count)
    emitter.EmitMov(bias, const_bias)

    if leftovers:
      emitter.EmitSubs(count, count, emitter.ImmediateConstant(leftovers))
      emitter.EmitBeqFront(3)

    emitter.EmitNumericalLabel(2)
    emitter.EmitSubs(count, count, emitter.ImmediateConstant(kernel_size))

    self._BiasAdd(emitter, registers, kernel_size, input_address, bias,
                  output_address)

    emitter.EmitBneBack(2)

    if leftovers:
      emitter.EmitNumericalLabel(3)
      self._BiasAdd(emitter, registers, leftovers, input_address, bias,
                    output_address)

  def _BiasAdd(self, emitter, registers, elements, input_address, bias,
               output_address):
    emitter.EmitNewline()
    emitter.EmitComment('BiasAdd::Transform')
    register_count = (elements + 3) / 4

    load_input = [
        registers.QuadRegister() for unused_i in range(register_count)
    ]
    load_bias = [registers.QuadRegister() for unused_i in range(register_count)]

    emitter.EmitVLoadAE(8, elements, load_input, input_address, None)
    emitter.EmitVLoadAE(8, elements, load_bias, bias, None)
    emitter.EmitPldOffset(input_address, emitter.ImmediateConstant(32))

    if len(load_input) is 1:
      emitter.EmitVMovl('u8', load_input[0], load_input[0])
      emitter.EmitVMovl('u8', load_bias[0], load_bias[0])
      emitter.EmitVMovl('s16', load_input[0], load_input[0])
      emitter.EmitVMovl('s16', load_bias[0], load_bias[0])
    elif len(load_input) is 2:
      emitter.EmitVMovl('u8', load_input[0], load_input[0])
      emitter.EmitVMovl('u8', load_bias[0], load_bias[0])
      emitter.EmitVMovl2('s16', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMovl2('s16', load_bias[0], load_bias[1], load_bias[0])
    elif len(load_input) is 3:
      emitter.EmitVMovl2('u8', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMovl2('u8', load_bias[0], load_bias[1], load_bias[0])
      emitter.EmitVMovl('s16', load_input[2], load_input[1])
      emitter.EmitVMovl('s16', load_bias[2], load_bias[1])
      emitter.EmitVMovl2('s16', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMovl2('s16', load_bias[0], load_bias[1], load_bias[0])
    elif len(load_input) is 4:
      emitter.EmitVMovl2('u8', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMovl2('u8', load_bias[0], load_bias[1], load_bias[0])
      emitter.EmitVMovl2('s16', load_input[2], load_input[3], load_input[1])
      emitter.EmitVMovl2('s16', load_bias[2], load_bias[3], load_bias[1])
      emitter.EmitVMovl2('s16', load_input[0], load_input[1], load_input[0])
      emitter.EmitVMovl2('s16', load_bias[0], load_bias[1], load_bias[0])
    else:
      assert False

    for register in load_input + load_bias:
      emitter.EmitVCvt('f32', 's32', register, register)

    for register in load_input:
      emitter.EmitVMul('f32', register, register, self.input_range_scale)

    for register in load_bias:
      emitter.EmitVMul('f32', register, register, self.bias_range_scale)

    for register in load_input:
      emitter.EmitVAdd('f32', register, register, self.input_range_min)

    for register in load_bias:
      emitter.EmitVAdd('f32', register, register, self.bias_range_min)

    for (register_1, register_2) in zip(load_input, load_bias):
      emitter.EmitVAdd('f32', register_1, register_1, register_2)

    for register in load_input:
      emitter.EmitVSub('f32', register, register, self.output_range_min)

    for register in load_input:
      emitter.EmitVMul('f32', register, register,
                       self.one_over_output_range_scale)

    for register in load_input:
      emitter.EmitVAdd('f32', register, register, self.output_range_offset)

    for register in load_input:
      emitter.EmitVCvt('s32', 'f32', register, register)

    emitter.EmitNewline()
    emitter.EmitVStoreAE(32, elements, load_input, output_address, None)
    emitter.EmitPld(output_address)
    registers.FreeRegisters(load_input + load_bias)


def GenerateKernels(cc_emitter, asm_emitter, shapes):
  """Generate the quantization/dequantization/requantization kernels."""
  requantize = Requantize(cc_emitter, asm_emitter)
  quantize = Quantize(cc_emitter, asm_emitter)
  dequantize = Dequantize(cc_emitter, asm_emitter)
  minmax = MinMax('uint8_t', cc_emitter, asm_emitter)
  biasadd = BiasAdd('uint8_t', cc_emitter, asm_emitter)

  for shape in shapes:
    requantize.SpecializeTransform1DKernel('int32_t', 'uint8_t', shape[0],
                                           shape[1])

  for shape in shapes:
    quantize.SpecializeTransform1DKernel('float', 'uint8_t', shape[0], shape[1])

  for shape in shapes:
    dequantize.SpecializeTransform1DKernel('uint8_t', 'float', shape[0],
                                           shape[1])

  for shape in shapes:
    minmax.SpecializeTransform1DKernel('uint8_t', 'uint8_t', shape[0], shape[1])

  for shape in shapes:
    biasadd.SpecializeTransform1DKernel('uint8_t', 'int32_t', shape[0],
                                        shape[1])
