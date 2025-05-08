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
"""32bit ARM/NEON assembly emitter.

Used by code generators to produce ARM assembly with NEON simd code.
Provides tools for easier register management: named register variable
allocation/deallocation, and offers a more procedural/structured approach
to generating assembly.

TODO: right now neon emitter prints out assembly instructions immediately,
it might be beneficial to keep the whole structure and emit the assembly after
applying some optimizations like: instruction reordering or register reuse.

TODO: NeonRegister object assigns explicit registers at allocation time.
Similarily to emiting code, register mapping and reuse can be performed and
optimized lazily.
"""


class Error(Exception):
  """Module level error."""


class RegisterAllocationError(Error):
  """Cannot alocate registers."""


class LaneError(Error):
  """Wrong lane number."""


class ArgumentError(Error):
  """Wrong argument."""


def _Low(register):
  assert register[0] == 'q'
  num = int(register[1:])
  return 'd%d' % (num * 2)


def _High(register):
  assert register[0] == 'q'
  num = int(register[1:])
  return 'd%d' % (num * 2 + 1)


def _ExpandQuads(registers):
  doubles = []
  for register in registers:
    if register[0] == 'q':
      doubles.append(_Low(register))
      doubles.append(_High(register))
    else:
      doubles.append(register)
  return doubles


def _MakeCompatible(op1, op2, op3):
  if op1[0] == 'd' or op2[0] == 'd' or op3[0] == 'd':
    if op1[0] == 'q':
      op1 = _Low(op1)
    if op2[0] == 'q':
      op2 = _Low(op2)
    if op3[0] == 'q':
      op3 = _Low(op3)
  return (op1, op2, op3)


class _NeonRegisters32Bit(object):
  """Utility that keeps track of used 32bit ARM/NEON registers."""

  def __init__(self):
    self.double = set()
    self.double_ever = set()
    self.general = set()
    self.general_ever = set()
    self.parameters = dict()
    self.output_parameters = dict()

  def MapParameter(self, parameter, parameter_value=None):
    if not parameter_value:
      parameter_value = parameter
    self.parameters[parameter] = (parameter_value, 'r')
    return '%%[%s]' % parameter

  def MapMemoryParameter(self, parameter, parameter_value=None):
    if not parameter_value:
      parameter_value = parameter
    self.parameters[parameter] = (parameter_value, 'm')
    return '%%[%s]' % parameter

  def MapOutputParameter(self, parameter, parameter_value=None):
    if not parameter_value:
      parameter_value = parameter
    self.output_parameters[parameter] = (parameter_value, '+r')
    return '%%[%s]' % parameter

  def DoubleRegister(self, min_val=0):
    for i in range(min_val, 32):
      if i not in self.double:
        self.double.add(i)
        self.double_ever.add(i)
        return 'd%d' % i
    raise RegisterAllocationError('Not enough double registers.')

  def QuadRegister(self, min_val=0):
    for i in range(min_val, 16):
      if ((i * 2) not in self.double) and ((i * 2 + 1) not in self.double):
        self.double.add(i * 2)
        self.double.add(i * 2 + 1)
        self.double_ever.add(i * 2)
        self.double_ever.add(i * 2 + 1)
        return 'q%d' % i
    raise RegisterAllocationError('Not enough quad registers.')

  def GeneralRegister(self):
    for i in range(0, 16):
      if i not in self.general:
        self.general.add(i)
        self.general_ever.add(i)
        return 'r%d' % i
    raise RegisterAllocationError('Not enough general registers.')

  def MappedParameters(self):
    return [(k, v) for (k, v) in self.parameters.items()]

  def MappedOutputParameters(self):
    return [(k, v) for (k, v) in self.output_parameters.items()]

  def Clobbers(self):
    return (['r%d' % i for i in self.general_ever] +
            ['d%d' % i for i in self.DoubleClobbers()])

  def DoubleClobbers(self):
    return sorted(self.double_ever)

  def FreeRegister(self, register):
    assert len(register) > 1
    if register[0] not in ['r', 'd', 'q']:
      return

    num = int(register[1:])

    if register[0] == 'r':
      assert num in self.general
      self.general.remove(num)
    elif register[0] == 'd':
      assert num in self.double
      self.double.remove(num)
    elif register[0] == 'q':
      assert num * 2 in self.double
      assert num * 2 + 1 in self.double
      self.double.remove(num * 2)
      self.double.remove(num * 2 + 1)
    else:
      raise RegisterDeallocationError('Register not allocated: %s' % register)

  def FreeRegisters(self, registers):
    for register in registers:
      self.FreeRegister(register)


class NeonEmitter(object):
  """Emits ARM/NEON assembly opcodes."""

  def __init__(self, debug=False):
    self.ops = {}
    self.indent = ''
    self.debug = debug

  def PushIndent(self, delta='  '):
    self.indent += delta

  def PopIndent(self, delta=2):
    self.indent = self.indent[:-delta]

  def EmitIndented(self, what):
    print self.indent + what

  def PushOp(self, op):
    if op in self.ops.keys():
      self.ops[op] += 1
    else:
      self.ops[op] = 1

  def ClearCounters(self):
    self.ops.clear()

  def EmitNewline(self):
    print ''

  def EmitPreprocessor1(self, op, param):
    print '#%s %s' % (op, param)

  def EmitPreprocessor(self, op):
    print '#%s' % op

  def EmitInclude(self, include):
    self.EmitPreprocessor1('include', include)

  def EmitCall1(self, function, param):
    self.EmitIndented('%s(%s);' % (function, param))

  def EmitAssert(self, assert_expression):
    if self.debug:
      self.EmitCall1('assert', assert_expression)

  def EmitHeaderBegin(self, header_name, includes):
    self.EmitPreprocessor1('ifndef', (header_name + '_H_').upper())
    self.EmitPreprocessor1('define', (header_name + '_H_').upper())
    self.EmitNewline()
    if includes:
      for include in includes:
        self.EmitInclude(include)
      self.EmitNewline()

  def EmitHeaderEnd(self):
    self.EmitPreprocessor('endif')

  def EmitCode(self, code):
    self.EmitIndented('%s;' % code)

  def EmitFunctionBeginA(self, function_name, params, return_type):
    self.EmitIndented('%s %s(%s) {' %
                      (return_type, function_name,
                       ', '.join(['%s %s' % (t, n) for (t, n) in params])))
    self.PushIndent()

  def EmitFunctionEnd(self):
    self.PopIndent()
    self.EmitIndented('}')

  def EmitAsmBegin(self):
    self.EmitIndented('asm volatile(')
    self.PushIndent()

  def EmitAsmMapping(self, elements):
    if elements:
      self.EmitIndented(': ' + ', '.join(
          ['[%s] "%s"(%s)' % (d, v[1], v[0]) for (d, v) in elements]))
    else:
      self.EmitIndented(':')

  def EmitClobbers(self, elements):
    if elements:
      self.EmitIndented(': ' + ', '.join(['"%s"' % c for c in elements]))
    else:
      self.EmitIndented(':')

  def EmitAsmEnd(self, registers):
    self.EmitAsmMapping(registers.MappedOutputParameters())
    self.EmitAsmMapping(registers.MappedParameters())
    self.EmitClobbers(registers.Clobbers() + ['cc', 'memory'])
    self.PopIndent()
    self.EmitIndented(');')

  def EmitComment(self, comment):
    self.EmitIndented('// ' + comment)

  def EmitNumericalLabel(self, label):
    self.EmitIndented('"%d:"' % label)

  def EmitOp1(self, op, param1):
    self.PushOp(op)
    self.EmitIndented('"%s %s\\n"' % (op, param1))

  def EmitOp2(self, op, param1, param2):
    self.PushOp(op)
    self.EmitIndented('"%s %s, %s\\n"' % (op, param1, param2))

  def EmitOp3(self, op, param1, param2, param3):
    self.PushOp(op)
    self.EmitIndented('"%s %s, %s, %s\\n"' % (op, param1, param2, param3))

  def EmitAdd(self, destination, source, param):
    self.EmitOp3('add', destination, source, param)

  def EmitSubs(self, destination, source, param):
    self.EmitOp3('subs', destination, source, param)

  def EmitSub(self, destination, source, param):
    self.EmitOp3('sub', destination, source, param)

  def EmitMul(self, destination, source, param):
    self.EmitOp3('mul', destination, source, param)

  def EmitMov(self, param1, param2):
    self.EmitOp2('mov', param1, param2)

  def EmitBeqBack(self, label):
    self.EmitOp1('beq', '%db' % label)

  def EmitBeqFront(self, label):
    self.EmitOp1('beq', '%df' % label)

  def EmitBgtBack(self, label):
    self.EmitOp1('bgt', '%db' % label)

  def EmitBgtFront(self, label):
    self.EmitOp1('bgt', '%df' % label)

  def EmitBleBack(self, label):
    self.EmitOp1('ble', '%db' % label)

  def EmitBleFront(self, label):
    self.EmitOp1('ble', '%df' % label)

  def EmitBneBack(self, label):
    self.EmitOp1('bne', '%db' % label)

  def EmitBneFront(self, label):
    self.EmitOp1('bne', '%df' % label)

  def EmitVAdd(self, add_type, destination, source_1, source_2):
    destination, source_1, source_2 = _MakeCompatible(destination, source_1,
                                                      source_2)
    self.EmitOp3('vadd.%s' % add_type, destination, source_1, source_2)

  def EmitVAddw(self, add_type, destination, source_1, source_2):
    self.EmitOp3('vaddw.%s' % add_type, destination, source_1, source_2)

  def EmitVSub(self, sub_type, destination, source_1, source_2):
    destination, source_1, source_2 = _MakeCompatible(destination, source_1,
                                                      source_2)
    self.EmitOp3('vsub.%s' % sub_type, destination, source_1, source_2)

  def EmitVCvt(self, cvt_to, cvt_from, destination, source):
    self.EmitOp2('vcvt.%s.%s' % (cvt_to, cvt_from), destination, source)

  def EmitVDup(self, dup_type, destination, source):
    self.EmitOp2('vdup.%s' % dup_type, destination, source)

  def EmitVMax(self, size, destination, source_1, source_2):
    self.EmitOp3('vmax.%s' % size, destination, source_1, source_2)

  def EmitVMin(self, size, destination, source_1, source_2):
    self.EmitOp3('vmin.%s' % size, destination, source_1, source_2)

  def EmitVMov(self, mov_type, destination, source):
    self.EmitOp2('vmov.%s' % mov_type, destination, source)

  def EmitVMovl(self, mov_type, destination, source):
    if source[0] == 'q':
      source = _Low(source)
    self.EmitOp2('vmovl.%s' % mov_type, destination, source)

  def EmitVMovl2(self, mov_type, destination_1, destination_2, source):
    self.EmitVMovl(mov_type, destination_2, _High(source))
    self.EmitVMovl(mov_type, destination_1, _Low(source))

  def EmitVQmovn(self, mov_type, destination, source):
    if destination[0] == 'q':
      destination = _Low(destination)
    self.EmitOp2('vqmovn.%s' % mov_type, destination, source)

  def EmitVQmovn2(self, mov_type, destination, source_1, source_2):
    self.EmitVQmovn(mov_type, _Low(destination), source_1)
    self.EmitVQmovn(mov_type, _High(destination), source_2)

  def EmitVQmovun(self, mov_type, destination, source):
    if destination[0] == 'q':
      destination = _Low(destination)
    self.EmitOp2('vqmovun.%s' % mov_type, destination, source)

  def EmitVQmovun2(self, mov_type, destination, source_1, source_2):
    self.EmitVQmovun(mov_type, _Low(destination), source_1)
    self.EmitVQmovun(mov_type, _High(destination), source_2)

  def EmitVMul(self, mul_type, destination, source_1, source_2):
    destination, source_1, source_2 = _MakeCompatible(destination, source_1,
                                                      source_2)
    self.EmitOp3('vmul.%s' % mul_type, destination, source_1, source_2)

  def EmitVMulScalar(self, mul_type, destination, source_1, source_2):
    self.EmitOp3('vmul.%s' % mul_type, destination, source_1, source_2)

  def EmitVMull(self, mul_type, destination, source_1, source_2):
    self.EmitOp3('vmull.%s' % mul_type, destination, source_1, source_2)

  def EmitVPadd(self, add_type, destination, source_1, source_2):
    self.EmitOp3('vpadd.%s' % add_type, destination, source_1, source_2)

  def EmitVPaddl(self, add_type, destination, source):
    self.EmitOp2('vpaddl.%s' % add_type, destination, source)

  def EmitVPadal(self, add_type, destination, source):
    self.EmitOp2('vpadal.%s' % add_type, destination, source)

  def EmitLdr(self, register, value):
    self.EmitOp2('ldr', register, value)

  def EmitVLoad(self, load_no, load_type, destination, source):
    self.EmitVLoadA(load_no, load_type, [destination], source)

  def EmitVLoadA(self, load_no, load_type, destinations, source):
    self.EmitOp2('vld%d.%d' % (load_no, load_type),
                 '{%s}' % ', '.join(_ExpandQuads(destinations)), source)

  def EmitVLoadAE(self,
                  load_type,
                  elem_count,
                  destinations,
                  source,
                  alignment=None):
    bits_to_load = load_type * elem_count
    destinations = _ExpandQuads(destinations)
    if len(destinations) * 64 < bits_to_load:
      raise ArgumentError('To few destinations: %d to load %d bits.' %
                          (len(destinations), bits_to_load))

    while bits_to_load > 0:
      if bits_to_load >= 256:
        self.EmitVLoadA(1, 32, destinations[:4],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 256
        destinations = destinations[4:]
      elif bits_to_load >= 192:
        self.EmitVLoadA(1, 32, destinations[:3],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 192
        destinations = destinations[3:]
      elif bits_to_load >= 128:
        self.EmitVLoadA(1, 32, destinations[:2],
                        self.DereferenceIncrement(source, alignment))
        bits_to_load -= 128
        destinations = destinations[2:]
      elif bits_to_load >= 64:
        self.EmitVLoad(1, 32, destinations[0],
                       self.DereferenceIncrement(source, alignment))
        bits_to_load -= 64
        destinations = destinations[1:]
      else:
        destination = destinations[0]
        if bits_to_load == 56:
          self.EmitVLoad(1, 32,
                         self.Lane(32, destination, 0),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 16,
                         self.Lane(16, destination, 2),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 8,
                         self.Lane(8, destination, 6),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 48:
          self.EmitVLoad(1, 32,
                         self.Lane(32, destination, 0),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 16,
                         self.Lane(16, destination, 2),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 40:
          self.EmitVLoad(1, 32,
                         self.Lane(32, destination, 0),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 8,
                         self.Lane(8, destination, 4),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 32:
          self.EmitVLoad(1, 32,
                         self.Lane(32, destination, 0),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 24:
          self.EmitVLoad(1, 16,
                         self.Lane(16, destination, 0),
                         self.DereferenceIncrement(source))
          self.EmitVLoad(1, 8,
                         self.Lane(8, destination, 2),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 16:
          self.EmitVLoad(1, 16,
                         self.Lane(16, destination, 0),
                         self.DereferenceIncrement(source))
        elif bits_to_load == 8:
          self.EmitVLoad(1, 8,
                         self.Lane(8, destination, 0),
                         self.DereferenceIncrement(source))
        else:
          raise ArgumentError('Wrong leftover: %d' % bits_to_load)
        return

  def EmitVLoadE(self, load_type, count, destination, source, alignment=None):
    self.EmitVLoadAE(load_type, count, [destination], source, alignment)

  def EmitVLoadAllLanes(self, load_no, load_type, destination, source):
    destinations = []
    if destination[0] == 'q':
      destinations.append(self.AllLanes(_Low(destination)))
      destinations.append(self.AllLanes(_High(destination)))
    else:
      destinations.append(self.AllLanes(destination))
    self.EmitVLoadA(load_no, load_type, destinations, source)

  def EmitVLoadOffset(self, load_no, load_type, destination, source, offset):
    self.EmitVLoadOffsetA(load_no, load_type, [destination], source, offset)

  def EmitVLoadOffsetA(self, load_no, load_type, destinations, source, offset):
    assert len(destinations) <= 4
    self.EmitOp3('vld%d.%d' % (load_no, load_type),
                 '{%s}' % ', '.join(_ExpandQuads(destinations)), source, offset)

  def EmitPld(self, load_address_register):
    self.EmitOp1('pld', '[%s]' % load_address_register)

  def EmitPldw(self, store_address_register):
    self.EmitOp1('pldw', '[%s]' % store_address_register)

  def EmitPldOffset(self, load_address_register, offset):
    self.EmitOp1('pld', '[%s, %s]' % (load_address_register, offset))

  def EmitPldwOffset(self, store_address_register, offset):
    self.EmitOp1('pldw', '[%s, %s]' % (store_address_register, offset))

  def EmitVShl(self, shift_type, destination, source, shift):
    self.EmitOp3('vshl.%s' % shift_type, destination, source, shift)

  def EmitVStore(self, store_no, store_type, source, destination):
    self.EmitVStoreA(store_no, store_type, [source], destination)

  def EmitVStoreA(self, store_no, store_type, sources, destination):
    self.EmitOp2('vst%d.%d' % (store_no, store_type),
                 '{%s}' % ', '.join(_ExpandQuads(sources)), destination)

  def EmitVStoreAE(self,
                   store_type,
                   elem_count,
                   sources,
                   destination,
                   alignment=None):
    bits_to_store = store_type * elem_count
    sources = _ExpandQuads(sources)
    if len(sources) * 64 < bits_to_store:
      raise ArgumentError('To few sources: %d to store %d bits.' %
                          (len(sources), bits_to_store))

    while bits_to_store > 0:
      if bits_to_store >= 256:
        self.EmitVStoreA(1, 32, sources[:4],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 256
        sources = sources[4:]
      elif bits_to_store >= 192:
        self.EmitVStoreA(1, 32, sources[:3],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 192
        sources = sources[3:]
      elif bits_to_store >= 128:
        self.EmitVStoreA(1, 32, sources[:2],
                         self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 128
        sources = sources[2:]
      elif bits_to_store >= 64:
        self.EmitVStore(1, 32, sources[0],
                        self.DereferenceIncrement(destination, alignment))
        bits_to_store -= 64
        sources = sources[1:]
      else:
        source = sources[0]
        if bits_to_store == 56:
          self.EmitVStore(1, 32,
                          self.Lane(32, source, 0),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 16,
                          self.Lane(16, source, 2),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 8,
                          self.Lane(8, source, 6),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 48:
          self.EmitVStore(1, 32,
                          self.Lane(32, source, 0),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 16,
                          self.Lane(16, source, 2),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 40:
          self.EmitVStore(1, 32,
                          self.Lane(32, source, 0),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 8,
                          self.Lane(8, source, 4),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 32:
          self.EmitVStore(1, 32,
                          self.Lane(32, source, 0),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 24:
          self.EmitVStore(1, 16,
                          self.Lane(16, source, 0),
                          self.DereferenceIncrement(destination))
          self.EmitVStore(1, 8,
                          self.Lane(8, source, 2),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 16:
          self.EmitVStore(1, 16,
                          self.Lane(16, source, 0),
                          self.DereferenceIncrement(destination))
        elif bits_to_store == 8:
          self.EmitVStore(1, 8,
                          self.Lane(8, source, 0),
                          self.DereferenceIncrement(destination))
        else:
          raise ArgumentError('Wrong leftover: %d' % bits_to_store)
        return

  def EmitVStoreE(self, store_type, count, source, destination, alignment=None):
    self.EmitVStoreAE(store_type, count, [source], destination, alignment)

  def EmitVStoreOffset(self, store_no, store_type, source, destination, offset):
    self.EmitVStoreOffsetA(store_no, store_type, [source], destination, offset)

  def EmitVStoreOffsetA(self, store_no, store_type, sources, destination,
                        offset):
    self.EmitOp3('vst%d.%d' % (store_no, store_type),
                 '{%s}' % ', '.join(_ExpandQuads(sources)), destination, offset)

  def EmitVStoreOffsetE(self, store_type, count, source, destination, offset):
    """Emit assembly to store a number elements from the source registers."""
    if store_type is not 32:
      raise ArgumentError('Unsupported store_type: %d' % store_type)

    sources = []
    if source[0] == 'q':
      sources.append(_Low(source))
      sources.append(_High(source))
      if count * store_type > 128:
        raise ArgumentError('To many %dbit elements in a q register: %d' %
                            (store_type, count))
    else:
      sources.append(source)
      if count * store_type > 64:
        raise ArgumentError('To many %dbit elements in a d register: %d' %
                            (store_type, count))

    if count == 1:
      self.EmitVStoreOffset(1, store_type,
                            self.Lane(store_type, sources[0], 0),
                            self.Dereference(destination, None), offset)
    elif count == 2:
      self.EmitVStoreOffset(1, store_type, sources[0],
                            self.Dereference(destination, None), offset)
    elif count == 3:
      self.EmitVStore(1, store_type, sources[0],
                      self.DereferenceIncrement(destination, None))
      self.EmitVStoreOffset(1, store_type,
                            self.Lane(store_type, sources[1], 0),
                            self.Dereference(destination, None), offset)
      self.EmitSub(destination, destination, self.ImmediateConstant(8))
    elif count == 4:
      self.EmitVStoreOffsetA(1, store_type, sources,
                             self.Dereference(destination, None), offset)
    else:
      raise ArgumentError('To many elements: %d' % count)

  def EmitVSumReduce(self, reduce_type, elem_count, reduce_count, destinations,
                     sources):
    """Emit assembly for n-fold horizontal sum reduction."""
    if reduce_type is not 'u32':
      raise ArgumentError('Unsupported reduce: %s' % reduce_type)

    sources = _ExpandQuads(sources)

    destinations = _ExpandQuads(destinations)

    if len(destinations) * 2 < elem_count:
      raise ArgumentError('Not enough space in destination: %d vs %d' %
                          (len(destinations) * 2, elem_count))

    if len(sources) * 2 != elem_count * reduce_count:
      raise ArgumentError('Wrong number of sources: %d vs %d' %
                          (len(sources) * 2, elem_count * reduce_count))

    if reduce_count <= 1:
      raise ArgumentError('Unsupported reduce_count: %d' % reduce_count)

    while reduce_count > 1:
      if len(sources) % 2 == 1:
        sources.append(sources[-1])

      if reduce_count == 2:
        for i in range(len(sources) / 2):
          self.EmitVPadd(reduce_type, destinations[i], sources[2 * i],
                         sources[2 * i + 1])
        return
      else:
        sources_2 = []
        for i in range(len(sources) / 2):
          self.EmitVPadd(reduce_type, sources[2 * i], sources[2 * i],
                         sources[2 * i + 1])
          sources_2.append(sources[2 * i])
        reduce_count /= 2
        sources = sources_2

  def EmitVUzp(self, uzp_type, operand_1, operand_2):
    self.EmitOp2('vuzp.%d' % uzp_type, operand_1, operand_2)

  def EmitVTrn(self, trn_type, operand_1, operand_2):
    self.EmitOp2('vtrn.%d' % trn_type, operand_1, operand_2)

  def EmitColBlockStride(self, cols, stride, new_stride):
    assert cols in [1, 2, 3, 4, 5, 6, 7, 8]
    if cols in [5, 6, 7]:
      self.EmitSub(new_stride, stride, self.ImmediateConstant(4))

  def EmitLoadColBlock(self, unused_registers, load_type, cols, elements, block,
                       input_address, stride):
    """Load a block of column major data."""
    assert cols is len(block)
    assert load_type is 8

    input_deref = self.Dereference(input_address, None)
    input_deref_increment = self.DereferenceIncrement(input_address, None)

    if cols is 1:
      for i in range(elements):
        self.EmitVLoadOffset(1, 8,
                             self.Lane(8, block[0], i), input_deref, stride)
      self.EmitPld(input_address)
    elif cols is 2:
      for i in range(elements):
        self.EmitVLoadOffset(1, 16,
                             self.Lane(16, block[i / 4], i % 4), input_deref,
                             stride)
      self.EmitPld(input_address)
      self.EmitVUzp(8, block[0], block[1])
    elif cols is 3:
      for i in range(elements):
        self.EmitVLoadOffsetA(3, 8, [self.Lane(8, row, i) for row in block],
                              input_deref, stride)
    elif cols is 4:
      for i in range(elements):
        self.EmitVLoadOffset(1, 32,
                             self.Lane(32, block[i % 4], i / 4), input_deref,
                             stride)
      self.EmitPld(input_address)
      self.EmitVTrn(16, block[0], block[2])
      self.EmitVTrn(16, block[1], block[3])
      self.EmitVTrn(8, block[0], block[1])
      self.EmitVTrn(8, block[2], block[3])
    elif cols is 5:
      for i in range(elements):
        self.EmitVLoad(1, 32,
                       self.Lane(32, block[i % 4], i / 4),
                       input_deref_increment)
        self.EmitVLoadOffset(1, 8,
                             self.Lane(8, block[4], i), input_deref, stride)
      self.EmitPld(input_address)
      self.EmitVTrn(16, block[0], block[2])
      self.EmitVTrn(16, block[1], block[3])
      self.EmitVTrn(8, block[0], block[1])
      self.EmitVTrn(8, block[2], block[3])
    elif cols is 6:
      for i in range(elements):
        self.EmitVLoad(1, 32,
                       self.Lane(32, block[i % 4], i / 4),
                       input_deref_increment)
        self.EmitVLoadOffset(1, 16,
                             self.Lane(16, block[4 + i / 4], i % 4),
                             input_deref, stride)
      self.EmitPld(input_address)
      self.EmitVTrn(16, block[0], block[2])
      self.EmitVTrn(16, block[1], block[3])
      self.EmitVUzp(8, block[4], block[5])
      self.EmitVTrn(8, block[0], block[1])
      self.EmitVTrn(8, block[2], block[3])
    elif cols is 7:
      for i in range(elements):
        self.EmitVLoad(1, 32,
                       self.Lane(32, block[i % 4], i / 4),
                       input_deref_increment)
        self.EmitVLoadOffsetA(3, 8,
                              [self.Lane(8, row, i) for row in block[4:]],
                              input_deref, stride)
      self.EmitPld(input_address)
      self.EmitVTrn(16, block[0], block[2])
      self.EmitVTrn(16, block[1], block[3])
      self.EmitVTrn(8, block[0], block[1])
      self.EmitVTrn(8, block[2], block[3])
    elif cols is 8:
      for i in range(elements):
        self.EmitVLoadOffset(1, 32, block[i], input_deref, stride)
      self.EmitPld(input_address)
      self.EmitVTrn(8, block[0], block[1])
      self.EmitVTrn(8, block[2], block[3])
      self.EmitVTrn(8, block[4], block[5])
      self.EmitVTrn(8, block[6], block[7])
      self.EmitVTrn(16, block[0], block[2])
      self.EmitVTrn(16, block[1], block[3])
      self.EmitVTrn(16, block[4], block[6])
      self.EmitVTrn(16, block[5], block[7])
      self.EmitVTrn(32, block[0], block[4])
      self.EmitVTrn(32, block[1], block[5])
      self.EmitVTrn(32, block[2], block[6])
      self.EmitVTrn(32, block[3], block[7])
    else:
      assert False
    return block

  def Dereference(self, value, alignment=None):
    if alignment:
      return '[%s:%d]' % (value, alignment)
    else:
      return '[%s]' % value

  def DereferenceIncrement(self, value, alignment=None):
    return '%s!' % self.Dereference(value, alignment)

  def ImmediateConstant(self, value):
    return '#%d' % value

  def AllLanes(self, value):
    return '%s[]' % value

  def Lane(self, bits, value, lane):
    """Get the proper n-bit lane from the given register."""
    registers = []
    if value[0] == 'q':
      registers.append(_Low(value))
      registers.append(_High(value))
    else:
      registers.append(value)

    elems_per_register = 64 / bits
    register = lane / elems_per_register
    lane %= elems_per_register

    return '%s[%d]' % (registers[register], lane)

  def CreateRegisters(self):
    return _NeonRegisters32Bit()
