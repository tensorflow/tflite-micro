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

_HEADER_COPYRIGHT = (
    '''// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
''')


def GenerateHeader(cc, header_name, preprocessor_directive):
  cc.EmitCodeNoSemicolon(_HEADER_COPYRIGHT)
  cc.EmitHeaderBegin(header_name)

  cc.EmitPreprocessor1('ifdef', preprocessor_directive)
  cc.EmitNewline()

  cc.EmitInclude('<cassert>')
  cc.EmitInclude('<cstdint>')
  cc.EmitNewline()


def GenerateFooter(cc, message):
  cc.EmitPreprocessor('else')
  cc.EmitPreprocessor1('warning', '"%s"' % message)
  cc.EmitPreprocessor('endif')
  cc.EmitNewline()
  cc.EmitHeaderEnd()


def GenerateDebugLog(cc, message):
  cc.EmitPreprocessor1('ifdef', 'DEBUG')
  cc.EmitPreprocessor1('ifdef', 'DEBUG_METAGEMM_VERBOSE')
  cc.EmitCode('std::cout << __FILE__ << \"(\" << __LINE__ << \") %s\" '
              '<< std::endl << std::flush' % message)
  cc.EmitPreprocessor('endif')
  cc.EmitPreprocessor('endif')


def _TemplateName(base, params):
  return '%s<%s>' % (base, ', '.join(map(str, params)))


class StreamGenerator(object):
  """."""

  def __init__(self, emitter, name):
    self.name = name
    self.emitter = emitter

  def SpecializeStream(self, in_type, lanes_count, pack_size, leftovers):
    if callable(getattr(self, 'EmitPack', None)):
      template_params = [in_type, lanes_count, pack_size, leftovers, self.name]
      self.emitter.EmitMemberFunctionBegin(
          'Stream', [], template_params, 'Pack',
          [['const %s*' % in_type, 'in'], ['const %s&' % self.name, 'params'],
           ['%s*' % in_type, 'out']], 'inline void')
      GenerateDebugLog(self.emitter,
                       '%s::Pack()' % _TemplateName(self.name, template_params))
      self.EmitPack(in_type, lanes_count, pack_size, leftovers)
      self.emitter.EmitFunctionEnd()


class MulKernelGenerator(object):
  """."""

  def __init__(self, emitter, kernel_name, output_stream_name):
    self.kernel_name = kernel_name
    self.output_stream_name = output_stream_name
    self.emitter = emitter

  def SpecializeMulKernel(self, in_type, out_type, kernel_m, kernel_n,
                          pack_size):
    """Generates the kernel wrapped in a MulKernel template specialization."""
    template_params = [
        in_type, out_type, self.kernel_name, self.output_stream_name, kernel_m,
        kernel_n, pack_size
    ]
    self.emitter.EmitMemberFunctionBegin(
        'MulKernel', [], template_params, 'Multiply',
        [['const %s*' % in_type, 'lhs'], ['const %s*' % in_type, 'rhs'], [
            'const FusedKernelParams<%s, %s>&' % (self.kernel_name,
                                                  self.output_stream_name),
            'params'
        ], ['%s*' % out_type, 'result']], 'inline void')
    GenerateDebugLog(self.emitter, '%s::Multiply()' %
                     _TemplateName(self.kernel_name + self.output_stream_name,
                                   template_params))
    self.EmitMultiply(in_type, out_type, kernel_m, kernel_n, pack_size)
    self.emitter.EmitFunctionEnd()


class Transform1DKernelGenerator(object):
  """."""

  def __init__(self, emitter, kernel_name):
    self.kernel_name = kernel_name
    self.emitter = emitter

  def SpecializeTransform1DKernel(self, in_type, out_type, kernel_size,
                                  leftovers):
    """Generates the kernel wrapped in a Transform1DKernel specialization."""
    template_params = [
        in_type, out_type, self.kernel_name, kernel_size, leftovers
    ]
    self.emitter.EmitMemberFunctionBegin(
        'Transform1DKernel', [], template_params, 'Transform',
        [['const %s*' % in_type, 'input'],
         ['const %s&' % self.kernel_name, 'params'],
         ['%s*' % out_type, 'output']], 'inline void')
    GenerateDebugLog(self.emitter, '%s::Transform()' %
                     _TemplateName(self.kernel_name, template_params))
    self.EmitTransform(in_type, out_type, kernel_size, leftovers)
    self.emitter.EmitFunctionEnd()
