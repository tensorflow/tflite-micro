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
"""Generates the arm32 headers used by the gemm/gemv lib."""

import cc_emitter
import common
import neon_emitter
import transform_kernels_common


def Main():
  """."""
  cc = cc_emitter.CCEmitter()
  common.GenerateHeader(cc, 'gemmlowp_meta_transform_kernels_arm_32',
                        'GEMMLOWP_NEON_32')

  cc.EmitNamespaceBegin('gemmlowp')
  cc.EmitNamespaceBegin('meta')
  cc.EmitNewline()

  transform_kernels_common.GenerateKernels(cc,
                                           neon_emitter.NeonEmitter(),
                                           [(16, x) for x in range(16)])

  cc.EmitNamespaceEnd()
  cc.EmitNamespaceEnd()
  cc.EmitNewline()

  common.GenerateFooter(cc, 'Meta gemm for arm32 requires: GEMMLOWP_NEON_32!')


if __name__ == '__main__':
  Main()
