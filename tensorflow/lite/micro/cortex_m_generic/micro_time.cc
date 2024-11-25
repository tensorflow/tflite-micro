/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/micro_time.h"

// Set in micro/tools/make/targets/cortex_m_generic_makefile.inc.
// Needed for the DWT and PMU counters.
#ifdef CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE
#include CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE
#endif

namespace tflite {

#if defined(PROJECT_GENERATION)

// Stub functions for the project_generation target since these will be replaced
// by the target-specific implementation in the overall infrastructure that the
// TFLM project generation will be a part of.
uint32_t ticks_per_second() { return 0; }
uint32_t GetCurrentTimeTicks() { return 0; }

#else

uint32_t ticks_per_second() { return 0; }

uint32_t GetCurrentTimeTicks() {
  static bool is_initialized = false;

  if (!is_initialized) {
#if (!defined(TF_LITE_STRIP_ERROR_STRINGS) && !defined(ARMCM0) && \
     !defined(ARMCM0plus))
#ifdef ARM_MODEL_USE_PMU_COUNTERS
    ARM_PMU_Enable();
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;

    ARM_PMU_CYCCNT_Reset();
    ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);

#else
#ifdef ARMCM7
    DWT->LAR = 0xC5ACCE55;
#endif
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;

    // Reset and DWT cycle counter.
    DWT->CYCCNT = 0;
    DWT->CTRL |= 1UL;

#endif
#endif

    is_initialized = true;
  }

#if (!defined(TF_LITE_STRIP_ERROR_STRINGS) && !defined(ARMCM0) && \
     !defined(ARMCM0plus))
#ifdef ARM_MODEL_USE_PMU_COUNTERS
  return ARM_PMU_Get_CCNTR();
#else
  return DWT->CYCCNT;
#endif
#else
  return 0;
#endif
}

#endif  // defined(PROJECT_GENERATION)

}  // namespace tflite
