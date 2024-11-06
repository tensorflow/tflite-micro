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

#ifdef ETHOS_U
#include <ethosu_driver.h>
#include <inttypes.h>
#include <pmu_ethosu.h>

#include <algorithm>
#endif

// This is set in micro/tools/make/targets/cortex_m_corstone_300_makefile.inc.
// It is needed for the calls to NVIC_SetVector()/NVIC_EnableIR(),
#include CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE

#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/system_setup.h"

#ifdef ETHOS_U

bool npuPmuCycleCounterIsSet;
uint64_t npuPmuCycleCounter;

extern "C" {
void ethosu_inference_begin(struct ethosu_driver* drv, void* userArg) {
  // Enable PMU
  ETHOSU_PMU_Enable(drv);

  // Enable cycle counter
  ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
  ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);
  ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk);
  ETHOSU_PMU_CYCCNT_Reset(drv);

  // Reset all counters
  ETHOSU_PMU_EVCNTR_ALL_Reset(drv);
}

void ethosu_inference_end(struct ethosu_driver* drv, void* userArg) {
  // Save cycle counter
  npuPmuCycleCounter += ETHOSU_PMU_Get_CCNTR(drv);
  npuPmuCycleCounterIsSet = true;

  // Disable PMU
  ETHOSU_PMU_Disable(drv);
}
}
#endif

namespace tflite {

namespace {
#ifdef ETHOS_U
constexpr uint32_t kClocksPerSecond = 200e6;
#else
constexpr uint32_t kClocksPerSecond = 25e6;
#endif
}  // namespace

uint32_t ticks_per_second() { return kClocksPerSecond; }

uint32_t GetCurrentTimeTicks() {
#if (!defined(TF_LITE_STRIP_ERROR_STRINGS))
#ifdef ETHOS_U
  uint32_t ticks = static_cast<uint32_t>(npuPmuCycleCounter);

  // Note cycle counter will be reset here for next iteration
  if (npuPmuCycleCounterIsSet) {
    npuPmuCycleCounter = 0;
    npuPmuCycleCounterIsSet = false;
  }

  return ticks;
#else

#if defined(ARMCM0)
  return 0;
#else
#ifdef ARMCM55
  return ARM_PMU_Get_CCNTR();
#else
  return DWT->CYCCNT;
#endif
#endif

#endif
#else
  return 0;
#endif
}

#ifdef ETHOS_U
#if defined(ETHOSU_FAST_MEMORY_SIZE) && ETHOSU_FAST_MEMORY_SIZE > 0
__attribute__((aligned(16), section(".bss.ethosu_scratch")))
uint8_t ethosu0_scratch[ETHOSU_FAST_MEMORY_SIZE];
#else
#define ethosu0_scratch 0
#define ETHOSU_FAST_MEMORY_SIZE 0
#endif

struct ethosu_driver ethosu0_driver;

void ethosuIrqHandler0() { ethosu_irq_handler(&ethosu0_driver); }
#endif

extern "C" {
void uart_init(void);
}

void InitializeTarget() {
  uart_init();

#if (!defined(TF_LITE_STRIP_ERROR_STRINGS) && !defined(ARMCM0))
#ifdef ARMCM55
  ARM_PMU_Enable();
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;

  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);

#else
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;

  // Reset and enable DWT cycle counter.
  DWT->CYCCNT = 0;
  DWT->CTRL |= 1UL;

#endif
#endif

#ifdef ETHOS_U
  constexpr int ethosu_base_address = 0x48102000;
  constexpr int ethosu_irq = 56;
  constexpr int ethosu_irq_priority = 5;

  // Initialize Ethos-U NPU driver.
  if (ethosu_init(&ethosu0_driver, reinterpret_cast<void*>(ethosu_base_address),
                  ethosu0_scratch, ETHOSU_FAST_MEMORY_SIZE, 1, 1)) {
    MicroPrintf("Failed to initialize Ethos-U driver");
    return;
  }
  NVIC_SetVector(static_cast<IRQn_Type>(ethosu_irq),
                 reinterpret_cast<uint32_t>(&ethosuIrqHandler0));
  NVIC_SetPriority(static_cast<IRQn_Type>(ethosu_irq), ethosu_irq_priority);
  NVIC_EnableIRQ(static_cast<IRQn_Type>(ethosu_irq));
#endif
}

}  // namespace tflite
