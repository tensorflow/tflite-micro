/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "ethosu_driver.h"
#endif

// This is set in micro/tools/make/targets/cortex_m_corstone_300_makefile.inc.
// It is needed for the calls to NVIC_SetVector()/NVIC_EnableIR() and for the
// DWT and PMU counters.
#include CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace tflite {

namespace {
constexpr uint32_t kClocksPerSecond = 25e6;
}  // namespace

uint32_t ticks_per_second() { return kClocksPerSecond; }

uint32_t GetCurrentTimeTicks() {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
#ifdef ARMCM55
  return ARM_PMU_Get_CCNTR();
#else
  return DWT->CYCCNT;
#endif
#else
  return 0;
#endif  // TF_LITE_STRIP_ERROR_STRINGS
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

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#ifdef ARMCM55
  ARM_PMU_Enable();
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;

  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);

#else
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

  // Reset and enable DWT cycle counter.
  DWT->CYCCNT = 0;
  DWT->CTRL |= 1UL;

#endif
#endif  // TF_LITE_STRIP_ERROR_STRINGS

#ifdef ETHOS_U
  constexpr int ethosu_base_address = 0x48102000;
  constexpr int ethosu_irq = 56;

  // Initialize Ethos-U NPU driver.
  if (ethosu_init(&ethosu0_driver, reinterpret_cast<void*>(ethosu_base_address),
                  ethosu0_scratch, ETHOSU_FAST_MEMORY_SIZE, 1, 1)) {
    MicroPrintf("Failed to initialize Ethos-U driver");
  }
  NVIC_SetVector(static_cast<IRQn_Type>(ethosu_irq),
                 (uint32_t)&ethosuIrqHandler0);
  NVIC_EnableIRQ(static_cast<IRQn_Type>(ethosu_irq));
#endif
}

}  // namespace tflite
