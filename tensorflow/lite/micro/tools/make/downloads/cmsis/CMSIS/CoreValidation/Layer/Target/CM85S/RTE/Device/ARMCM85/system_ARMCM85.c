/**************************************************************************//**
 * @file     system_ARMCM85.c
 * @brief    CMSIS Device System Source File for ARMCM85 Device
 * @version  V1.0.0
 * @date     30. March 2022
 ******************************************************************************/
/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined (ARMCM85)
  #include "ARMCM85.h"

  #if defined (__ARM_FEATURE_CMSE) &&  (__ARM_FEATURE_CMSE == 3U)
    #include "partition_ARMCM85.h"
  #endif
#else
  #error device not specified!
#endif


/*----------------------------------------------------------------------------
  Define clocks
 *----------------------------------------------------------------------------*/
#define  XTAL            (50000000UL)     /* Oscillator frequency */

#define  SYSTEM_CLOCK    (XTAL / 2U)

/*----------------------------------------------------------------------------
  Exception / Interrupt Vector table
 *----------------------------------------------------------------------------*/
extern const VECTOR_TABLE_Type __VECTOR_TABLE[496];

/*----------------------------------------------------------------------------
  System Core Clock Variable
 *----------------------------------------------------------------------------*/
uint32_t SystemCoreClock = SYSTEM_CLOCK;  /* System Core Clock Frequency */

/*----------------------------------------------------------------------------
  System Core Clock update function
 *----------------------------------------------------------------------------*/
void SystemCoreClockUpdate (void)
{
  SystemCoreClock = SYSTEM_CLOCK;
}

/*----------------------------------------------------------------------------
  System initialization function
 *----------------------------------------------------------------------------*/
void SystemInit (void)
{

#if defined (__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
  SCB->VTOR = (uint32_t)(&__VECTOR_TABLE[0]);
#endif

  /* Set CPDLPSTATE.RLPSTATE to 0
     Set CPDLPSTATE.ELPSTATE to 0, to stop the processor from trying to switch the EPU into retention state.
     Set CPDLPSTATE.CLPSTATE to 0, so PDCORE will not enter low-power state. */
  PWRMODCTL->CPDLPSTATE &= ~(PWRMODCTL_CPDLPSTATE_RLPSTATE_Msk |
                             PWRMODCTL_CPDLPSTATE_ELPSTATE_Msk |
                             PWRMODCTL_CPDLPSTATE_CLPSTATE_Msk  );

#if (defined (__FPU_USED) && (__FPU_USED == 1U)) || \
    (defined (__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE > 0U))
  SCB->CPACR |= ((3U << 10U*2U) |           /* enable CP10 Full Access */
                 (3U << 11U*2U)  );         /* enable CP11 Full Access */

  /* Favor best FP/MVE performance by default, avoid EPU switch-ON delays */
  /* PDEPU ON, Clock OFF */
  PWRMODCTL->CPDLPSTATE |= 0x1 << PWRMODCTL_CPDLPSTATE_ELPSTATE_Pos;
#endif

#ifdef UNALIGNED_SUPPORT_DISABLE
  SCB->CCR |= SCB_CCR_UNALIGN_TRP_Msk;
#endif

  /* Enable Loop and branch info cache */
  SCB->CCR |= SCB_CCR_LOB_Msk;

  /* Enable Branch Prediction */
  SCB->CCR |= SCB_CCR_BP_Msk;

  __DSB();
  __ISB();

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
  TZ_SAU_Setup();
#endif

  SystemCoreClock = SYSTEM_CLOCK;
}
