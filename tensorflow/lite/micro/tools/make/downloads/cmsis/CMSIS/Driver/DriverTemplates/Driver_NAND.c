/*
 * Copyright (c) 2013-2020 Arm Limited. All rights reserved.
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
 
#include "Driver_NAND.h"

#define ARM_NAND_DRV_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0)   /* driver version */

/* Driver Version */
static const ARM_DRIVER_VERSION DriverVersion = {
  ARM_NAND_API_VERSION,
  ARM_NAND_DRV_VERSION
};

/* Driver Capabilities */
static const ARM_NAND_CAPABILITIES DriverCapabilities = {
  0,                 /* Signal Device Ready event (R/Bn rising edge)                        */
  0,                 /* Supports re-entrant operation (SendCommand/Address, Read/WriteData) */
  0,                 /* Supports Sequence operation (ExecuteSequence, AbortSequence)        */
  0,                 /* Supports VCC Power Supply Control                                   */
  0,                 /* Supports 1.8 VCC Power Supply                                       */
  0,                 /* Supports VCCQ I/O Power Supply Control                              */
  0,                 /* Supports 1.8 VCCQ I/O Power Supply                                  */
  0,                 /* Supports VPP High Voltage Power Supply Control                      */
  0,                 /* Supports WPn (Write Protect) Control                                */
  0,                 /* Number of CEn (Chip Enable) lines: ce_lines + 1                     */
  0,                 /* Supports manual CEn (Chip Enable) Control                           */
  0,                 /* Supports R/Bn (Ready/Busy) Monitoring                               */
  0,                 /* Supports 16-bit data                                                */
  0,                 /* Supports NV-DDR  Data Interface (ONFI)                              */
  0,                 /* Supports NV-DDR2 Data Interface (ONFI)                              */
  0,                 /* Fastest (highest) SDR     Timing Mode supported (ONFI)              */
  0,                 /* Fastest (highest) NV_DDR  Timing Mode supported (ONFI)              */
  0,                 /* Fastest (highest) NV_DDR2 Timing Mode supported (ONFI)              */
  0,                 /* Supports Driver Strength 2.0x = 18 Ohms                             */
  0,                 /* Supports Driver Strength 1.4x = 25 Ohms                             */
  0,                 /* Supports Driver Strength 0.7x = 50 Ohms                             */
#if (ARM_NAND_API_VERSION > 0x201U)
  0                  /* Reserved (must be zero)                                             */
#endif
};

/* Exported functions */

static ARM_DRIVER_VERSION ARM_NAND_GetVersion (void) {
  return DriverVersion;
}

static ARM_NAND_CAPABILITIES ARM_NAND_GetCapabilities (void) {
  return DriverCapabilities;
}

static int32_t ARM_NAND_Initialize (ARM_NAND_SignalEvent_t cb_event) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_Uninitialize (void) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_PowerControl (ARM_POWER_STATE state) {

  switch ((int32_t)state) {
    case ARM_POWER_OFF:
      return ARM_DRIVER_ERROR_UNSUPPORTED;

    case ARM_POWER_LOW:
      return ARM_DRIVER_ERROR_UNSUPPORTED;

    case ARM_POWER_FULL:
      return ARM_DRIVER_ERROR_UNSUPPORTED;

    default:
      return ARM_DRIVER_ERROR_UNSUPPORTED;
  }
  return ARM_DRIVER_OK;
}

static int32_t ARM_NAND_DevicePower (uint32_t voltage) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_WriteProtect (uint32_t dev_num, bool enable) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_ChipEnable (uint32_t dev_num, bool enable) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_GetDeviceBusy (uint32_t dev_num) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_SendCommand (uint32_t dev_num, uint8_t cmd) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_SendAddress (uint32_t dev_num, uint8_t addr) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_ReadData (uint32_t dev_num, void *data, uint32_t cnt, uint32_t mode) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_WriteData (uint32_t dev_num, const void *data, uint32_t cnt, uint32_t mode) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_ExecuteSequence (uint32_t dev_num, uint32_t code, uint32_t cmd,
                                         uint32_t addr_col, uint32_t addr_row,
                                         void *data, uint32_t data_cnt,
                                         uint8_t *status, uint32_t *count) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_AbortSequence (uint32_t dev_num) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_NAND_Control (uint32_t dev_num, uint32_t control, uint32_t arg) {

  switch (control) {
    case ARM_NAND_BUS_MODE:
      return ARM_DRIVER_ERROR_UNSUPPORTED;

    case ARM_NAND_BUS_DATA_WIDTH:
      return ARM_DRIVER_ERROR_UNSUPPORTED; 

    case ARM_NAND_DEVICE_READY_EVENT:
      return ARM_DRIVER_ERROR_UNSUPPORTED;

    default:
      return ARM_DRIVER_ERROR_UNSUPPORTED;
  }

  return ARM_DRIVER_ERROR;
}

static ARM_NAND_STATUS ARM_NAND_GetStatus (uint32_t dev_num) {
  ARM_NAND_STATUS stat;

  stat.busy      = 0U;
  stat.ecc_error = 0U;

  return stat;
}

static int32_t ARM_NAND_InquireECC (int32_t index, ARM_NAND_ECC_INFO *info) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

/* NAND Driver Control Block */
extern \
ARM_DRIVER_NAND Driver_NAND0;
ARM_DRIVER_NAND Driver_NAND0 = {
  ARM_NAND_GetVersion,
  ARM_NAND_GetCapabilities,
  ARM_NAND_Initialize,
  ARM_NAND_Uninitialize,
  ARM_NAND_PowerControl,
  ARM_NAND_DevicePower,
  ARM_NAND_WriteProtect,
  ARM_NAND_ChipEnable,
  ARM_NAND_GetDeviceBusy,
  ARM_NAND_SendCommand,
  ARM_NAND_SendAddress,
  ARM_NAND_ReadData,
  ARM_NAND_WriteData,
  ARM_NAND_ExecuteSequence,
  ARM_NAND_AbortSequence,
  ARM_NAND_Control,
  ARM_NAND_GetStatus,
  ARM_NAND_InquireECC
};
