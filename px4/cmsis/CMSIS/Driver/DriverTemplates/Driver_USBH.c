/*
 * Copyright (c) 2013-2024 Arm Limited. All rights reserved.
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
 
#include "Driver_USBH.h"

/* USB Host Driver */

#define ARM_USBH_DRV_VERSION    ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0) /* driver version */

/* Driver Version */
static const ARM_DRIVER_VERSION usbh_driver_version = { 
    ARM_USBH_API_VERSION,
    ARM_USBH_DRV_VERSION
};

/* Driver Capabilities */
static const ARM_USBH_CAPABILITIES usbh_driver_capabilities = {
    0x0001, /* Root HUB available Ports Mask   */
    0,      /* Automatic SPLIT packet handling */
    0,      /* Signal Connect event */
    0,      /* Signal Disconnect event */
    0,      /* Signal Overcurrent event */
    0       /* Reserved (must be zero) */
};

//
// Functions
//

static ARM_DRIVER_VERSION ARM_USBH_GetVersion(void)
{
  return usbh_driver_version;
}

static ARM_USBH_CAPABILITIES ARM_USBH_GetCapabilities(void)
{
  return usbh_driver_capabilities;
}

static int32_t ARM_USBH_Initialize(ARM_USBH_SignalPortEvent_t cb_port_event,
                                   ARM_USBH_SignalPipeEvent_t cb_pipe_event)
{
}

static int32_t ARM_USBH_Uninitialize(void)
{
}

static int32_t ARM_USBH_PowerControl(ARM_POWER_STATE state)
{
    switch (state)
    {
    case ARM_POWER_OFF:
        break;

    case ARM_POWER_LOW:
        break;

    case ARM_POWER_FULL:
        break;
    }
    return ARM_DRIVER_OK;
}

static int32_t ARM_USBH_PortVbusOnOff(uint8_t port, bool vbus)
{
}

static int32_t ARM_USBH_PortReset(uint8_t port)
{
}

static int32_t ARM_USBH_PortSuspend(uint8_t port)
{
}

static int32_t ARM_USBH_PortResume(uint8_t port)
{
}

static ARM_USBH_PORT_STATE ARM_USBH_PortGetState(uint8_t port)
{
}

static ARM_USBH_PIPE_HANDLE ARM_USBH_PipeCreate(uint8_t  dev_addr,
                                                uint8_t  dev_speed,
                                                uint8_t  hub_addr,
                                                uint8_t  hub_port,
                                                uint8_t  ep_addr,
                                                uint8_t  ep_type,
                                                uint16_t ep_max_packet_size,
                                                uint8_t  ep_interval)
{
}

static int32_t ARM_USBH_PipeModify(ARM_USBH_PIPE_HANDLE pipe_hndl,
                            uint8_t  dev_addr,
                            uint8_t  dev_speed,
                            uint8_t  hub_addr,
                            uint8_t  hub_port,
                            uint16_t ep_max_packet_size)
{
}

static int32_t ARM_USBH_PipeDelete(ARM_USBH_PIPE_HANDLE pipe_hndl)
{
}

static int32_t ARM_USBH_PipeReset(ARM_USBH_PIPE_HANDLE pipe_hndl)
{
}

static int32_t ARM_USBH_PipeTransfer(ARM_USBH_PIPE_HANDLE pipe_hndl,
                              uint32_t packet,
                              uint8_t *data,
                              uint32_t num)
{
}

static uint32_t ARM_USBH_PipeTransferGetResult(ARM_USBH_PIPE_HANDLE pipe_hndl)
{
}

static int32_t ARM_USBH_PipeTransferAbort(ARM_USBH_PIPE_HANDLE pipe_hndl)
{
}

static uint16_t ARM_USBH_GetFrameNumber(void)
{
}

static void ARM_USBH_SignalPortEvent(uint8_t port, uint32_t event)
{
    // function body
}

static void ARM_USBH_SignalPipeEvent(ARM_USBH_PIPE_HANDLE pipe_hndl, uint32_t event)
{
    // function body
}

// End USBH Interface

extern \
ARM_DRIVER_USBH Driver_USBH0;
ARM_DRIVER_USBH Driver_USBH0 = {
  ARM_USBH_GetVersion,
  ARM_USBH_GetCapabilities,
  ARM_USBH_Initialize,
  ARM_USBH_Uninitialize,
  ARM_USBH_PowerControl,
  ARM_USBH_PortVbusOnOff,
  ARM_USBH_PortReset,
  ARM_USBH_PortSuspend,
  ARM_USBH_PortResume,
  ARM_USBH_PortGetState,
  ARM_USBH_PipeCreate,
  ARM_USBH_PipeModify,
  ARM_USBH_PipeDelete,
  ARM_USBH_PipeReset,
  ARM_USBH_PipeTransfer,
  ARM_USBH_PipeTransferGetResult,
  ARM_USBH_PipeTransferAbort,
  ARM_USBH_GetFrameNumber
};

