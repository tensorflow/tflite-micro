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
 
#include "Driver_USBD.h"

#define ARM_USBD_DRV_VERSION    ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0) /* driver version */

/* Driver Version */
static const ARM_DRIVER_VERSION usbd_driver_version = { 
    ARM_USBD_API_VERSION,
    ARM_USBD_DRV_VERSION
};

/* Driver Capabilities */
static const ARM_USBD_CAPABILITIES usbd_driver_capabilities = {
    0, /* vbus_detection */
    0, /* event_vbus_on */
    0, /* event_vbus_off */
    0  /* reserved */
};

//
// Functions
//

static ARM_DRIVER_VERSION ARM_USBD_GetVersion(void)
{
  return usbd_driver_version;
}

static ARM_USBD_CAPABILITIES ARM_USBD_GetCapabilities(void)
{
  return usbd_driver_capabilities;
}

static int32_t ARM_USBD_Initialize(ARM_USBD_SignalDeviceEvent_t cb_device_event,
                                   ARM_USBD_SignalEndpointEvent_t cb_endpoint_event)
{
}

static int32_t ARM_USBD_Uninitialize(void)
{
}

static int32_t ARM_USBD_PowerControl(ARM_POWER_STATE state)
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

static int32_t ARM_USBD_DeviceConnect(void)
{
}

static int32_t ARM_USBD_DeviceDisconnect(void)
{
}

static ARM_USBD_STATE ARM_USBD_DeviceGetState(void)
{
}

static int32_t ARM_USBD_DeviceRemoteWakeup(void)
{
}

static int32_t ARM_USBD_DeviceSetAddress(uint8_t dev_addr)
{
}

static int32_t ARM_USBD_ReadSetupPacket(uint8_t *setup)
{
}

static int32_t ARM_USBD_EndpointConfigure(uint8_t  ep_addr,
                                          uint8_t  ep_type,
                                          uint16_t ep_max_packet_size)
{
}

static int32_t ARM_USBD_EndpointUnconfigure(uint8_t ep_addr)
{
}

static int32_t ARM_USBD_EndpointStall(uint8_t ep_addr, bool stall)
{
}

static int32_t ARM_USBD_EndpointTransfer(uint8_t ep_addr, uint8_t *data, uint32_t num)
{
}

static uint32_t ARM_USBD_EndpointTransferGetResult(uint8_t ep_addr)
{
}

static int32_t ARM_USBD_EndpointTransferAbort(uint8_t ep_addr)
{
}

static uint16_t ARM_USBD_GetFrameNumber(void)
{
}

static void ARM_USBD_SignalDeviceEvent(uint32_t event)
{
    // function body
}

static void ARM_USBD_SignalEndpointEvent(uint8_t ep_addr, uint32_t ep_event)
{
    // function body
}

// End USBD Interface

extern \
ARM_DRIVER_USBD Driver_USBD0;
ARM_DRIVER_USBD Driver_USBD0 =
{
    ARM_USBD_GetVersion,
    ARM_USBD_GetCapabilities,
    ARM_USBD_Initialize,
    ARM_USBD_Uninitialize,
    ARM_USBD_PowerControl,
    ARM_USBD_DeviceConnect,
    ARM_USBD_DeviceDisconnect,
    ARM_USBD_DeviceGetState,
    ARM_USBD_DeviceRemoteWakeup,
    ARM_USBD_DeviceSetAddress,
    ARM_USBD_ReadSetupPacket,
    ARM_USBD_EndpointConfigure,
    ARM_USBD_EndpointUnconfigure,
    ARM_USBD_EndpointStall,
    ARM_USBD_EndpointTransfer,
    ARM_USBD_EndpointTransferGetResult,
    ARM_USBD_EndpointTransferAbort,
    ARM_USBD_GetFrameNumber
};
