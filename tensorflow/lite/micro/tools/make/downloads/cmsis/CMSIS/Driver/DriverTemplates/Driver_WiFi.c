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

#include "Driver_WiFi.h"

#define ARM_WIFI_DRV_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0)        // Driver version

// Driver Version
static const ARM_DRIVER_VERSION driver_version = {
  ARM_WIFI_API_VERSION,
  ARM_WIFI_DRV_VERSION 
};

// Driver Capabilities
static const ARM_WIFI_CAPABILITIES driver_capabilities = { 
  0U,                                   // Station supported
  0U,                                   // Access Point supported
  0U,                                   // Concurrent Station and Access Point not supported
  0U,                                   // WiFi Protected Setup (WPS) for Station supported
  0U,                                   // WiFi Protected Setup (WPS) for Access Point not supported
  0U,                                   // Access Point: event generated on Station connect
  0U,                                   // Access Point: event not generated on Station disconnect
  0U,                                   // Event not generated on Ethernet frame reception in bypass mode
  0U,                                   // Bypass or pass-through mode (Ethernet interface) not supported
  0U,                                   // IP (UDP/TCP) (Socket interface) supported
  0U,                                   // IPv6 (Socket interface) not supported
  0U,                                   // Ping (ICMP) supported
  0U                                    // Reserved (must be zero)
};
static ARM_DRIVER_VERSION ARM_WiFi_GetVersion (void) {
  return driver_version;
}

static ARM_WIFI_CAPABILITIES ARM_WiFi_GetCapabilities (void) { 
  return driver_capabilities;
}

static int32_t ARM_WiFi_Initialize (ARM_WIFI_SignalEvent_t cb_event) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_Uninitialize (void) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_PowerControl (ARM_POWER_STATE state) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_GetModuleInfo (char *module_info, uint32_t max_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SetOption (uint32_t interface, uint32_t option, const void *data, uint32_t len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_GetOption (uint32_t interface, uint32_t option, void *data, uint32_t *len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}
static int32_t ARM_WiFi_Scan (ARM_WIFI_SCAN_INFO_t scan_info[], uint32_t max_num) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_Activate (uint32_t interface, const ARM_WIFI_CONFIG_t *config) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_Deactivate (uint32_t interface) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static uint32_t ARM_WiFi_IsConnected (void) {
  return 0U;
}

static int32_t ARM_WiFi_GetNetInfo (ARM_WIFI_NET_INFO_t *net_info) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_BypassControl (uint32_t interface, uint32_t mode) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_EthSendFrame (uint32_t interface, const uint8_t *frame, uint32_t len){
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_EthReadFrame (uint32_t interface, uint8_t *frame, uint32_t len){
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static uint32_t ARM_WiFi_EthGetRxFrameSize (uint32_t interface){
  return 0U;
}

static int32_t ARM_WiFi_SocketCreate (int32_t af, int32_t type, int32_t protocol) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketBind (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketListen (int32_t socket, int32_t backlog) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketAccept (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketConnect (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketRecv (int32_t socket, void *buf, uint32_t len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketRecvFrom (int32_t socket, void *buf, uint32_t len, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketSend (int32_t socket, const void *buf, uint32_t len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketSendTo (int32_t socket, const void *buf, uint32_t len, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketGetSockName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketGetPeerName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketGetOpt (int32_t socket, int32_t opt_id, void *opt_val, uint32_t *opt_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketSetOpt (int32_t socket, int32_t opt_id, const void *opt_val, uint32_t opt_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketClose (int32_t socket) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_SocketGetHostByName (const char *name, int32_t af, uint8_t *ip, uint32_t *ip_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t ARM_WiFi_Ping (const uint8_t *ip, uint32_t ip_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

/* WiFi Driver Control Block */
extern \
ARM_DRIVER_WIFI Driver_WiFi0;
ARM_DRIVER_WIFI Driver_WiFi0 = { 
  ARM_WiFi_GetVersion,
  ARM_WiFi_GetCapabilities,
  ARM_WiFi_Initialize,
  ARM_WiFi_Uninitialize,
  ARM_WiFi_PowerControl,
  ARM_WiFi_GetModuleInfo,
  ARM_WiFi_SetOption,
  ARM_WiFi_GetOption,
  ARM_WiFi_Scan,
  ARM_WiFi_Activate,
  ARM_WiFi_Deactivate,
  ARM_WiFi_IsConnected,
  ARM_WiFi_GetNetInfo,
  ARM_WiFi_BypassControl,
  ARM_WiFi_EthSendFrame,
  ARM_WiFi_EthReadFrame,
  ARM_WiFi_EthGetRxFrameSize,
  ARM_WiFi_SocketCreate,
  ARM_WiFi_SocketBind,
  ARM_WiFi_SocketListen,
  ARM_WiFi_SocketAccept,
  ARM_WiFi_SocketConnect,
  ARM_WiFi_SocketRecv,
  ARM_WiFi_SocketRecvFrom,
  ARM_WiFi_SocketSend,
  ARM_WiFi_SocketSendTo,
  ARM_WiFi_SocketGetSockName,
  ARM_WiFi_SocketGetPeerName,
  ARM_WiFi_SocketGetOpt,
  ARM_WiFi_SocketSetOpt,
  ARM_WiFi_SocketClose,
  ARM_WiFi_SocketGetHostByName,
  ARM_WiFi_Ping
};
