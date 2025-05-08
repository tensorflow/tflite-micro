/******************************************************************************
 * @file     cmsis_vio.h
 * @brief    CMSIS Virtual I/O header file
 * @version  V1.0.0
 * @date     24. May 2023
 ******************************************************************************/
/*
 * Copyright (c) 2019-2023 Arm Limited. All rights reserved.
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

#ifndef __CMSIS_VIO_H
#define __CMSIS_VIO_H

#include <stdint.h>

/*******************************************************************************
 * Generic I/O mapping recommended for CMSIS-VIO
 * Note: not every I/O must be physically available
 */
 
// vioSetSignal: mask values 
#define vioLED0             (1U << 0)   ///< \ref vioSetSignal \a mask parameter: LED 0 (for 3-color: red)
#define vioLED1             (1U << 1)   ///< \ref vioSetSignal \a mask parameter: LED 1 (for 3-color: green)
#define vioLED2             (1U << 2)   ///< \ref vioSetSignal \a mask parameter: LED 2 (for 3-color: blue)
#define vioLED3             (1U << 3)   ///< \ref vioSetSignal \a mask parameter: LED 3
#define vioLED4             (1U << 4)   ///< \ref vioSetSignal \a mask parameter: LED 4
#define vioLED5             (1U << 5)   ///< \ref vioSetSignal \a mask parameter: LED 5
#define vioLED6             (1U << 6)   ///< \ref vioSetSignal \a mask parameter: LED 6
#define vioLED7             (1U << 7)   ///< \ref vioSetSignal \a mask parameter: LED 7

// vioSetSignal: signal values
#define vioLEDon            (0xFFU)     ///< \ref vioSetSignal \a signal parameter: pattern to turn any LED on
#define vioLEDoff           (0x00U)     ///< \ref vioSetSignal \a signal parameter: pattern to turn any LED off

// vioGetSignal: mask values and return values
#define vioBUTTON0          (1U << 0)   ///< \ref vioGetSignal \a mask parameter: Push button 0
#define vioBUTTON1          (1U << 1)   ///< \ref vioGetSignal \a mask parameter: Push button 1
#define vioBUTTON2          (1U << 2)   ///< \ref vioGetSignal \a mask parameter: Push button 2
#define vioBUTTON3          (1U << 3)   ///< \ref vioGetSignal \a mask parameter: Push button 3
#define vioJOYup            (1U << 4)   ///< \ref vioGetSignal \a mask parameter: Joystick button: up
#define vioJOYdown          (1U << 5)   ///< \ref vioGetSignal \a mask parameter: Joystick button: down
#define vioJOYleft          (1U << 6)   ///< \ref vioGetSignal \a mask parameter: Joystick button: left
#define vioJOYright         (1U << 7)   ///< \ref vioGetSignal \a mask parameter: Joystick button: right
#define vioJOYselect        (1U << 8)   ///< \ref vioGetSignal \a mask parameter: Joystick button: select
#define vioJOYall           (vioJOYup    | \
                             vioJOYdown  | \
                             vioJOYleft  | \
                             vioJOYright | \
                             vioJOYselect)  ///< \ref vioGetSignal \a mask Joystick button: all

// vioSetValue / vioGetValue: id values
#define vioAIN0             (0U)        ///< \ref vioSetValue / \ref vioGetValue \a id parameter: Analog input value 0
#define vioAIN1             (1U)        ///< \ref vioSetValue / \ref vioGetValue \a id parameter: Analog input value 1
#define vioAIN2             (2U)        ///< \ref vioSetValue / \ref vioGetValue \a id parameter: Analog input value 2
#define vioAIN3             (3U)        ///< \ref vioSetValue / \ref vioGetValue \a id parameter: Analog input value 3
#define vioAOUT0            (4U)        ///< \ref vioSetValue / \ref vioGetValue \a id parameter: Analog output value 0

#ifdef  __cplusplus
extern "C"
{
#endif

/// Initialize test input, output.
void vioInit (void);

/// Set signal output.
/// \param[in]     mask         bit mask of signals to set.
/// \param[in]     signal       signal value to set.
void vioSetSignal (uint32_t mask, uint32_t signal);

/// Get signal input.
/// \param[in]     mask         bit mask of signals to read.
/// \return signal value.
uint32_t vioGetSignal (uint32_t mask);

/// Set value output.
/// \param[in]     id           output identifier.
/// \param[in]     value        value to set.
void vioSetValue (uint32_t id, int32_t value);

/// Get value input.
/// \param[in]     id           input identifier.
/// \return  value retrieved from input.
int32_t vioGetValue (uint32_t id);

#ifdef  __cplusplus
}
#endif

#endif /* __CMSIS_VIO_H */
