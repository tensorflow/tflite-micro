# Driver Validation {#driverValidation}

Developers can use **CMSIS-Driver Validation** framework to verify that an implementation of a peripheral driver is compliant with the corresponding CMSIS-Driver Specification. Verified drivers can then be reliably used with middleware components and user applications that rely on CMSIS-Driver APIs.

The CMSIS-Driver Validation is maintained in a separate public [GitHub repository](https://github.com/ARM-software/CMSIS-Driver_Validation), and is also released as a [CMSIS Software Pack](https://www.keil.arm.com/packs/) named **ARM::CMSIS-Driver_Validation**.

This page gives an overview about driver validation. Refer to [CMSIS-Driver Validation Guide](https://arm-software.github.io/CMSIS-Driver_Validation/latest/index.html)) for full documentation.

The CMSIS-Driver Validation Suite performs the following tests:

 - Generic Validation of API function calls
 - Validation of Configuration Parameters
 - Validation of Communication with loopback tests
 - Validation of Communication Parameters such as baudrate
 - Validation of Event functions

The following CMSIS-Drivers can be tested with the current release:

 - \ref can_interface_gr : with loop back test of communication.
 - \ref eth_interface_gr : MAC and PHY with loop back test of communication.
 - \ref i2c_interface_gr : only API and setup; does not test data transfer.
 - \ref mci_interface_gr : only API and setup; does not test data transfer.
 - \ref spi_interface_gr : with loop back test of communication.
 - \ref usart_interface_gr : with loop back test of communication.
 - \ref usbd_interface_gr : only API and setup; does not test data transfer.
 - \ref usbh_interface_gr : only API and setup; does not test data transfer.
 - \ref wifi_interface_gr : extensive tests for WiFi Driver.

 ## Sample Test Output {#test_output}

The Driver Validation output can be printed to a console or saved in an XML file, via standard output (usually ITM).

```
CMSIS-Driver USART Test Report   Dec  6 2019   11:44:30

TEST 01: USART_GetCapabilities            PASSED
TEST 02: USART_Initialization             PASSED
TEST 03: USART_PowerControl
  DV_USART.c (301): [WARNING] Low power is not supported
                                          PASSED
TEST 04: USART_Config_PolarityPhase       PASSED
TEST 05: USART_Config_DataBits
  DV_USART.c (387): [WARNING] Data Bits = 9 are not supported
                                          PASSED
TEST 06: USART_Config_StopBits
  DV_USART.c (425): [WARNING] Stop Bits = 1.5 are not supported
  DV_USART.c (429): [WARNING] Stop Bits = 0.5 are not supported
                                          PASSED
TEST 07: USART_Config_Parity              PASSED
TEST 08: USART_Config_Baudrate            PASSED
TEST 09: USART_Config_CommonParams        PASSED
TEST 10: USART_Send                       PASSED
TEST 11: USART_AsynchronousReceive        PASSED
TEST 12: USART_Loopback_CheckBaudrate     PASSED
TEST 13: USART_Loopback_Transfer          PASSED
TEST 14: USART_CheckInvalidInit           PASSED

Test Summary: 14 Tests, 14 Passed, 0 Failed.
Test Result: PASSED
```

## Setup for Loop Back Communication {#loop_back_setup}

To perform loop back communication tests it is required to connect the input and the output of the peripherals as shown in this table:

Peripheral       | Loop Back Configuration
:----------------|:----------------------------
Ethernet         | Connect TX+ (Pin 1) with RX+ (Pin 3), TX- (Pin 2) with RX- (Pin 6)
SPI              | Connect MISO to MOSI
USART            | Connect TX with RX

The following picture shows the necessary external loop back connections for the Keil MCBSTM32F400 evaluation board:

 - SPI: PB14 (SPI2_MISO) and PB15 (SPI2_MOSI)
 - USART: PB6 (USART1_TX) and PB7 (USART1_RX)
 - Ethernet: Pin 1 (TX+) and Pin 3 (RX+), Pin 2 (TX-) and Pin 6 (RX-)

![Connections for Loop Back Communication Tests on Keil MCBSTM32F400](./images/image006.png)
