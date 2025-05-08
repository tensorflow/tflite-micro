## Revision History {#revision_history}

The table on this page provides high-level overview of the CMSIS Base Software release history.

In addition, each component of the CMSIS Base software has its own release history listed on following pages:

 - [**CMSIS-Core Revision History**](../Core/core_revisionHistory.html)
 - [**CMSIS-Driver Revision History**](../Driver/driver_revisionHistory.html)
 - [**CMSIS-RTOS2 Revision History**](../RTOS2/rtos_revisionHistory.html)

Release history of other CMSIS components and tools can be found in their documentation referenced in \ref cmsis_components.

<table class="cmtable" summary="Revision History">
    <tr>
      <th>Version</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>6.1.0</td>
      <td>
       - CMSIS-Core: 6.1.0
         - Added support for Cortex-M52
         - Added deprecated CoreDebug symbols for CMSIS 5 compatibility
         - Added define CMSIS_DISABLE_DEPRECATED to hide deprecated symbols
       - CMSIS-Driver: 2.10.0
         - Updated USB Host API 2.4.0
      </td>
    </tr>
    <tr>
      <td>6.0.0</td>
      <td>
       - CMSIS-Core: 6.0.0
         - Core(M) and Core(A) joined into single Core component
         - Core header files reworked, aligned with TRMs
         - Previously deprecated features removed
         - Dropped support for Arm Compiler 5
       - CMSIS-DSP: moved into separate pack
       - CMSIS-NN: moved into separate pack
       - CMSIS-RTOS: deprecated and removed
         - RTX4 is deprecated and removed
       - CMSIS-RTOS2: 2.3.0
         - OS Tick API moved from Device to CMSIS class
         - Added provisional support for processor affinity in SMP systems
         - RTX5 is moved into separate CMSIS-RTX pack
       - CMSIS-Driver: 2.9.0
         - Updated VIO API 1.0.0
         - Added GPIO Driver API 1.0.0
       - CMSIS-Pack: moved into Open-CMSIS-Pack project
       - CMSIS-SVD: moved into Open-CMSIS-Pack project
       - CMSIS-DAP: moved into separate repository
       - Devices: moved into separate Cortex_DFP pack
       - Utilities: moved into CMSIS-Toolbox project
      </td>
    </tr>
    <tr>
      <td>5.9.0</td>
      <td>
       - CMSIS-Core(M): 5.6.0 (see revision history for details)
         - Arm Cortex-M85 cpu support
         - Arm China STAR-MC1 cpu support
         - Updated system_ARMCM55.c
       - CMSIS-Core(A): 1.2.1 (unchanged)
       - CMSIS-Driver: 2.8.0 (unchanged)
       - CMSIS-DSP: 1.10.0 (see revision history for details)
       - CMSIS-NN: 3.1.0 (see revision history for details)
         - Support for int16 convolution and fully connected for reference implementation
         - Support for DSP extension optimization for int16 convolution and fully connected
         - Support dilation for int8 convolution
         - Support dilation for int8 depthwise convolution
         - Support for int16 depthwise conv for reference implementation including dilation
         - Support for int16 average and max pooling for reference implementation
         - Support for elementwise add and mul int16 scalar version
         - Support for softmax int16 scalar version
         - Support for SVDF with 8 bit state tensor
       - CMSIS-RTOS2: 2.1.3 (unchanged)
          - RTX 5.5.4 (see revision history for details)
       - CMSIS-Pack: deprecated (moved to Open-CMSIS-Pack)
       - CMSIS-Build: deprecated (moved to CMSIS-Toolbox in Open-CMSIS-Pack)
       - CMSIS-SVD: 1.3.9 (see revision history for details)
       - CMSIS-DAP: 2.1.1 (see revision history for details)
         - Allow default clock frequency to use fast clock mode
       - CMSIS-Zone: 1.0.0 (unchanged)
       - Devices
         - Support for Cortex-M85
       - Utilities
          - SVDConv 3.3.42
          - PackChk 1.3.95
      </td>
    </tr>
    <tr>
      <td>5.8.0</td>
      <td>
        - CMSIS-Build 0.10.0 (beta)
          - Enhancements (see revision history for details)
        - CMSIS-Core (Cortex-M) 5.5.0
          - Updated GCC LinkerDescription, GCC Assembler startup
          - Added ARMv8-M Stack Sealing (to linker, startup) for toolchain ARM, GCC
          - Changed C-Startup to default Startup.
        - CMSIS-Core (Cortex-A) 1.2.1
        - CMSIS-Driver 2.8.0 (unchanged)
        - CMSIS-DSP 1.9.0
          - Purged pre-built libs from Git
        - CMSIS-NN 3.0.0
          - Major interface change for functions compatible with TensorFlow Lite for Microcontroller
          - Added optimization for SVDF kernel
          - Improved MVE performance for fully Connected and max pool operator
          - NULL bias support for fully connected operator in non-MVE case(Can affect performance)
          - Expanded existing unit test suite along with support for FVP
        - CMSIS-RTOS 2.1.3 (unchanged)
          - RTX 5.5.3 (see revision history for details)
        - CMSIS-Pack 1.7.2
          - Support for Microchip XC32 compiler
          - Support for Custom Datapath Extension
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 2.0.0 (unchanged)
        - CMSIS-Zone 1.0.0 (unchanged)
        - Devices
        - Utilities
          - SVDConv 3.3.35
          - PackChk 1.3.89
      </td>
    </tr>
    <tr>
      <td>5.7.0</td>
      <td>
        - CMSIS-Build 0.9.0 (beta)
          - Draft for CMSIS Project description (CPRJ)
        - CMSIS-Core (Cortex-M) 5.4.0
          - Cortex-M55 cpu support
          - Enhanced MVE support for Armv8.1-MML
          - Fixed device config define checks.
          - L1 Cache functions for Armv7-M and later
        - CMSIS-Core (Cortex-A) 1.2.0
          - Fixed GIC_SetPendingIRQ to use GICD_SGIR
          - Added missing DSP intrinsics
          - Reworked assembly intrinsics: volatile, barriers and clobber
        - CMSIS-Driver 2.8.0
          - Added VIO API 0.1.0 (preview)
        - CMSIS-DSP 1.8.0
          - Added new functions and function groups
          - Added MVE support
        - CMSIS-NN 1.3.0
          - Added MVE support
          - Further optimizations for kernels using DSP extension
        - CMSIS-RTOS 2.1.3 (unchanged)
          - RTX 5.5.2 (see revision history for details)
        - CMSIS-Pack 1.6.3
          - deprecating all types specific to cpdsc format. Cpdsc is replaced by Cprj with dedicated schema.
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 2.0.0 (unchanged)
        - CMSIS-Zone 1.0.0
        - Devices
          - ARMCM55 device
          - ARMv81MML startup code recognizing __MVE_USED macro
          - Refactored vector table references for all Cortex-M devices
          - Reworked ARMCM* C-StartUp files.
          - Include L1 Cache functions in ARMv8MML/ARMv81MML devices
        - Utilities
          Attention: Linux binaries moved to Linux64 folder!
          - SVDConv 3.3.35
          - PackChk 1.3.89
      </td>
    </tr>
    <tr>
      <td>5.6.0</td>
      <td>
        - CMSIS-Core (Cortex-M) 5.3.0
          - Added provisions for compiler-independent C startup code.
        - CMSIS-Core (Cortex-A) 1.1.4
          - Fixed __FPU_Enable.
        - CMSIS-Driver 2.7.1
          - Finalized WiFi Interface API 1.0.0
        - CMSIS-DSP 1.7.0 (see revision history for details)
          - New Neon versions of f32 functions
          - Compilation flags for FFTs
        - CMSIS-NN 1.2.0 (unchanged)
        - CMSIS-RTOS1 1.03 (unchanged)
          - RTX 4.82.0 (see revision history for details)
        - CMSIS-RTOS 2.1.3 (unchanged)
          - RTX 5.5.1 (see revision history for details)
        - CMSIS-Pack 1.6.0 (unchanged)
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 2.0.0 (unchanged)
        - CMSIS-Zone 0.12.0 (preview)
          - Completely reworked
        - Devices
          - Generalized C startup code for all Cortex-M family devices.
          - Updated Cortex-A memory regions and system configuration files.
        - Utilities
          - SVDConv 3.3.27
          - PackChk 1.3.82 (unchanged)
      </td>
    </tr>
    <tr>
      <td>5.5.1</td>
      <td>
        - CMSIS-Core (Cortex-M) 5.2.1
          - Fixed compilation issue in cmsis_armclang_ltm.h
        - CMSIS-Core (Cortex-A) 1.1.3 (unchanged)
        - CMSIS-Driver 2.7.0 (unchanged)
        - CMSIS-DSP 1.6.0 (unchanged)
        - CMSIS-NN 1.1.0 (unchanged)
        - CMSIS-RTOS 2.1.3 (unchanged)
          - RTX 5.5.0 (unchanged)
        - CMSIS-Pack 1.6.0 (unchanged)
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 2.0.0 (unchanged)
        - CMSIS-Zone 0.9.0 (unchanged)
      </td>
    </tr>
    <tr>
      <td>5.5.0</td>
      <td>
        - CMSIS-Core (Cortex-M) 5.2.0
          - Reworked Stack/Heap configuration for ARM startup files.
          - Added Cortex-M35P device support.
          - Added generic Armv8.1-M Mainline device support.
        - CMSIS-Core (Cortex-A) 1.1.3 Minor fixes.
        - CMSIS-DSP 1.6.0
          - reworked DSP library source files
            - added macro ARM_MATH_LOOPUNROLL
            - removed macro UNALIGNED_SUPPORT_DISABLE
            - added const-correctness
            - replaced SIMD pointer construct with memcopy solution
            - replaced macro combination `CMSIS_INLINE __STATIC_INLINE` with `__STATIC_FORCEINLINE`
          - reworked DSP library documentation
          - Changed DSP folder structure
            - moved DSP libraries to ./DSP/Lib
          - moved DSP libraries to folder ./DSP/Lib
          - ARM DSP Libraries are built with ARMCLANG
          - Added DSP Libraries Source variant
        - CMSIS-NN 1.1.0 (unchanged)
        - CMSIS-Driver 2.7.0
          - Added WiFi Interface API 1.0.0-beta
          - Added custom driver selection to simplify implementation of new CMSIS-Driver
        - CMSIS-RTOS 2.1.3
          - RTX 5.5.0 (see revision history)
        - CMSIS-Pack 1.6.0
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 2.0.0 (unchanged)
        - CMSIS-Zone 0.9.0 (Preview)
        - Devices
          - Added Cortex-M35P and ARMv81MML device templates.
          - Fixed C-Startup Code for GCC (aligned with other compilers)
            - Moved call to SystemInit before memory initialization.
        - Utilities
          - SVDConv 3.3.25
          - PackChk 1.3.82
      </td>
    </tr>
    <tr>
      <td>5.4.0</td>
      <td>
        - CMSIS-Core (Cortex-M) 5.1.2 Minor fixes and slight enhancements, e.g. beta for Cortex-M1.
        - CMSIS-Core (Cortex-A) 1.1.2 Minor fixes.
        - CMSIS-Driver 2.6.0 (unchanged)
        - CMSIS-DSP 1.5.2 (unchanged)
        - CMSIS-NN 1.1.0 Added new math function (see revision history)
        - CMSIS-RTOS 2.1.3 Relaxed interrupt usage.
          - RTX 5.4.0 (see revision history)
        - CMSIS-Pack 1.5.0 (unchanged)
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 2.0.0 (unchanged)
        - CMSIS-Zone 0.0.1 (unchanged)
      </td>
    </tr>
    <tr>
      <td>5.3.0</td>
      <td>
        - CMSIS-Core (Cortex-M) 5.1.1
        - CMSIS-Core (Cortex-A) 1.1.1
        - CMSIS-Driver 2.6.0 (unchanged)
        - CMSIS-DSP 1.5.2 (unchanged)
        - CMSIS-NN 1.0.0 Initial contribution of Neural Network Library.
        - CMSIS-RTOS 2.1.2 (unchanged)
        - CMSIS-Pack 1.5.0 (unchanged)
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 2.0.0 Communication via WinUSB to achieve high-speed transfer rates.
        - CMSIS-Zone 0.0.1 (unchanged)
      </td>
    </tr>
    <tr>
      <td>5.2.0</td>
      <td>
        - CMSIS-Core (Cortex-M) 5.1.0 MPU functions for ARMv8-M, cmsis_iccarm.h replacing cmsis_iar.h
        - CMSIS-Core (Cortex-A) 1.1.0 cmsis_iccarm.h, additional physical timer access functions
        - CMSIS-Driver 2.6.0 Enhanced CAN and NAND driver interface.
        - CMSIS-DSP 1.5.2 Fixed diagnostics and moved SSAT/USST intrinsics to CMSIS-Core.
        - CMSIS-RTOS 2.1.2 Relaxed some ISR-callable restrictions.
        - CMSIS-Pack 1.5.0 (unchanged)
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 1.2.0 (unchanged)
        - CMSIS-Zone 0.0.1 (unchanged)
      </td>
    </tr>
    <tr>
      <td>5.1.1</td>
      <td>
        - CMSIS-Core (Cortex-M) (unchanged)
        - CMSIS-Core (Cortex-A) (unchanged)
        - CMSIS-Driver 2.05 (unchanged)
        - CMSIS-DSP 1.5.2 (unchanged)
        - CMSIS-RTOS 2.1.1 Fixed RTX5 pre-built libraries for Cortex-M.
        - CMSIS-Pack 1.5.0 (unchanged)
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 1.1.0 (unchanged)
        - CMSIS-Zone 0.0.1 (unchanged)
      </td>
    </tr>
    <tr>
      <td>5.1.0</td>
      <td>
        - CMSIS-Core (Cortex-M) 5.0.2 several minor corrections and enhancements
        - CMSIS-Core (Cortex-A) 1.0.0 implements a basic run-time system for Cortex-A5/A7/A9
        - CMSIS-Driver 2.05 status typedef made volatile
        - CMSIS-DSP 1.5.2 fixed GNU Compiler specific diagnostics
        - CMSIS-RTOS 2.1.1 added support for Cortex-A5/A7/A9 to RTX5
        - CMSIS-Pack 1.5.0 added SDF format specification
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 1.1.0 (unchanged)
        - CMSIS-Zone 0.0.1 (Preview) format to describe system resources and tool for partitioning of resources
      </td>
    </tr>
    <tr>
      <td>5.0.1</td>
      <td>
        - CMSIS-Core 5.0.1 added __PACKED_STRUCT macro and uVisor support
        - CMSIS-Driver 2.05 updated all typedefs related to status now being volatile.
        - CMSIS-DSP 1.5.1 added ARMv8M DSP libraries
        - CMSIS-RTOS 2.1.0 added support for critical and uncritical sections
        - CMSIS-Pack 1.4.8 add Pack Index File specification
        - CMSIS-SVD 1.3.3 (unchanged)
        - CMSIS-DAP 1.1.0 (unchanged)
      </td>
    </tr>
	<tr>
      <td>5.0.0</td>
      <td>
        Added support for: <a href="http://www.arm.com/products/processors/instruction-set-architectures/armv8-m-architecture.php" target="_blank"><b>ARMv8-M architecture</b></a> including TrustZone for ARMv8-M and Cortex-M23, Cortex-M33 processors
        - CMSIS-Core (Cortex-M) 5.0.0 added support for ARMv8-M and Cortex-M23, Cortex-M33 processors
        - CMSIS-Driver 2.04.0 (unchanged)
        - CMSIS-DSP 1.4.9 minor corrections and performance improvements
        - CMSIS-RTOS 2.0.0 new API with RTX 5.0.0 reference implementation and corrections in RTX 4.8.2
        - CMSIS-Pack 1.4.4 introducing CPDSC project description
        - CMSIS-SVD 1.3.3 several enhancements and rework of documentation
        - CMSIS-DAP 1.1.0 (unchanged)
      </td>
    </tr>
    <tr>
      <td>4.5.0</td>
      <td>
        Maintenance release that is fixing defects. See component's revision history for more details.
        See component's revision history for more details.
        - CMSIS-Core (Cortex-M) 4.30.0
        - CMSIS-DAP 1.1.0 (unchanged)
        - CMSIS-Driver 2.04.0
        - CMSIS-DSP 1.4.7
        - CMSIS-Pack 1.4.1
        - CMSIS-RTOS RTX 4.80.0
        - CMSIS-SVD 1.3.1
      </td>
    </tr>
    <tr>
      <td>4.4.0</td>
      <td>
        Feature release adding CMSIS-DAP (see extended End User Licence Agreement) and CMSIS-Driver for CAN.
        See component's revision history for more details.
        - CMSIS-Core (Cortex-M) 4.20.0
        - CMSIS-DAP 1.1.0
        - CMSIS-Driver 2.03.0
        - CMSIS-DSP 1.4.5  (unchanged)
        - CMSIS-RTOS RTX 4.79.0
        - CMSIS-Pack 1.4.0
        - CMSIS-SVD 1.3.0
      </td>
    </tr>
    <tr>
      <td>4.3.0</td>
      <td>
        Maintenance release adding SAI CMSIS-Driver and fixing defects. See component's revision history for more details.
        - CMSIS-Core (Cortex-M) 4.10.0
        - CMSIS-Driver 2.02.0
        - CMSIS-DSP 1.4.5
        - CMSIS-RTOS RTX 4.78.0
        - CMSIS-Pack 1.3.3
        - CMSIS-SVD (unchanged)
      </td>
    </tr>
    <tr>
      <td>4.2</td>
      <td>Introducing processor support for Cortex-M7.
      </td>
    </tr>
    <tr>
      <td>4.1</td>
      <td>Enhancements in CMSIS-Pack and CMSIS-Driver.\n
      Added: PackChk validation utility\n
      Removed support for GNU: Sourcery G++ Lite Edition for ARM</td>
    </tr>
    <tr>
      <td>4.0</td>
      <td>First release in CMSIS-Pack format.\n Added specifications for CMSIS-Pack, CMSIS-Driver</td>
    </tr>
    <tr>
      <td>3.30</td>
      <td>Maintenance release with enhancements in each component</td>
    </tr>
    <tr>
      <td>3.20</td>
      <td>Maintenance release with enhancements in each component</td>
    </tr>
    <tr>
      <td>3.01</td>
      <td>Added support for Cortex-M0+ processors</td>
    </tr>
    <tr>
      <td>3.00</td>
      <td>Added support for SC000 and SC300 processors\n
      Added support for GNU GCC Compiler\n
      Added CMSIS-RTOS API</td>
    </tr>
    <tr>
      <td>2.10</td>
      <td>Added CMSIS-DSP Library</td>
    </tr>
    <tr>
      <td>2.0</td>
      <td>Added support for Cortex-M4 processor</td>
    </tr>
    <tr>
      <td>1.30</td>
      <td>Reworked CMSIS startup concept</td>
    </tr>
    <tr>
      <td>1.01</td>
      <td>Added support for Cortex-M0 processor</td>
    </tr>
    <tr>
      <td>1.00</td>
      <td>Initial release of CMSIS-Core (Cortex-M) for Cortex-M3 processor</td>
    </tr>
</table>
