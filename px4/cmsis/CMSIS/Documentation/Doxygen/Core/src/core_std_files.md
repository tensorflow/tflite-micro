# CMSIS-Core Files {#cmsis_core_files}

CMSIS-Core files can be differentiated in two main groups:

 1. \subpage cmsis_standard_files are provided by Arm for supported CPU cores as part of the CMSIS-Core software component. These files typically do not require any modifications and are expected to be included via CMSIS-Core device files.
 2. \subpage cmsis_device_files are specified in CMSIS-Core methodology, and are typically provided by CPU device vendors to correctly cover their specific functionalities. Some of them may expect additional application-specific changes.

The detailed file structure of the CMSIS-Core files is shown in the following picture.

![CMSIS-Core File Structure](./images/CMSIS_CORE_Files.png)

\subpage cmsis_files_dfps explains how \ref cmsis_core_files can be distributed in CMSIS-Pack format.

## CMSIS-Core Standard Files {#cmsis_standard_files}

The CMSIS-Core Standard file implement all attributes specific to Arm processor cores and generally do not need any modifications.

The files are provided by Arm as CMSIS-Core software component that is part of the [CMSIS Software pack](../General/cmsis_pack.html). The CMSIS-Core standard files can be split into following categories explained below:

 - \ref cmsis_processor_files
 - \ref cmsis_compiler_files
 - \ref cmsis_feature_files

### CMSIS-Core Processor Files {#cmsis_processor_files}

The CMSIS-Core processor files define the core peripherals and provide helper functions for their access.

The files have naming convention `core_<cpu>.h`, with one file available for each supported processor `<cpu>` as listed in the table below.

Header File            | Target Processor Core
:----------------------|:-------------------------------
📂 CMSIS/Core/Include  | CMSIS-Core include folder ([See on GitHub](https://github.com/ARM-software/CMSIS_6/tree/main/CMSIS/Core/Include/))
 ┣ 📄 core_cm0.h       | Cortex-M0 processor
 ┣ 📄 core_cm0plus.h   | Cortex-M0+ processor
 ┣ 📄 core_cm1.h       | Cortex-M1 processor
 ┣ 📄 core_cm3.h       | Cortex-M3 processor
 ┣ 📄 core_cm4.h       | Cortex-M4 processor
 ┣ 📄 core_cm7.h       | Cortex-M7 processor
 ┣ 📄 core_cm23.h      | Cortex-M23 processor
 ┣ 📄 core_cm33.h      | Cortex-M33 processor
 ┣ 📄 core_cm35p.h     | Cortex-M35P processor
 ┣ 📄 core_cm52.h      | Cortex-M52 processor
 ┣ 📄 core_cm55.h      | Cortex-M55 processor
 ┣ 📄 core_cm85.h      | Cortex-M85 processor
 ┣ 📄 core_starmc1.h   | STAR-MC1 processor
 ┣ 📄 core_sc000.h     | SC000 processor
 ┗ 📄 core_sc300.h     | SC300 processor

The files also include the \ref cmsis_compiler_files and depending on the features supported by the core also correponding \ref cmsis_feature_files.

### CMSIS-Core Compiler Files {#cmsis_compiler_files}

The CMSIS-Core compiler files provide consistent implementations of `#define` symbols that allow toolchain-agnostic usage of CMSIS-Core. \ref cmsis_processor_files rely on such toolchain-agnostic abstractions by including `cmsis_compiler.h` file that then selects the target compiler-specific implementatation depending on the toolchain used in the project.

CMSIS-Core compiler files are provided in `CMSIS/Core/Include/` directory, and define the supported compilers as listed in \ref tested_tools_sec. \ref compiler_conntrol_gr documents the functionalities provided by the CMSIS compliant toolchains.

Header File                            | Description
:--------------------------------------|:-------------------
📂 CMSIS/Core/Include    | CMSIS-Core include folder ([See on GitHub](https://github.com/ARM-software/CMSIS_6/tree/main/CMSIS/Core/Include/))
 ┣ 📄 cmsis_compiler.h                  | Main CMSIS-Core compiler header file
 ┗ 📂 m-profile                         | Directory for M-Profile specific files
 &emsp;&nbsp; ┣ 📄 cmsis_armclang_m.h   | CMSIS-Core Arm Clang compiler file for Cortex-M
 &emsp;&nbsp; ┣ 📄 cmsis_clang_m.h      | CMSIS-Core Clang compiler file for Cortex-M
 &emsp;&nbsp; ┣ 📄 cmsis_gcc_m.h        | CMSIS-Core GCC compiler file for Cortex-M
 &emsp;&nbsp; ┣ 📄 cmsis_iccarm_m.h     | CMSIS-Core IAR compiler file for Cortex-M
 &emsp;&nbsp; ┗ 📄 cmsis_tiarmclang_m.h | CMSIS-Core TI Clang compiler file

### CMSIS-Core Architecture Feature Files {#cmsis_feature_files}

Several architecture-specific features are implemented in separate header files that then gets included by \ref cmsis_processor_files if corresponding feature is supported. 

For Cortex-M cores following architecture feature files are provided in the `CMSIS/Core/Include/m-profile/` folder:

Header File         | Feature
:-------------------|:-------------------
📂 CMSIS/Core/Include    | CMSIS-Core include folder ([See on GitHub](https://github.com/ARM-software/CMSIS_6/tree/main/CMSIS/Core/Include/))
 ┣ 📂 m-profile                    | Directory for M-Profile specific files
 &emsp;&nbsp; ┣ 📄 armv7m_cache1.h | \ref cache_functions_m7
 &emsp;&nbsp; ┣ 📄 armv7m_mpu.h    | \ref mpu_functions
 &emsp;&nbsp; ┣ 📄 armv8m_mpu.h    | \ref mpu8_functions
 &emsp;&nbsp; ┣ 📄 armv8m_pmu.h    | \ref pmu8_functions
 &emsp;&nbsp; ┗ 📄 armv81m_pac.h   | PAC functions
 ┗ 📄 tz_context.h                 | API header file for \ref context_trustzone_functions

### CMSIS Version and Processor Information {#core_version_sect}

\ref __CM_CMSIS_VERSION is defined in the `cmsis_version.h` file and provides the version of the CMSIS-Core (Cortex-M). It is constructed as follows:

```c
#define __CM_CMSIS_VERSION_MAIN  ( 5U)                                    /* [31:16] CMSIS Core(M) main version */
#define __CM_CMSIS_VERSION_SUB   ( 7U)                                    /* [15:0]  CMSIS Core(M) sub version */
#define __CM_CMSIS_VERSION       ((__CM_CMSIS_VERSION_MAIN << 16U) | \
                                   __CM_CMSIS_VERSION_SUB           )     /* CMSIS Core(M) version number */

```

The `cmsis_version.h` is included by each `cpu_<core>.h` so the CMSIS version defines are available via them already.

The `cpu_<core>.h` files use specific defines (such as \ref __CORTEX_M) that provide identification of the target processor core.

These defines can be used in the \ref device_h_pg to verify a minimum CMSIS-Core version as well as the target processor. Read more at \ref version_control_gr.
