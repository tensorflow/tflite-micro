# Introduction {#mainpage}

The **CMSIS** (Common Microcontroller Software Interface Standard) is a set of APIs, software components, tools, and workflows that help to simplify software re-use, reduce the learning curve for microcontroller developers, speed-up project build and debug, and thus reduce the time to market for new applications.

CMSIS started as a vendor-independent hardware abstraction layer Arm&reg; Cortex&reg;-M based processors and was later extended to support entry-level Arm Cortex-A based processors. To simplify access, CMSIS defines generic tool interfaces and enables consistent device support by providing simple software interfaces to the processor and the peripherals.

CMSIS has been created to help the industry in standardization. It enables consistent software layers and device support across a wide range of development tools and microcontrollers. CMSIS is not a huge software layer that introduces overhead and does not define standard peripherals. The silicon industry can therefore support the wide variations of Arm Cortex processor-based devices with this common standard.

## CMSIS Components {#cmsis_components}

![CMSIS Components Overview](./images/cmsis_components.png)

<h2>CMSIS Base Software Components</h2>

 - Provide software abstractions for basic level functionalities of a device.
 - Maintained in the same GitHub repository and delivered as one \ref cmsis_pack with the name `Arm::CMSIS`.
<div class="tiles">
  <div class="tile" onclick="document.location='../Core/index.html'">
    <span class="tileh h2">CMSIS-Core</span><span class="tiletxt">Standardized access to Arm Cortex processor cores</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS_6/latest/Core/index.html">Guide</a> | <a href="https://github.com/ARM-software/CMSIS_6">GitHub</a> | <a href="https://www.keil.arm.com/packs/cmsis-arm/versions/">Pack</a></span>
  </div>
<div class="tile" onclick="document.location='../Driver/index.html'">
    <span class="tileh h2">CMSIS-Driver</span><span class="tiletxt">Generic peripheral driver interfaces for middleware</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS_6/latest/Driver/index.html">Guide</a> | <a href="https://github.com/ARM-software/CMSIS_6">GitHub</a> | <a href="https://www.keil.arm.com/packs/cmsis-arm/versions/">Pack</a></span>
  </div>
  <div class="tile" onclick="document.location='../RTOS2/index.html'">
   <span class="tileh h2">CMSIS-RTOS2</span><span class="tiletxt">Common API for real-time operating systems</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS_6/latest/RTOS2/index.html">Guide</a> | <a href="https://github.com/ARM-software/CMSIS_6">GitHub</a> | <a href="https://www.keil.arm.com/packs/cmsis-arm/versions/">Pack</a></span>
  </div>
</div>

<h2>CMSIS Extended Software Components</h2>

 - Implement specific functionalities optimized for execution on Arm processors.
 - Maintained in separate GitHub repositories and delivered in standalone CMSIS-Packs.
<div class="tiles">
  <div class="tile" onclick="document.location='../DSP/index.html'">
    <span class="tileh h2">CMSIS-DSP</span><span class="tiletxt">Optimized compute functions for embedded systems</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS-DSP/latest/">Guide</a> | <a href="https://github.com/ARM-software/CMSIS-DSP">GitHub</a> | <a href="https://www.keil.arm.com/packs/cmsis-dsp-arm/versions/">Pack</a></span>
  </div>
  <div class="tile" onclick="document.location='../NN/index.html'">
    <span class="tileh h2">CMSIS-NN</span><span class="tiletxt">Efficient and performant neural network kernels</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS-NN/latest/">Guide</a> | <a href="https://github.com/ARM-software/CMSIS-NN">GitHub</a> | <a href="https://www.keil.arm.com/packs/cmsis-nn-arm/versions/">Pack</a></span>
  </div>
  <div class="tile" onclick="document.location='../View/index.html'">
    <span class="tileh h2">CMSIS-View</span><span class="tiletxt">Event Recorder and Component Viewer technology</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS-View/latest/">Guide</a> | <a href="https://github.com/ARM-software/CMSIS-View">GitHub</a> | <a href="https://www.keil.arm.com/packs/cmsis-view-arm/versions/">Pack</a></span>
  </div>
  <div class="tile" onclick="document.location='../Compiler/index.html'">
    <span class="tileh h2">CMSIS-Compiler</span><span class="tiletxt">Retarget I/O functions of the standard C run-time library</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS-Compiler/latest/">Guide</a> | <a href="https://github.com/ARM-software/CMSIS-Compiler">GitHub</a> | <a href="https://www.keil.arm.com/packs/cmsis-compiler-arm/versions/">Pack</a></span>
  </div>
</div>

<h2>CMSIS Tools</h2>

- Provide useful utilities for software development workflows with CMSIS-based components.
- Maintained in separate GitHub repositories.
<div class="tiles">
  <div class="tile" onclick="document.location='../Toolbox/index.html'">
    <span class="tileh h2">CMSIS-Toolbox</span><span class="tiletxt">A set of command-line tools to work with software packs</span><span class="tilelinks"><a href="https://github.com/Open-CMSIS-Pack/cmsis-toolbox/blob/main/README.md">Guide</a> | <a href="https://github.com/Open-CMSIS-Pack/cmsis-toolbox">GitHub</a></span>
  </div>
  <div class="tile" onclick="document.location='../Stream/index.html'">
    <span class="tileh h2">CMSIS-Stream</span><span class="tiletxt">Tools and methods for optimizing DSP/ML block data streams</span><span class="tilelinks"><a href="https://github.com/ARM-software/CMSIS-Stream/blob/main/README.md">Guide</a> | <a href="https://github.com/ARM-software/cmsis-stream">GitHub</a></span>
  </div>
  <div class="tile" onclick="document.location='../DAP/index.html'">
    <span class="tileh h2">CMSIS-DAP</span><span class="tiletxt">Firmware for debug units interfacing to CoreSight Debug Access Port</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS-DAP/latest/">Guide</a> | <a href="https://github.com/ARM-software/CMSIS-DAP">GitHub</a></span>
  </div>
  <div class="tile" onclick="document.location='../Zone/index.html'">
    <span class="tileh h2">CMSIS-Zone</span><span class="tiletxt">Defines methods to describe system resources and to partition them</span><span class="tilelinks"><a href="https://arm-software.github.io/CMSIS-Zone/latest/">Guide</a> | <a href="https://github.com/ARM-software/CMSIS-Zone">GitHub</a></span>
  </div>
</div>


<h2>CMSIS Specifications</h2>

- Define methodologies and workflows for embedded software development.
<div class="tiles">
  <div class="tile" onclick="document.location='https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html'">
    <span class="tileh h2">CMSIS-Pack</span><span class="tiletxt">Delivery mechanism for software components and device/board support</span><span class="tilelinks"><a href="https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html">Guide</a> | <a href="https://github.com/Open-CMSIS-Pack/Open-CMSIS-Pack-Spec">GitHub</a></span>
  </div>
  <div class="tile" onclick="document.location='https://open-cmsis-pack.github.io/svd-spec'">
    <span class="tileh h2">CMSIS-SVD</span><span class="tiletxt">Peripheral description of a device for debug view</span><span class="tilelinks"><a href="https://open-cmsis-pack.github.io/svd-spec">Guide</a> | <a href="https://github.com/Open-CMSIS-Pack/svd-spec">GitHub</a></span>
  </div>
</div>

## Benefits {#benefits}

The benefits of the CMSIS are:

 - CMSIS reduces the learning curve, development costs, and time-to-market. Developers can write software quicker through a variety of easy-to-use, standardized software interfaces.
 - Consistent software interfaces improve the software portability and re-usability. Generic software libraries and interfaces provide consistent software framework.
 - It provides interfaces for debug connectivity, debug peripheral views, software delivery, and device support to reduce time-to-market for new microcontroller deployment.
 - It allows to use the compiler of your choice, as it is compiler independent and thus supported by mainstream compilers.
 - It enhances program debugging with peripheral information for debuggers and ITM channels for printf-style output.
 - CMSIS is delivered in CMSIS-Pack format which enables fast software delivery, simplifies updates, and enables consistent integration into development tools.
 - CMSIS-Zone will simplify system resource and partitioning as it manages the configuration of multiple processors, memory areas, and peripherals.
 - IDE and Continuous Integration (CI) are important workflows for embedded software developers. The CMSIS-Toolbox provides command-line build tools with CMake backend and integration into IDEs such as VS Code.

## Development {#development}

CMSIS is defined in close cooperation with various silicon and software vendors and provides a common approach to interface to peripherals, real-time operating systems, and middleware components. It is intended to enable the combination of software components from multiple vendors.

CMSIS is open-source and collaboratively developed. The repository for the base components is [github.com/Arm-software/CMSIS_6](https://github.com/ARM-software/CMSIS_6).

## Coding Rules {#coding_rules}

The CMSIS uses the following essential coding rules and conventions:

 - Compliant with ANSI C (C99) and C++ (C++03).
 - Uses ANSI C standard data types defined in **<stdint.h>**.
 - Variables and parameters have a complete data type.
 - Expressions for `#define` constants are enclosed in parenthesis.
 - Conforms to MISRA 2012 (but does not claim MISRA compliance). MISRA rule violations are documented.

In addition, the CMSIS recommends the following conventions for identifiers:

 - **CAPITAL** names to identify Core Registers, Peripheral Registers, and CPU Instructions.
 - **CamelCase** names to identify function names and interrupt functions.
 - **Namespace_** prefixes avoid clashes with user identifiers and provide functional groups (i.e. for peripherals, RTOS, or DSP Library).

The CMSIS is documented within the source files with:

 - Comments that use the C or C++ style.
 - [Doxygen](https://www.doxygen.nl/) compliant **function comments** that provide:
    - brief function overview.
    - detailed description of the function.
    - detailed parameter explanation.
    - detailed information about return values.

Doxygen comment example:

```c
/**
 * @brief  Enable Interrupt in NVIC Interrupt Controller
 * @param  IRQn  interrupt number that specifies the interrupt
 * @return none.
 * Enable the specified interrupt in the NVIC Interrupt Controller.
 * Other settings of the interrupt such as priority are not affected.
 */
```

## Validation {#validation}

The various components of CMSIS are validated using mainstream compilers. To get a diverse coverage, Arm Compiler v6 (based on LLVM front-end) and GCC are used in the various tests. For each component, the section **Validation** describes the scope of the various verification steps.

CMSIS components are compatible with a range of C and C++ language standards. The CMSIS components comply with the [Application Binary Interface (ABI) for the Arm Architecture](https://github.com/ARM-software/abi-aa). This ensures C API interfaces that support inter-operation between various toolchains.

As CMSIS defines API interfaces and functions that scale to a wide range of processors and devices, the scope of the run-time test coverage is limited. However, several components are validated using dedicated test suites ([CMSIS-Driver](../Driver/driverValidation.html), and [CMSIS-RTOS v2](../RTOS2/rtosValidation.html)).

The CMSIS source code is checked for MISRA C:2012 conformance. MISRA deviations are documented with reasonable effort, however Arm does not claim MISRA compliance as there is today for example no guideline enforcement plan. The CMSIS source code is not checked for MISRA C++:2008 conformance as there is a risk that it is incompatible with C language standards, specifically warnings that may be generated by the various C compilers.

## Migration from CMSIS v5 {#migration_cmsis5}

The functionality of invidivdual CMSIS v6 software components is kept primarily same as in CMSIS v5.9.0. However, some CMSIS components are now delivered in their standalone CMSIS packs and may also have different naming, structure and dependencies.

While use of CMSIS-Pack concept greatly helps to abstract many of this changes from users, there are still some simple adaptation required to fully enable CMSIS v6 support in software developed based on CMSIS v5 structure. Following migration guides explain necessary steps for such porting:

 - [Migrating CMSIS-based projects from CMSIS v5 to CMSIS v6](https://learn.arm.com/learning-paths/microcontrollers/project-migration-cmsis-v6)
 - [CMSIS-Pack Migration Guide](https://learn.arm.com/learning-paths/microcontrollers/pack-migration-cmsis-v6)

> **Note**
> - In the version 6.0.0, the CMSIS-Core header files for Cortex-M devices have received some modifications that are incompatible with previous CMSIS-Core versions. Refer to the [CMSIS-Core Revision History](../Core/core_revisionHistory.html) for more information.

## License {#License}

CMSIS is provided free of charge by Arm under the [Apache 2.0 License](https://raw.githubusercontent.com/ARM-software/CMSIS_6/main/LICENSE).
