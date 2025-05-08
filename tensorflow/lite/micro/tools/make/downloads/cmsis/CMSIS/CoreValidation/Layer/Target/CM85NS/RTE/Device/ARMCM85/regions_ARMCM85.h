#ifndef REGIONS_ARMCM85_H
#define REGIONS_ARMCM85_H


//-------- <<< Use Configuration Wizard in Context Menu >>> --------------------

// <n>Device pack:   ARM::Cortex_DFP@1.0.0-dev16
// <i>Device pack used to generate this file

// <h>ROM Configuration
// =======================
// <h> ROM_S=<__ROM0>
//   <o> Base address <0x0-0xFFFFFFFF:8>
//   <i> Defines base address of memory region.
//   <i> Default: 0x00000000
#define __ROM0_BASE 0x00200000
//   <o> Region size [bytes] <0x0-0xFFFFFFFF:8>
//   <i> Defines size of memory region.
//   <i> Default: 0x00200000
#define __ROM0_SIZE 0x00200000
//   <q>Default region
//   <i> Enables memory region globally for the application.
#define __ROM0_DEFAULT 1
//   <q>Startup
//   <i> Selects region to be used for startup code.
#define __ROM0_STARTUP 1
// </h>

// <h> ROM_NS=<__ROM1>
//   <o> Base address <0x0-0xFFFFFFFF:8>
//   <i> Defines base address of memory region.
//   <i> Default: 0x00200000
#define __ROM1_BASE 0x00000000
//   <o> Region size [bytes] <0x0-0xFFFFFFFF:8>
//   <i> Defines size of memory region.
//   <i> Default: 0x00200000
#define __ROM1_SIZE 0x00200000
//   <q>Default region
//   <i> Enables memory region globally for the application.
#define __ROM1_DEFAULT 0
//   <q>Startup
//   <i> Selects region to be used for startup code.
#define __ROM1_STARTUP 0
// </h>

// </h>

// <h>RAM Configuration
// =======================
// <h> RAM_S=<__RAM0>
//   <o> Base address <0x0-0xFFFFFFFF:8>
//   <i> Defines base address of memory region.
//   <i> Default: 0x20000000
#define __RAM0_BASE 0x20200000
//   <o> Region size [bytes] <0x0-0xFFFFFFFF:8>
//   <i> Defines size of memory region.
//   <i> Default: 0x00020000
#define __RAM0_SIZE 0x00020000
//   <q>Default region
//   <i> Enables memory region globally for the application.
#define __RAM0_DEFAULT 1
//   <q>No zero initialize
//   <i> Excludes region from zero initialization.
#define __RAM0_NOINIT 0
// </h>

// <h> RAM_NS=<__RAM1>
//   <o> Base address <0x0-0xFFFFFFFF:8>
//   <i> Defines base address of memory region.
//   <i> Default: 0x20200000
#define __RAM1_BASE 0x20000000
//   <o> Region size [bytes] <0x0-0xFFFFFFFF:8>
//   <i> Defines size of memory region.
//   <i> Default: 0x00020000
#define __RAM1_SIZE 0x00020000
//   <q>Default region
//   <i> Enables memory region globally for the application.
#define __RAM1_DEFAULT 0
//   <q>No zero initialize
//   <i> Excludes region from zero initialization.
#define __RAM1_NOINIT 0
// </h>

// </h>

// <h>Stack / Heap Configuration
//   <o0> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//   <o1> Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
#define __STACK_SIZE 0x00000400
#define __HEAP_SIZE 0x00000C00
// </h>


#endif /* REGIONS_ARMCM85_H */
