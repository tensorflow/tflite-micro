# Using CMSIS-RTOS2 Interface {#usingOS2}

This sections explains concepts for using CMSIS-RTOS2 interface. They are recommeded to follow for best reusability of the software across different platforms.

 - \ref rtos2_functionalities lists the OS services supported by CMSIS-RTOS2 API.
 - \ref cmsis_os2_h explains how CMSIS-RTOS2 interface is defined and can be used along with the RTOS-specific files as well as part of CMSIS pack.
 - \ref SystemStartup shows the flow for kernel initialization.
 - \ref rtos_objects explains the approach to creating and using RTOS Objects.
 - \ref CMSIS_RTOS_MemoryMgmt provides information about options for memory management with CMSIS-RTOS2.
 - \subpage CMSIS_RTOS_ProcessIsolation describes the CMSIS-RTOS2 concepts for protecting execution of critical software tasks against potential flaws in other parts of a program.

> **Note**
> - For the guidance on migration from CMSIS-RTOS API v1 to CMSIS-RTOS2, see section [Detailed API Function Differences](https://arm-software.github.io/CMSIS_5/latest/RTOS2/html/os2MigrationFunctions.html) in CMSIS 5 documentation.

## Functionality overview {#rtos2_functionalities}

The CMSIS-RTOS2 defines APIs for common RTOS services as listed below:

 - \ref CMSIS_RTOS_KernelCtrl provides system information and controls the RTOS Kernel.
 - \ref CMSIS_RTOS_ThreadMgmt allows you to define, create, and control RTOS threads (tasks).
 - \ref CMSIS_RTOS_Wait for controlling time delays. Also see \ref CMSIS_RTOS_TimeOutValue.
 - \ref CMSIS_RTOS_TimerMgmt functions are used to trigger the execution of functions.
 - Three different event types support communication between multiple threads and/or ISR:
   - \ref CMSIS_RTOS_ThreadFlagsMgmt "Thread Flags": may be used to indicate specific conditions to a thread.
   - \ref CMSIS_RTOS_EventFlags "Event Flags": may be used to indicate events to a thread or ISR.
   - \ref CMSIS_RTOS_Message "Messages": can be sent to a thread or an ISR. Messages are buffered in a queue.
 - \ref CMSIS_RTOS_MutexMgmt and \ref CMSIS_RTOS_SemaphoreMgmt are incorporated.

The referenced pages contain theory of operation for corresponding services as well as detailed API description with example code.

## cmsis_os2.h API header file {#cmsis_os2_h}

The **CMSIS-RTOS2** interface is provided in **cmsis_os2.h** file - a standard C header file that user applications and middleware components need to include to access CMSIS-RTOS2 API. It contains all the function declarations, as well as type and macro definitions.

An implementation specific header file (*rtos*.h in the picture below) provides access to such definitions. Using the **cmsis_os2.h** along with dynamic object allocation allows to create source code or libraries that require no modifications when using on a different CMSIS-RTOS2 implementation.

![CMSIS-RTOS File Structure](./images/CMSIS_RTOS_Files.png)

Once the files are added to a project, the user can start working with the CMSIS-RTOS functions.

CMSIS-RTOS2 is especially easy use to integrate in projects that support [CMSIS-Pack format](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html). CMSIS-RTOS2 is provided in the [CMSIS 6 Software Pack](../General/cmsis_pack.html) as a software component in form of [a central API defintion](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/cp_Packs.html#cp_APIDef) It is part of component class **CMSIS** and belongs to component group **RTOS2**.

## Coding Rules {#cmsis_os2_coding_rules}

CMSIS-RTOS2 follows [the general CMSIS coding rules](../General/index.html#coding_rules). Additionally following Namespace prefixes are used in CMSIS-RTOS2 API:

 - `os` for all definitions and function names. Examples: \ref osThreadPrivileged, \ref osKernelStart.
 - `os` with postfix `_t` for all typedefs. Examples: \ref osStatus_t, \ref osThreadAttr_t.

## System Startup {#SystemStartup}

When program execution reaches `main()` function there is a recommended order to initialize the hardware and start the kernel.

Your application's `main()` should implement at least the following in the given order:

 1. Initialize and configure hardware including peripherals, memory, pins, clocks and the interrupt system.
 2. Update the system core clock using the respective [CMSIS-Core (Cortex-M)](../Core/group__system__init__gr.html) \if ARMCA or [CMSIS-Core (Cortex-A)](../Core_A/group__system__init__gr.html) \endif function.
 3. Initialize the RTOS kernel using \ref osKernelInitialize.
 4. Optionally, create one thread (for example `app_main`) using \ref osThreadNew, which will be used as a main thread . This thread should take care of creating and starting objects, once it is run by the scheduler. Alternatively, threads can be created directly in `main()`.
 5. Start the RTOS scheduler using \ref osKernelStart which also configures the system tick timer and initializes RTOS specific interrupts. This function does not return in case of successful execution. Therefore, any application code after `osKernelStart` will not be executed.

> **Note**
> - Modifying priorities and groupings in the NVIC by the application after the above sequence is not recommended.
> - Before executing \ref osKernelStart, only the functions \ref osKernelGetInfo, \ref osKernelGetState, and object creation functions (osXxxNew) may be called.

**Code Example**

```c
/*----------------------------------------------------------------------------
 * CMSIS-RTOS 'main' function template
 *---------------------------------------------------------------------------*/
 
#include "RTE_Components.h"
#include  CMSIS_device_header
#include "cmsis_os2.h"
 
/*----------------------------------------------------------------------------
 * Application main thread
 *---------------------------------------------------------------------------*/
void app_main (void *argument) {
 
  // ...
  for (;;) {}
}
 
int main (void) {
 
  // System Initialization
  SystemCoreClockUpdate();
  // ...
 
  osKernelInitialize();                 // Initialize CMSIS-RTOS
  osThreadNew(app_main, NULL, NULL);    // Create application main thread
  osKernelStart();                      // Start thread execution
  for (;;) {}
}
```

## Lifecycle of RTOS Objects {#rtos_objects}

All RTOS objects share a common design concept. The overall life-cycle of an object can be summarized as created -> in use -> destroyed.

### Create Objects {#rtos_objects_create}

An RTOS object (thread, timer, flags, mutex, etc.) is created by calling its `osXxxNew` function (for example \ref osThreadNew, \ref osTimerNew, \ref osEventFlagsNew, etc). The new function returns a numeric identifier that can be used to operate with the new object.

The actual state of an object is typically stored in an object specific control block. The memory layout (and size needed) for the control block is implementation specific. One should not make any specific assumptions about the control block. The control block layout might change and hence should be seen as an implementation internal detail.

In order to expose control about object specific options all `osXxxNew` functions provide an optional `attr` argument, which can be left as \token{NULL} by default. It takes a pointer to an object specific attribute structure, commonly containing the fields:

 - `name` to attach a human readable name to the object for identification,
 - `attr_bits` to control object-specific options,
 - `cb_mem` to provide memory for the control block manually, and
 - `cb_size` to quantify the memory size provided for the control block.

The `name` attribute is only used for object identification, e.g. using RTOS-aware debugging. The attached string is not used for any other purposes internally.

The `cb_mem` and `cb_size` attributes can be used to provide memory for the control block manually instead of relying on the implementation internal memory allocation. One has to assure that the amount of memory pointed to by `cb_mem` is sufficient for the objects control block structure. If the size given as `cb_size` is not sufficient the `osXxxNew` function returns with an error, i.e. returning \token{NULL}. Furthermore providing control block memory manually is less portable. Thus one has to take care about implementation specific alignment and placement requirements for instance. Refer to \ref CMSIS_RTOS_MemoryMgmt for further details.

### Object Usage {#rtos_objects_usage}

After an object has been created successfully it can be used until it is destroyed. The actions
defined for an object depends on its type. Commonly all the `osXxxDoSomething` access function
require the reference to the object to work with as the first `xxx_id` parameter.

The access function can be assumed to apply some sort of sanity checking on the id parameter. So that it is assured one cannot accidentally call an access function with a \token{NULL} object reference. Furthermore the concrete object type is verified, i.e. one cannot call access functions of one object type with a reference to another object type.

All further parameter checks applied are either object and action specific or may even be implementation specific. Thus one should always check action function return values for `osErrorParameter` to assure the provided arguments were accepted.

As a rule of thumb only non-blocking access function can be used from \ref CMSIS_RTOS_ISR_Calls "Interrupt Service Routines" (ISR). This incorporates `osXxxWait` functions (and similar) limited to be called with parameter `timeout` set to \token{0}, i.e. usage of try-semantics.

### Object Destruction {#rtos_objects_delete}

Objects that are not needed anymore can be destructed on demand to free the control block memory. Objects are not destructed implicitly. Thus one can assume an object id to be valid until `osXxxDelete` is called explicitly. The delete function finally frees the control block memory. In case of user provided control block memory, see above, the memory must be freed manually as well.

The only exception one has to take care of are Threads which do not have an explicit `osThreadDelete` function. Threads can either be `detached` or `joinable`. Detached threads are automatically destroyed on termination, i.e. call to \ref osThreadTerminate or \ref osThreadExit or return from thread function. On the other hand joinable threads are kept alive until one explicitly calls \ref osThreadJoin.

## Timeout Values {#CMSIS_RTOS_TimeOutValue}

Timeout value is an argument in many API functions that allows to set the maximum time delay for resolving a request. The timeout value specifies the number of timer ticks until the time delay elapses. The value is an upper bound and depends on the actual time elapsed since the last timer tick.

Examples:

 - timeout value **0** : the system does not wait, even when no resource is available the RTOS function returns instantly. 
 - timeout value **1** : the system waits until the next timer tick occurs; depending on the previous timer tick, it may be a very short wait time.
 - timeout value **2** : actual wait time is between 1 and 2 timer ticks.
 - timeout value \ref osWaitForever : system waits infinite until a resource becomes available. Or one forces the thread to resume using \ref osThreadResume which is discouraged.

![Example of timeout using osDelay()](./images/TimerValues.png)

 - CPU time can be scheduled with the following functionalities:
   - A \a timeout parameter is incorporated in many CMSIS-RTOS2 functions to avoid system lockup. When a timeout is specified, the system waits until a resource is available or an event occurs. While waiting, other threads are scheduled.
   - The \ref osDelay and \ref osDelayUntil functions put a thread into the **WAITING** state for a specified period of time.
   - The \ref osThreadYield provides co-operative thread switching and passes execution to another thread of the same priority.


## Calls from Interrupt Service Routines {#CMSIS_RTOS_ISR_Calls}

The following CMSIS-RTOS2 functions can be called from threads and Interrupt Service Routines (ISR):

 - \ref osKernelGetInfo, \ref osKernelGetState, \ref osKernelGetTickCount, \ref osKernelGetTickFreq, \ref osKernelGetSysTimerCount, \ref osKernelGetSysTimerFreq
 - \ref osThreadGetName, \ref osThreadGetId, \ref osThreadFlagsSet
 - \ref osTimerGetName
 - \ref osEventFlagsGetName, \ref osEventFlagsSet, \ref osEventFlagsClear, \ref osEventFlagsGet, \ref osEventFlagsWait
 - \ref osMutexGetName
 - \ref osSemaphoreGetName, \ref osSemaphoreAcquire, \ref osSemaphoreRelease, \ref osSemaphoreGetCount
 - \ref osMemoryPoolGetName, \ref osMemoryPoolAlloc, \ref osMemoryPoolFree, \ref osMemoryPoolGetCapacity, \ref osMemoryPoolGetBlockSize, \ref osMemoryPoolGetCount, \ref osMemoryPoolGetSpace
 - \ref osMessageQueueGetName, \ref osMessageQueuePut, \ref osMessageQueueGet, \ref osMessageQueueGetCapacity, \ref osMessageQueueGetMsgSize, \ref osMessageQueueGetCount, \ref osMessageQueueGetSpace

Functions that cannot be called from an ISR are verifying the interrupt status and return the status code \ref osErrorISR, in case they are called from an ISR context. In some implementations, this condition might be caught using the HARD_FAULT

## Memory Management {#CMSIS_RTOS_MemoryMgmt}

The \ref CMSIS_RTOS offers two options for memory management the user can choose. For object storage one can either use

 - \ref CMSIS_RTOS_MemoryMgmt_Automatic (fully portable), or
 - \ref CMSIS_RTOS_MemoryMgmt_Manual (implementation specific).

In order to affect the memory allocation scheme all RTOS objects that can be created on request, i.e. those having a `osXxxNew` function, accept an optional `osXxxAttr_t attr` argument on creation. As a rule of thumb the object attributes at least have members to assign custom control block memory, i.e. `cb_mem` and `cb_size` members. By default, i.e. `attr` is `NULL` or `cb_mem` is `NULL`, \ref CMSIS_RTOS_MemoryMgmt_Automatic is used. Providing a pointer to user memory in `cb_mem` switches
to \ref CMSIS_RTOS_MemoryMgmt_Manual.

### Automatic Dynamic Allocation {#CMSIS_RTOS_MemoryMgmt_Automatic}

The automatic allocation is the default and viable for many use-cases. Moreover it is fully portable across different implementations of the \ref CMSIS_RTOS. The common drawback of dynamic memory allocation is the possibility of memory fragmentation and exhaustion. Given that all needed objects are created once upon system initialization and never deleted at runtime this class of runtime failures can be prevented, though.

The actual allocation strategy used is implementation specific, i.e. whether global heap or preallocated memory pools are used.

**Code Example:**

```c
#include "cmsis_os2.h"                          // CMSIS-RTOS2 API header
  
osMutexId_t mutex_id;
osMutexId_t mutex2_id;
  
const osMutexAttr_t Thread_Mutex_attr = {
  "myThreadMutex",                              // human readable mutex name
  osMutexRecursive | osMutexPrioInherit,        // attr_bits
  NULL,                                         // memory for control block (default)
  0U                                            // size for control block (default)
};
  
void CreateMutex (void)  {
  mutex_id = osMutexNew(NULL);                  // use default values for all attributes
  mutex2_id = osMutexNew(&Thread_Mutex_attr);   // use attributes from defined structure
  :
}
```

The Mutexes in this example are created using automatic memory allocation.

### Manual User-defined Allocation {#CMSIS_RTOS_MemoryMgmt_Manual}

One can get fine grained control over memory allocation by providing user-defined memory. The actual requirements such user-defined memory are kernel-specific. Thus one needs to carefully refer to the size and alignment rules of the implementation used.

**Code Example:**

```c
#include "cmsis_os2.h"                          // CMSIS-RTOS2 API header
#include "rtx_os.h"                             // kernel include file
  
osMutexId_t mutex_id;
  
static osRtxMutex_t mutex_cb __attribute__((section(".bss.os.mutex.cb")));  // Placed on .bss.os.mutex.cb section for RTX5 aware debugging
  
const osMutexAttr_t Thread_Mutex_attr = {
  "myThreadMutex",                              // human readable mutex name
  osMutexRecursive | osMutexPrioInherit,        // attr_bits
  &mutex_cb,                                    // memory for control block (user-defined)
  sizeof(mutex_cb)                              // size for control block (user-defined)
};
  
void CreateMutex (void)  {
  mutex_id = osMutexNew(&Thread_Mutex_attr);    // use attributes from defined structure
  :
}
```

The above example uses user-defined memory for the mutex control block. For this `mutex_cb` is defined with the control block type provided by the kernel header file `rtx_os.h` from CMSIS-RTX RTOS kernel.
