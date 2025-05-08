# Revision History {#rtos_revisionHistory}

CMSIS-RTOS2 API component is maintained with its own versioning that gets offically updated upon releases of the [CMSIS Software Pack](../General/cmsis_pack.html) releases.

The table below provides information about the changes delivered with specific versions of CMSIS-RTOS2 API.

<table class="cmtable" summary="Revision History">
    <tr>
      <th>Version</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>V2.3.0</td>
      <td>
         Changed:
         - OS Tick API moved from Device to CMSIS class
         Added:
         - Provisional support for processor affinity in SMP systems:
           - \ref osThreadAttr_t :: affinity_mask
           - \ref osThreadSetAffinityMask, \ref osThreadGetAffinityMask
      </td>
    </tr>
    <tr>
      <td>V2.2.0</td>
      <td>
        Added:
         - support for \ref CMSIS_RTOS_ProcessIsolation :
           - Kernel Management: \ref osKernelProtect, \ref osKernelDestroyClass
           - Thread Management: \ref osThreadGetClass, \ref osThreadGetZone, \ref osThreadSuspendClass, \ref osThreadResumeClass, \ref osThreadTerminateZone, \ref osThreadFeedWatchdog, \ref osThreadProtectPrivileged
           - Thread attributes: \ref osThreadZone, \ref osThreadUnprivileged, \ref osThreadPrivileged
           - Object attributes: \ref osSafetyClass
           - Handler functions: \ref osWatchdogAlarm_Handler
           - Zone Management: \ref osZoneSetup_Callback
           - Exception Faults: \ref osFaultResume
         - Additional functions allowed to be called from Interrupt Service Routines:
           - \ref osThreadGetName, \ref osEventFlagsGetName, \ref osTimerGetName, \ref osMutexGetName, \ref osSemaphoreGetName, \ref osMemoryPoolGetName, \ref osMessageQueueGetName
      </td>
    </tr>
    <tr>
      <td>V2.1.3</td>
      <td>
        Added:
          - Additional functions allowed to be called from Interrupt Service Routines:
            - \ref osThreadGetId
     </td>
    </tr>
    <tr>
      <td>V2.1.2</td>
      <td>
        Added:
          - Additional functions allowed to be called from Interrupt Service Routines:
            - \ref osKernelGetInfo, \ref osKernelGetState
     </td>
    </tr>
    <tr>
      <td>V2.1.1</td>
      <td>
        Additional functions allowed to be called from Interrupt Service Routines:
         - \ref osKernelGetTickCount, \ref osKernelGetTickFreq
         
        Changed Kernel Tick type to uint32_t:
         - updated: \ref osKernelGetTickCount, \ref osDelayUntil
     </td>
    </tr>
    <tr>
      <td>V2.1.0</td>
      <td>
        Support for critical and uncritical sections (nesting safe):
         - updated: \ref osKernelLock, \ref osKernelUnlock
         - added: \ref osKernelRestoreLock

        Updated \ref CMSIS_RTOS_ThreadFlagsMgmt "Thread Flags" and \ref CMSIS_RTOS_EventFlags "Event Flags":
         - changed flags parameter and return type from int32_t to uint32_t
     </td>
    </tr>
    <tr>
      <td>V2.0.0</td>
      <td>
        New API Version 2.0 available. 
         - See \ref CMSIS_RTOS for a detailed function reference.
     </td>
    </tr>
    <tr>
      <td>V1.02 - only documentation changes</td>
      <td>
      Added:
        - Overview of the \ref rtosValidation "CMSIS-RTOS Validation" Software Pack.

      Clarified:
        - Behavior of \ref CMSIS_RTOS_TimeOutValue.
     </td>
    </tr>
    <tr>
      <td>V1.02</td>
      <td>
      Added: 
       - New control functions for short timeouts in microsecond resolution **osKernelSysTick**, **osKernelSysTickFrequency**, **osKernelSysTickMicroSec**.

      Removed:
       - osSignalGet.
     </td>
    </tr>
    <tr>
      <td>V1.01</td>
      <td>
      Added:
        - capabilities for C++, kernel initialization and object deletion.
        - Prepared for C++ class interface. In this context to \em const attribute has been moved from osXxxxDef_t typedefs to the osXxxxDef macros.
        - \ref osTimerDelete, \ref osMutexDelete, \ref osSemaphoreDelete .
        - \ref osKernelInitialize that prepares the kernel for object creation.
      </td>
    </tr>
    <tr>
      <td>
      V1.00</td>
      <td>First official Release.<br>
      Added:
       - \ref osKernelStart; starting 'main' as a thread is now an optional feature.
       - Semaphores have now the standard behavior.
       - osTimerCreate does no longer start the timer.
       - \ref osTimerStart (replaces osTimerRestart).

      Changed:
       - osThreadPass is renamed to \ref osThreadYield.
      </td>
    </tr>
    <tr>
      <td>V0.02</td>
      <td>Preview Release.</td>
    </tr>
</table>
