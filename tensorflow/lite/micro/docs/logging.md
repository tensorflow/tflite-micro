<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->
* [Message Logging in TFLite Micro](#message-logging-in-tflite-micro)
   * [To use MicroPrintf in your application code or kernel implementations:](#to-use-microprintf-in-your-application-code-or-kernel-implementations)
      * [Include this header file:](#include-this-header-file)
      * [Introduce this Bazel BUILD dependency:](#introduce-this-bazel-build-dependency)
      * [Example usage:](#example-usage)
   * [Do Not Use:](#do-not-use)

<!-- Added by: vamsimanchala, at: Wed Nov 23 12:35:39 AM UTC 2022 -->

<!--te-->

# Message Logging in TFLite Micro

TFLM currently support `MicroPrintf` to log errors or messages to the terminal. This is a light-weight printf-lite utility available to log messages to the terminal. The `MicroPrintf` calls are designed to be ignored or optimized-out by the compiler, during deployment, if `TF_LITE_STRIP_ERROR_STRINGS` environment flag is set. This is useful to reduce the binary size of the TFLM application.


## To use MicroPrintf in your application code or kernel implementations:
### Include this header file:
```c++
#include "tensorflow/lite/micro/micro_log.h"
```

### Introduce this Bazel BUILD dependency:
```c++
"//tensorflow/lite/micro:micro_log",
```

### Example usage:
```c++
size_t buffer_size = 1024;
...
MicroPrintf("Failed to allocate buffer of size- %d", buffer_size);

MicroPrintf("TFLM is the best! Bring ML to Embedded targets!");
```

## Do Not Use:
TFLM does not support/recommend the use of `TF_LITE_KERNEL_LOG` and `TF_LITE_REPORT_ERROR` to log errors or messages to the terminal. 
