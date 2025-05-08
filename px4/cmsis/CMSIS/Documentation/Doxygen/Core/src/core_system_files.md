# System Configuration Files system_<Device>.c and system_<Device>.h {#system_c_pg}

CMSIS-Core system configuration files provide as a minimum the functions for system initialization and clock configuration as described under \ref system_init_gr. The file names use naming convention `system_<Device>.h` and `system_<Device>.c`, where `<Device>` corresponds to the device name.

`system_<Device>.h` file shall contain the prototypes for accessing the public functions and `system_<Device>.c` shall contain corresponding implementations.

The system configuration functions are device specific and need adaptation for the target device. The silicon vendor might expose other functions and configuration parameters such as XTAL, power configuration, etc.

Additional application-specific adaptations may be required in the initialization code and therefore the system configuration file shall be located in the application project. \ref cmsis_files_dfps explains how this can be achieved when device support is provided in [CMSIS pack format](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/index.html).

# system_Device Template Files {#system_Device_sec}

CMSIS-Core \ref cmsis_template_files include `system_Device.c` and `system_Device.h` files that can be used as a starting point for implementing device-specific system configuration files.
