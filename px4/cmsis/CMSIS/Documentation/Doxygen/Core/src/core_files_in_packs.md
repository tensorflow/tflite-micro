# Delivery in CMSIS-Packs {#cmsis_files_dfps}

In simple individual cases the \ref cmsis_core_files can be delivered to developers as part of example projects and SDKs. But the most scalable approach with improved usability is to provide them packaged in [Open-CMSIS-Pack format](https://www.open-cmsis-pack.org), and this page explains how to do this.

The \ref cmsis_standard_files are provided as part of [CMSIS Software pack](../General/cmsis_pack.html) under **CMSIS** class of components and compose there the **CORE** group.

The \ref cmsis_device_files are delivered in [CMSIS Device Family Packs (DFPs)](https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/cp_PackTutorial.html#createPack_DFP) that provide necessary mechanisms for device support in embedded software. The DFPs are typically maintained by device vendors.

Since \ref device_h_pg does not need to be modified in the project, it can be made accessible with its include path.

For uniform experience, the files \ref startup_c_pg and \ref system_c_pg should be provided in the DFP as part of **Device** class in the **Startup** group and defined as configuration files, meaning they are copied from the pack into a project folder and can be modifed there if necessary.

Below is an example of how CMSIS-Core device files can be defined in a .pdsc file of DFP pack, based on the implementation for a generic Cortex-M55 device.

```xml
  <components>
    <!-- CMSIS-Startup components -->
    
    <!-- Cortex-M55 -->
    <component Cclass="Device" Cgroup="Startup" Cvariant="C Startup" Cversion="2.2.0" condition="ARMCM55 CMSIS" isDefaultVariant="true">
      <description>System and Startup for Generic Cortex-M55 device</description>
      <files>
        <!-- include folder / device header file -->
        <file category="include" name="Device/ARMCM55/Include/"/>
        <!-- startup / system file -->
        <file category="sourceC" name="Device/ARMCM55/Source/startup_ARMCM55.c"   version="1.1.0" attr="config"/>
        <file category="sourceC" name="Device/ARMCM55/Source/system_ARMCM55.c"    version="1.1.0" attr="config"/>
        <!-- SAU configuration -->
        <file category="header"  name="Device/ARMCM55/Config/partition_ARMCM55.h" version="1.0.0" attr="config" condition="TZ Secure"/>
      </files>
    </component>
    
  </components>
```

Since the \ref cmsis_device_files require access to implementations from \ref cmsis_standard_files, the Device Family Packs need to specify the dependency on **CMSIS:CORE** component. In the code above this is done with the `condition="ARMCM55 CMSIS"` that is defined in the same .psdc file as:

```xml
  <conditions>
    
    <condition id="ARMCM55 CMSIS">
      <description>Generic Arm Cortex-M55 device startup and depends on CMSIS Core</description>
      <require Dvendor="ARM:82" Dname="ARMCM55*"/>
      <require Cclass="CMSIS" Cgroup="CORE"/>
    </condition>
    
  </conditions>
```

## Device Examples {#device_examples}

The [Cortex_DFP pack](https://github.com/ARM-software/Cortex_DFP) provides generic device definitions for standard Arm Cortex-M cores and contains corresponding \ref cmsis_device_files.

These files can be used as a reference for device support in a DFP pack, but can also be used in projects that require standard Arm processors functionality. See \ref using_arm.

Looking at other Device Family Packs can be also helpful to understand the CMSIS-Core support with DFPs. The list of public CMSIS packs (including DFPs) can be found at [keil.arm.com/packs](https://www.keil.arm.com/packs/).
