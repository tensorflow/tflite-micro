/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Corstone-300 FVP Memory Map
   0x00000000 : ITCM (Instruction TCM) up to 256KB for Vectors & Code
   0x20000000 : SRAM (4MB) for Data, BSS, and Heap
   0x60000000 : DDR (16MB) for large external arenas if configured
*/

MEMORY
{
  ITCM  (rx)  : ORIGIN = 0x00000000, LENGTH = 4096K
  SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 4096K
  DDR   (rwx) : ORIGIN = 0x60000000, LENGTH = 16384K
}

ENTRY(Reset_Handler)

SECTIONS
{
  .vectors :
  {
    KEEP(*(.vectors))
  } > ITCM

  .vectors_sram :
  {
    KEEP(*(.vectors_sram))
  } > SRAM

  .text :
  {
    *(.text*)
    *(.rodata*)
    KEEP (*(SORT_NONE(.init)))
    KEEP (*(SORT_NONE(.fini)))
    . = ALIGN(16);
  } > ITCM

  .ARM.exidx :
  {
    __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    __exidx_end = .;
  } > ITCM

  .data :
  {
    *(.data*)
    . = ALIGN(16);
  } > SRAM

  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);
    . = ALIGN(16);
  } > SRAM

  .bss (NOLOAD) :
  {
    __bss_start__ = .;
    *(.bss*)
    *(COMMON)
    *(.heap_pool*)
    *(.bss.ethosu_scratch*)
    *(.bss.tensor_arena*)
    . = ALIGN(16);
    __bss_end__ = .;
  } > SRAM

  __INITIAL_SP  = 0x20400000;
  __StackTop    = __INITIAL_SP;
  __stack       = __INITIAL_SP;
  __STACK_LIMIT = 0x203E0000;
}
