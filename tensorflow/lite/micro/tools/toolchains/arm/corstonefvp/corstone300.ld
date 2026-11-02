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

/* Corstone-300 FVP Memory Map (matching CFGITCMSZ=13 / CFGDTCMSZ=13 = 4 MiB):
   0x00000000 : ITCM (4 MiB) for Vectors, Code, and Read-Only Data
   0x20000000 : DTCM / SRAM (4 MiB) for Data, BSS, Heap, and 128 KiB Stack
   0x60000000 : DDR (16 MiB) for large external arenas if configured

   Note: .data uses LMA == VMA in SRAM because FVP's ELF loader populates DTCM
   directly prior to CPU reset (no ROM-to-RAM .data copy-down in Reset_Handler).
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

  .text :
  {
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.vfp11_veneer)
    *(.v4_bx)
    *(.iplt)
    *(.rel.iplt)
    *(.rodata*)
    KEEP(*(SORT_NONE(.init)))
    KEEP(*(SORT_NONE(.fini)))
    KEEP(*(.eh_frame*))
    . = ALIGN(16);
  } > ITCM

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
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
    *(.igot.plt)
    *(.tm_clone_table)
    . = ALIGN(16);
  } > SRAM

  .preinit_array :
  {
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array*))
    PROVIDE_HIDDEN(__preinit_array_end = .);
    . = ALIGN(16);
  } > SRAM

  .init_array :
  {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
    KEEP(*(.init_array .ctors))
    PROVIDE_HIDDEN(__init_array_end = .);
    . = ALIGN(16);
  } > SRAM

  .fini_array :
  {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
    KEEP(*(.fini_array .dtors))
    PROVIDE_HIDDEN(__fini_array_end = .);
    . = ALIGN(16);
  } > SRAM

  .bss (NOLOAD) :
  {
    __bss_start__ = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(16);
    __bss_end__ = .;
  } > SRAM

  __INITIAL_SP  = ORIGIN(SRAM) + LENGTH(SRAM);
  __StackTop    = __INITIAL_SP;
  __stack       = __INITIAL_SP;
  __STACK_LIMIT = __INITIAL_SP - 128K;

  __HeapBase    = ALIGN(__bss_end__, 16);
  __HeapLimit   = __STACK_LIMIT;

  ASSERT(__HeapBase + 32K <= __HeapLimit,
         "SRAM overflow: .data/.bss leaves <32KB heap below 128KB stack")

  .ARM.attributes 0 : { *(.ARM.attributes) }
  .comment        0 : { *(.comment) }
  .debug_abbrev   0 : { *(.debug_abbrev) }
  .debug_aranges  0 : { *(.debug_aranges) }
  .debug_frame    0 : { *(.debug_frame) }
  .debug_info     0 : { *(.debug_info .gnu.linkonce.wi.*) }
  .debug_line     0 : { *(.debug_line .debug_line.* .debug_line_end) }
  .debug_line_str 0 : { *(.debug_line_str) }
  .debug_loc      0 : { *(.debug_loc) }
  .debug_loclists 0 : { *(.debug_loclists) }
  .debug_macro    0 : { *(.debug_macro) }
  .debug_pubnames 0 : { *(.debug_pubnames) }
  .debug_pubtypes 0 : { *(.debug_pubtypes) }
  .debug_ranges   0 : { *(.debug_ranges) }
  .debug_rnglists 0 : { *(.debug_rnglists) }
  .debug_str      0 : { *(.debug_str) }
  .debug_str_offsets 0 : { *(.debug_str_offsets) }

  /DISCARD/ :
  {
    *(.note.GNU-stack)
  }
}
