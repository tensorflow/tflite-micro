/*
 * Copyright (c) 2023 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------------
  Stack seal size definition
 *----------------------------------------------------------------------------*/
#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
#define __STACKSEAL_SIZE   ( 8 )
#else
#define __STACKSEAL_SIZE   ( 0 )
#endif

/* ----------------------------------------------------------------------------
  Memory definition
 *----------------------------------------------------------------------------*/
MEMORY
{
  ROM0  (rx)  : ORIGIN = __ROM0_BASE, LENGTH = __ROM0_SIZE
#if __ROM1_SIZE > 0
  ROM1  (rx)  : ORIGIN = __ROM1_BASE, LENGTH = __ROM1_SIZE
#endif
#if __ROM2_SIZE > 0
  ROM2  (rx)  : ORIGIN = __ROM2_BASE, LENGTH = __ROM2_SIZE
#endif
#if __ROM3_SIZE > 0
  ROM3  (rx)  : ORIGIN = __ROM3_BASE, LENGTH = __ROM3_SIZE
#endif

  RAM0  (rwx) : ORIGIN = __RAM0_BASE, LENGTH = __RAM0_SIZE
#if __RAM1_SIZE > 0
  RAM1  (rwx) : ORIGIN = __RAM1_BASE, LENGTH = __RAM1_SIZE
#endif
#if __RAM2_SIZE > 0
  RAM2  (rwx) : ORIGIN = __RAM2_BASE, LENGTH = __RAM2_SIZE
#endif
#if __RAM3_SIZE > 0
  RAM3  (rwx) : ORIGIN = __RAM3_BASE, LENGTH = __RAM3_SIZE
#endif
}

/* Linker script to place sections and symbol values. Should be used together
 * with other linker script that defines memory regions FLASH and RAM.
 * It references following symbols, which must be defined in code:
 *   Reset_Handler : Entry of reset handler
 *
 * It defines following symbols, which code can use without definition:
 *   __exidx_start
 *   __exidx_end
 *   __copy_table_start__
 *   __copy_table_end__
 *   __zero_table_start__
 *   __zero_table_end__
 *   __etext          (deprecated)
 *   __data_start__
 *   __preinit_array_start
 *   __preinit_array_end
 *   __init_array_start
 *   __init_array_end
 *   __fini_array_start
 *   __fini_array_end
 *   __data_end__
 *   __bss_start__
 *   __bss_end__
 *   __end__
 *   end
 *   __HeapLimit
 *   __StackLimit
 *   __StackTop
 *   __stack
 */
ENTRY(Reset_Handler)

SECTIONS
{
  .text :
  {
    KEEP(*(.vectors))
    *(.text*)

    KEEP(*(.init))
    KEEP(*(.fini))

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    *(.rodata*)

    KEEP(*(.eh_frame*))
  } > ROM0

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
  .gnu.sgstubs :
  {
    . = ALIGN(32);
  } > ROM0
#endif

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > ROM0

  __exidx_start = .;
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > ROM0
  __exidx_end = .;

  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;

    LONG (LOADADDR(.data))
    LONG (ADDR(.data))
    LONG (SIZEOF(.data) / 4)

    /* Add each additional data section here */
/*
    LONG (LOADADDR(.data2))
    LONG (ADDR(.data2))
    LONG (SIZEOF(.data2) / 4)
*/
    __copy_table_end__ = .;
  } > ROM0

  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;

/*  .bss initialization to zero is already done during C Run-Time Startup.
    LONG (ADDR(.bss))
    LONG (SIZEOF(.bss) / 4)
*/

    /* Add each additional bss section here */
/*
    LONG (ADDR(.bss2))
    LONG (SIZEOF(.bss2) / 4)
*/
    __zero_table_end__ = .;
  } > ROM0

  /*
   * This __etext variable is kept for backward compatibility with older,
   * ASM based startup files.
   */
  PROVIDE(__etext = LOADADDR(.data));

  .data : ALIGN(4)
  {
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)

    . = ALIGN(4);
    /* preinit data */
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    . = ALIGN(4);
    /* init data */
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);

    . = ALIGN(4);
    /* finit data */
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    KEEP(*(.jcr*))
    . = ALIGN(4);
    /* All data end */
    __data_end__ = .;

  } > RAM0 AT > ROM0

  /*
   * Secondary data section, optional
   *
   * Remember to add each additional data section
   * to the .copy.table above to assure proper
   * initialization during startup.
   */
/*
  .data2 : ALIGN(4)
  {
    . = ALIGN(4);
    __data2_start__ = .;
    *(.data2)
    *(.data2.*)
    . = ALIGN(4);
    __data2_end__ = .;

  } > RAM1 AT > ROM0 
*/

  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > RAM0 AT > RAM0

  /*
   * Secondary bss section, optional
   *
   * Remember to add each additional bss section
   * to the .zero.table above to assure proper
   * initialization during startup.
   */
/*
  .bss2 :
  {
    . = ALIGN(4);
    __bss2_start__ = .;
    *(.bss2)
    *(.bss2.*)
    . = ALIGN(4);
    __bss2_end__ = .;
  } > RAM1 AT > RAM1
*/

  .heap (NOLOAD) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > RAM0

  .stack (ORIGIN(RAM0) + LENGTH(RAM0) - __STACK_SIZE - __STACKSEAL_SIZE) (NOLOAD) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > RAM0
  PROVIDE(__stack = __StackTop);

#if __STACKSEAL_SIZE > 0
  .stackseal (ORIGIN(RAM0) + LENGTH(RAM0) - __STACKSEAL_SIZE) (NOLOAD) :
  {
    . = ALIGN(8);
    __StackSeal = .;
    . = . + 8;
    . = ALIGN(8);
  } > RAM0
#endif

  /* Check if data + heap + stack exceeds RAM limit */
  ASSERT(__StackLimit >= __HeapLimit, "region RAM overflowed with stack")
}
