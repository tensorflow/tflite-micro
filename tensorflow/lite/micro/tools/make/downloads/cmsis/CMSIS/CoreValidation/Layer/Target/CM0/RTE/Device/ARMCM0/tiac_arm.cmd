/****************************************************************************/
/* tiac_arm.cmd - COMMAND FILE FOR LINKING ARM C PROGRAMS                   */
/*                                                                          */
/*   Description: This file is a sample command file that can be used       */
/*                for linking programs built with the TI Arm Clang          */
/*                Compiler.   Use it as a guideline; you may want to change */
/*                the allocation scheme according to the size of your       */
/*                program and the memory layout of your target system.      */
/*                                                                          */
/****************************************************************************/
-c                                         /* LINK USING C CONVENTIONS      */
-stack  0x4000                             /* SOFTWARE STACK SIZE           */
-heap   0x4000                             /* HEAP AREA SIZE                */
--args 0x1000

/* SPECIFY THE SYSTEM MEMORY MAP */
MEMORY
{
    V_MEM    : org = 0x00000000   len = 0x00001000  /* INT VECTOR */
    P_MEM    : org = 0x00001000   len = 0x20000000  /* PROGRAM MEMORY (ROM) */
    D_MEM    : org = 0x20001000   len = 0x20000000  /* DATA MEMORY    (RAM) */
}

/* SPECIFY THE SECTIONS ALLOCATION INTO MEMORY */
SECTIONS
{
    .intvecs    : {} > 0x0             /* INTERRUPT VECTORS                 */
    .bss        : {} > D_MEM           /* GLOBAL & STATIC VARS              */
    .data       : {} > D_MEM
    .sysmem     : {} > D_MEM           /* DYNAMIC MEMORY ALLOCATION AREA    */
    .stack      : {} > D_MEM           /* SOFTWARE SYSTEM STACK             */

    .text       : {} > P_MEM           /* CODE                              */
    .cinit      : {} > P_MEM           /* INITIALIZATION TABLES             */
    .const      : {} > P_MEM           /* CONSTANT DATA                     */
    .rodata     : {} > P_MEM, palign(4)
    .init_array : {} > P_MEM           /* C++ CONSTRUCTOR TABLES            */


    .TI.ramfunc : {} load=P_MEM, run=D_MEM, table(BINIT)
}
