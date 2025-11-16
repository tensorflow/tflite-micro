ENTRY(_start)
SECTIONS
{
 # Typical start for RISC-V bare-metal (QEMU virt)'s memory layout 
  . = 0x80000000;

 # Code section: contains all executable instructions 
  text : {
    *(.text._start);  # Put _start first 
    *(.text*);
  }

 # Enforces a page boundary between code and data, letting the linker separate permissions (
 # RX for code, RW for data), removing the RWX warning. 
 . = ALIGN(0x1000);  

 # Data section: contains initialized global and static variables 
  .data : { 
    *(.data*)
    *(.rodata*)
  }

 # BSS section: contains zero-initialized or uninitialized variables 
  .bss : {
    *(.bss*)
    *(COMMON)
  }  

  # Stack section: reserves memory for the runtime stack 
  # Without proper initialization of stack top, sometimes QEMU halts and sometimes produces no output
  # (blinking cursor) 
  .stack : {
    . = . + 0x10000; 
    __stack_top = .;   
  }

 # Avoids undefined reference to _end in _sbrk function 
 _end = .;
}