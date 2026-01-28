# Places code in .text._start so linker puts it at reset address.
.section .text._start
.global _start
_start:
    la sp, __stack_top    # Load stack pointer
    add s0, sp, zero      # Set frame pointer
    jal zero, main        # Jump to main, do not expect return
loop:
    j loop

