# MISRA-C Deviations {#coreMISRA_Exceptions_pg}

CMSIS-Core (Cortex-M) uses the common coding rules for CMSIS components that are documented in [CMSIS Introduction](../General/index.html).

CMSIS-Core (Cortex-M) violates the following MISRA-C:2012 rules:

 - Directive 4.2, All usage of assembly language should be documented
   - CMSIS-Core uses assembly statements to access core registers on several places. These locations start with __ASM.
   - Inline assembly statements my be opaque to MISRA Checkers and can cause false-positive warnings.

 - Directive 4.9, function-like macro defined.
   - Violated since function-like macros are used to generate more efficient code.

 - Rule 1.3, multiple use of '#/##' operators in macro definition.
   - Violated since function-like macros are used to generate more efficient code.

 - Rule 11.4, conversion between a pointer and integer type.
   - Violated because of core register access.

 - Rule 11.6, cast from unsigned long to pointer.
   - Violated because of core register access.

 - Rule 13.5, side effects on right hand side of logical operator.
   - Violated because of shift operand is used in macros and functions.

 - Rule 14.4, conditional expression should have essentially Boolean type.
   - Violated since macros with several instructions are used.

 - Rule 15.5, return statement before end of function.
   - Violated to simplify code logic.

 - Rule 20.10, '#/##' operators used.
   - Violated since function-like macros are used to generate more efficient code.

 - Rules 21.1 and 21.2, reserved to the compiler.
   - Violated since macros with leading underscores are used.
