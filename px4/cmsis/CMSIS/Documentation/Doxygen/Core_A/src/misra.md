# MISRA-C Deviations {#coreMISRA_Exceptions_pg}

CMSIS-Core (Cortex-A) uses the common coding rules for CMSIS components that are documented in 
[CMSIS Introduction](../General/index.html).

CMSIS-Core (Cortex-A) violates the following MISRA-C:2012 rules:

 - Directive 4.9, function-like macro defined.<br>
   - Violated since function-like macros are used to generate more efficient code. 
   
 - Rule 1.3, multiple use of '#/##' operators in macro definition.<br>
   - Violated since function-like macros are used to generate more efficient code. 
   
 - Rule 11.4, conversion between a pointer and integer type.<br>
   - Violated because of core register access. 
   
 - Rule 11.6, cast from unsigned long to pointer.<br>
   - Violated because of core register access. 
   
 - Rule 13.5, side effects on right hand side of logical operator.<br>
   - Violated because of shift operand is used in macros and functions. 
   
 - Rule 14.4, conditional expression should have essentially Boolean type.<br>
   - Violated since macros with several instructions are used.
  
 - Rule 15.5, return statement before end of function.<br>
   - Violated to simplify code logic. 

 - Rule 20.10, '#/##' operators used.<br>
   - Violated since function-like macros are used to generate more efficient code. 
   
 - Rule 21.1, reserved to the compiler.<br>
   - Violated since macros with leading underscores are used. 
