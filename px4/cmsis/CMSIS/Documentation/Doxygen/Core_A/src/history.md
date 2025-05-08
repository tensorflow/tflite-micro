# Revision History {#rev_histCoreA}

CMSIS-Core(A) component is maintaned with own versioning that gets incremented together with the [CMSIS Software Pack](../General/cmsis_pack.html) releases.

The table below provides information about the changes delivered with specific versions of CMSIS-Core(A) updates.

<table class="cmtable" summary="Revision History">
    <tr>
      <th>Version</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>V6.1.0</td>
      <td>
        <ul>
          <li>Added support for Cortex-M52</li>
          <li>Added deprecated CoreDebug symbols for CMSIS 5 compatibility</li>
          <li>Added define CMSIS_DISABLE_DEPRECATED to hide deprecated symbols</li>
        </ul>
      </td>
    </tr>    <tr>
      <td>V6.0.0</td>
      <td>
        <ul>
          <li>Core(M) and Core(A) joined into single Core component</li>
          <li>Core header files reworked, aligned with TRMs</li>
          <li>Previously deprecated features removed</li>
          <li>Dropped support for Arm Compiler 5</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.2.1</td>
      <td>
        <ul>
          <li>Bugfixes for Cortex-A32</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.2.0</td>
      <td>
        <ul>
          <li>Fixed \ref GIC_SetPendingIRQ to use GICD_SGIR instead of GICD_SPENDSGIR
              for compliance with all GIC specification versions.</li>
          <li>Added missing DSP intrinsics.</li>
          <li>Reworked assembly intrinsics: volatile, barriers and clobbers.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.1.4</td>
      <td>
        <ul>
          <li>Fixed __FPU_Enable().</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.1.3</td>
      <td>
        <ul>
          <li>Fixed __get_SP_usr() / __set_SP_usr() for ArmClang.</li>
          <li>Fixed zero argument handling in __CLZ() .</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.1.2</td>
      <td>
        <ul>
          <li>Removed using get/set built-ins FPSCR in GCC >= 7.2 due to shortcomings.</li>
          <li>Fixed co-processor register access macros for Arm Compiler 5.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.1.1</td>
      <td>
        <ul>
          <li>Refactored L1 cache maintenance to be compiler agnostic.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.1.0</td>
      <td>
        <ul>
          <li>Added compiler_iccarm.h for IAR compiler.</li>
          <li>Added missing core access functions for Arm Compiler 5.</li>
          <li>Aligned access function to coprocessor 15.</li>
          <li>Additional generic Timer functions.</li>
          <li>Bug fixes and minor enhancements.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>V1.0.0</td>
      <td>Initial Release for Cortex-A5/A7/A9 processors.</td>
    </tr>
</table>
