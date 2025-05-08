# Revision History {#driver_revisionHistory}

CMSIS-Driver component is maintained with its own versioning that gets offically updated upon releases of the [CMSIS Software Pack](../General/cmsis_pack.html).

The table below provides information about the changes delivered with specific versions of CMSIS-Driver.

<table class="cmtable" summary="Revision History">
    <tr>
      <th>Version</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>2.10.0</td>
      <td>
        - Updated USB Host API 2.4.0:
          - deprecated: API for OHCI/EHCI Host Controller Interface (HCI)
      </td>
    </tr>
    <tr>
      <td>2.9.0</td>
      <td>
        - Updated VIO API 1.0.0:
          - removed: vioPrint
          - removed: vioSetXYZ, vioGetXYZ
          - removed: vioSetIPv4, vioGetIPv4, vioSetIPv6, vioGetIPv6
        - Added GPIO Driver API 1.0.0
      </td>
    </tr>
    <tr>
      <td>2.8.0</td>
      <td>
        - Changed: removed volatile from status related typedefs APIs
        - Enhanced WiFi Interface API with support for polling Socket Receive/Send
        - Added VIO API 0.1.0 (Preview)
      </td>
    </tr>
    <tr>
      <td>2.7.1</td>
      <td>
        - Finalized WiFi Interface API 1.0.0.
      </td>
    </tr>
    <tr>
      <td>2.7.0</td>
      <td>
        - Added WiFi Interface API 1.0.0-beta.
        - Added custom driver selection to simplify implementation of new CMSIS-Driver.
      </td>
    </tr>
    <tr>
      <td>2.6.0</td>
      <td>
        - Enhanced CAN-Driver API with explicit BUSOFF state.
        - Enhanced NAND-Driver API for ECC handling.
      </td>
    </tr>
    <tr>
      <td>2.05</td>
      <td>
        - Changed: All typedefs related to status have been made volatile.
      </td>
    </tr>
    <tr>
      <td>2.04</td>
      <td>
        - Added: template files for CAN interface driver.
      </td>
    </tr>
    <tr>
      <td>2.03</td>
      <td>
        - Added: CAN API for an interface to CAN peripherals
        - Added: Overview of the \ref driverValidation "CMSIS-Driver Validation" Software Pack.
        - Enhanced: documentation and clarified behavior of the \ref CallSequence.
      </td>
    </tr>
    <tr>
      <td>2.02</td>
      <td>
        - Minor API changes, for exact details refer to the header file of each driver.
        - Added: Flash Interface, NAND interface.
      </td>
    </tr>
    <tr>
      <td>2.00</td>
      <td>API with non-blocking data transfer, independent of CMSIS-RTOS.</td>
    </tr>
    <tr>
      <td>1.10</td>
      <td>Initial release</td>
    </tr>
</table>
