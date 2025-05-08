/**
\defgroup gpio_interface_gr GPIO Interface
\brief Driver API for GPIO Interface (%Driver_GPIO.h)
\details 
The <b>General-purpose Input/Output Interface</b> (GPIO) features Input/Output operations on pin level (does not support simultaneous operations on multiple pins belonging to the same port).

Features:
 - basic pin configuration (direction, output mode, pull-resistor, event trigger) excluding advanced settings (drive strength or speed, input filter, ...),
 - events on edge detection,
 - setting outputs,
 - reading inputs.

Each function operates on a pin level and uses a pin identification as the first parameter. Pin identification is a virtual number which is mapped to an actual pin.

<b>GPIO API</b>

The following header files define the Application Programming Interface (API) for the GPIO interface:
  - \b %Driver_GPIO.h : Driver API for GPIO Interface

The driver implementation is a typical part of the Device Family Pack (DFP) that supports the 
peripherals of the microcontroller family.


<b>Driver Functions</b>

The driver functions are published in the access struct as explained in \ref DriverFunctions
  - \ref ARM_DRIVER_GPIO : access struct for GPIO driver functions

  
<b>Example Code</b>

The following example code shows the usage of the GPIO interface.

\include GPIO_Demo.c
  
@{
*/


/**
\struct     ARM_DRIVER_GPIO
\details 
The functions of the GPIO driver are accessed by function pointers exposed by this structure.
Refer to \ref DriverFunctions for overview information.

Each instance of a GPIO interface provides such an access structure. 
The instance is identified by a postfix number in the symbol name of the access structure, for example:
 - \b Driver_GPIO0 is the name of the access struct of the first instance (no. 0).
 - \b Driver_GPIO1 is the name of the access struct of the second instance (no. 1).

A middleware configuration setting allows connecting the middleware to a specific driver instance \b %Driver_GPIO<i>n</i>.
The default is \token{0}, which connects a middleware to the first instance of a driver.
**************************************************************************************************************************/

/**
\typedef    ARM_GPIO_DIRECTION
\details
Specifies values for setting the direction.

<b>Parameter for:</b>
  - \ref ARM_GPIO_SetDirection
*****************************************************************************************************************/

/**
\typedef    ARM_GPIO_OUTPUT_MODE
\details
Specifies values for setting the output mode.

<b>Parameter for:</b>
  - \ref ARM_GPIO_SetOutputMode
*****************************************************************************************************************/

/**
\typedef    ARM_GPIO_PULL_RESISTOR
\details
Specifies values for setting the pull resistor.

<b>Parameter for:</b>
  - \ref ARM_GPIO_SetPullResistor
*****************************************************************************************************************/

/**
\typedef    ARM_GPIO_EVENT_TRIGGER
\details
Specifies values for setting the event trigger.

<b>Parameter for:</b>
  - \ref ARM_GPIO_SetEventTrigger
*****************************************************************************************************************/

/**
\typedef    ARM_GPIO_SignalEvent_t
\details
Provides the typedef for the callback function \ref ARM_GPIO_SignalEvent.

<b>Parameter for:</b>
  - \ref ARM_GPIO_Setup
*******************************************************************************************************************/

/**
\defgroup gpio_execution_status GPIO Status Error Codes
\ingroup gpio_interface_gr
\brief Negative values indicate errors (GPIO has specific codes in addition to common \ref execution_status). 
\details 
The GPIO driver has additional status error codes that are listed below.
Note that the GPIO driver also returns the common \ref execution_status. 
  
@{
\def ARM_GPIO_ERROR_PIN
The \b pin specified is not available.
@}
*/

/**
\defgroup GPIO_events GPIO Events
\ingroup gpio_interface_gr
\brief The GPIO driver generates call back events that are notified via the function \ref ARM_GPIO_SignalEvent.
\details 
This section provides the event values for the \ref ARM_GPIO_SignalEvent callback function.

The following call back notification events are generated:
@{
\def  ARM_GPIO_EVENT_RISING_EDGE
\def  ARM_GPIO_EVENT_FALLING_EDGE
\def  ARM_GPIO_EVENT_EITHER_EDGE
@}
*/

//
//  Functions
//

int32_t ARM_GPIO_Setup (ARM_GPIO_Pin_t pin, ARM_GPIO_SignalEvent_t cb_event) {
  return ARM_DRIVER_OK;
}
/**
\fn int32_t ARM_GPIO_Setup (ARM_GPIO_Pin_t pin, ARM_GPIO_SignalEvent_t cb_event)
\details
The function \b ARM_GPIO_Setup sets-up the specified \em pin as GPIO with default configuration.
Pin is configured as input without pull-resistor and without event trigger.

The parameter \em cb_event specifies a pointer to the \ref ARM_GPIO_SignalEvent callback function to register. 
Use a NULL pointer when no callback events are required or to deregister a callback function.
**************************************************************************************************************************/

int32_t ARM_GPIO_SetDirection (ARM_GPIO_Pin_t pin, ARM_GPIO_DIRECTION direction) {
  return ARM_DRIVER_OK;
}
/**
\fn int32_t ARM_GPIO_SetDirection (ARM_GPIO_Pin_t pin, ARM_GPIO_DIRECTION direction)
\details
The function \b ARM_GPIO_SetDirection configures the direction of the specified \em pin.

Direction is specified with parameter \em direction:
 - \ref ARM_GPIO_INPUT : Input (default),
 - \ref ARM_GPIO_OUTPUT : Output.
**************************************************************************************************************************/

int32_t ARM_GPIO_SetOutputMode (ARM_GPIO_Pin_t pin, ARM_GPIO_OUTPUT_MODE mode) {
  return ARM_DRIVER_OK;
}
/**
\fn int32_t ARM_GPIO_SetOutputMode (ARM_GPIO_Pin_t pin, ARM_GPIO_OUTPUT_MODE mode)
\details
The function \b ARM_GPIO_SetOutputMode configures the output mode of the specified \em pin.

Output mode is specified with parameter \em mode:
 - \ref ARM_GPIO_PUSH_PULL : Push-pull (default),
 - \ref ARM_GPIO_OPEN_DRAIN : Open-drain.

\note Output mode is relevant only when the pin is configured as output.
**************************************************************************************************************************/

int32_t ARM_GPIO_SetPullResistor (ARM_GPIO_Pin_t pin, ARM_GPIO_PULL_RESISTOR resistor) {
  return ARM_DRIVER_OK;
}
/**
\fn int32_t ARM_GPIO_SetPullResistor (ARM_GPIO_Pin_t pin, ARM_GPIO_PULL_RESISTOR resistor)
\details
The function \b ARM_GPIO_SetPullResistor configures the pull resistor of the specified \em pin.

Pull resistor is specified with parameter \em resistor:
 - \ref ARM_GPIO_PULL_NONE : None (default),
 - \ref ARM_GPIO_PULL_UP : Pull-up,
 - \ref ARM_GPIO_PULL_DOWN : Pull-down.

\note Pull resistor applies to the pin regardless of pin direction.
**************************************************************************************************************************/

int32_t ARM_GPIO_SetEventTrigger (ARM_GPIO_Pin_t pin, ARM_GPIO_EVENT_TRIGGER trigger) {
  return ARM_DRIVER_OK;
}
/**
\fn int32_t ARM_GPIO_SetEventTrigger (ARM_GPIO_Pin_t pin, ARM_GPIO_EVENT_TRIGGER trigger)
\details
The function \b ARM_GPIO_SetEventTrigger configures the event trigger of the specified \em pin.

Event trigger is specified with parameter \em trigger:
 - \ref ARM_GPIO_TRIGGER_NONE : None (default),
 - \ref ARM_GPIO_TRIGGER_RISING_EDGE : Rising-edge,
 - \ref ARM_GPIO_TRIGGER_FALLING_EDGE : Falling-edge,
 - \ref ARM_GPIO_TRIGGER_EITHER_EDGE : Either edge (rising and falling).

\note To disable event trigger use trigger parameter \ref ARM_GPIO_TRIGGER_NONE.
**************************************************************************************************************************/

void ARM_GPIO_SetOutput (ARM_GPIO_Pin_t pin, uint32_t val) {
}
/**
\fn void ARM_GPIO_SetOutput (ARM_GPIO_Pin_t pin, uint32_t val)
\details
The function \b ARM_GPIO_SetOutput sets the level of the specified \em pin defined as output to the value specified by \em val.

\note
When a pin is configured as input, the level is latched and will be driven once the pin is configured as output.
**************************************************************************************************************************/

uint32_t ARM_GPIO_GetInput (ARM_GPIO_Pin_t pin) {
  return 0U;
}
/**
\fn uint32_t ARM_GPIO_GetInput (ARM_GPIO_Pin_t pin)
\details
The function \b ARM_GPIO_GetInput reads the level of the specified \em pin.
**************************************************************************************************************************/

void ARM_GPIO_SignalEvent (ARM_GPIO_Pin_t pin, uint32_t event) {
}
/**
\fn void ARM_GPIO_SignalEvent (ARM_GPIO_Pin_t pin, uint32_t event)
\details
The function \b ARM_GPIO_SignalEvent is a callback functions registered by the function \ref ARM_GPIO_Setup. 
It is called by the GPIO driver to notify the application about \ref GPIO_events occurred during operation.

The parameter \em pin indicates on which pin the event occurred and parameter \em event indicates one or more events that occurred.

The following events can be generated:

<table class="cmtable" summary="">
<tr>
  <th> Parameter \em event              </th><th> Bit </th><th> Description </th>
</tr>
<tr>
  <td> \ref ARM_GPIO_EVENT_RISING_EDGE  </td><td>  0  </td><td> Occurs when rising-edge is detected on the indicated pin. </td>
</tr>
<tr>
  <td> \ref ARM_GPIO_EVENT_FALLING_EDGE </td><td>  1  </td><td> Occurs when falling-edge is detected on the indicated pin. </td>
</tr>
<tr>
  <td> \ref ARM_GPIO_EVENT_EITHER_EDGE  </td><td>  2  </td><td> Occurs when either edge is detected on the indicated pin 
                                                                when trigger is configured as \ref ARM_GPIO_TRIGGER_EITHER_EDGE 
                                                                and hardware is not able to distinguish between rising and falling edge. </td>
</tr>
</table>
**************************************************************************************************************************/

/**
@}
*/
// End GPIO Interface
