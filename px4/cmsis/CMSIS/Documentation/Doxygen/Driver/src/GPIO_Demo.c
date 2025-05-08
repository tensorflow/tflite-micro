#include "Driver_GPIO.h"

/* GPIO driver instance */
extern ARM_DRIVER_GPIO             Driver_GPIO0;
static ARM_DRIVER_GPIO *GPIOdrv = &Driver_GPIO0;

/* Pin mapping */
#define GPIO_PIN0       0U
#define GPIO_PIN1       1U
#define GPIO_PIN2       2U
#define GPIO_PIN3       3U

/* GPIO Signal Event callback */
static void GPIO_SignalEvent (ARM_GPIO_Pin_t pin, uint32_t event) {

  switch (pin) {
    case GPIO_PIN1:
      /* Events on pin GPIO_PIN1 */
      if (event & ARM_GPIO_EVENT_RISING_EDGE) {
        /* Rising-edge detected */
      }
      if (event & ARM_GPIO_EVENT_FALLING_EDGE) {
        /* Falling-edge detected */
      }
      break;
  }
}

/* Get GPIO Input 0 */
uint32_t GPIO_GetInput0 (void) {
  return (GPIOdrv->GetInput(GPIO_PIN0));
}

/* Get GPIO Input 1 */
uint32_t GPIO_GetInput1 (void) {
  return (GPIOdrv->GetInput(GPIO_PIN1));
}

/* Set GPIO Output Pin 2 */
void GPIO_SetOutput2 (uint32_t val) {
  GPIOdrv->SetOutput(GPIO_PIN2, val);
}

/* Set GPIO Output Pin 3 */
void GPIO_SetOutput3 (uint32_t val) {
  GPIOdrv->SetOutput(GPIO_PIN3, val);
}

/* Setup GPIO pins */
void GPIO_Setup (void) {

  /* Pin GPIO_PIN0: Input */
  GPIOdrv->Setup          (GPIO_PIN0, NULL);
  GPIOdrv->SetDirection   (GPIO_PIN0, ARM_GPIO_INPUT);

  /* Pin GPIO_PIN1: Input with trigger on rising and falling edge */
  GPIOdrv->Setup          (GPIO_PIN1, GPIO_SignalEvent);
  GPIOdrv->SetDirection   (GPIO_PIN1, ARM_GPIO_INPUT);
  GPIOdrv->SetEventTrigger(GPIO_PIN1, ARM_GPIO_TRIGGER_EITHER_EDGE);

  /* Pin GPIO_PIN2: Output push-pull (initial level 0) */
  GPIOdrv->Setup          (GPIO_PIN2, NULL);
  GPIOdrv->SetOutput      (GPIO_PIN2, 0U);
  GPIOdrv->SetDirection   (GPIO_PIN2, ARM_GPIO_OUTPUT);

  /* Pin GPIO_PIN3: Output open-drain with pull-up resistor (initial level 1) */
  GPIOdrv->Setup          (GPIO_PIN3, NULL);
  GPIOdrv->SetPullResistor(GPIO_PIN3, ARM_GPIO_PULL_UP);
  GPIOdrv->SetOutputMode  (GPIO_PIN3, ARM_GPIO_OPEN_DRAIN);
  GPIOdrv->SetOutput      (GPIO_PIN3, 1U);
  GPIOdrv->SetDirection   (GPIO_PIN3, ARM_GPIO_OUTPUT);
}
