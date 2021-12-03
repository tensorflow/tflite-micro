<!-- mdformat off(b/169948621#comment2) -->

# Micro Speech Example

This example shows how to run a 20 kB model that can recognize 2 keywords,
"yes" and "no", from speech data.

The application listens to its surroundings with a microphone and indicates
when it has detected a word by lighting an LED or displaying data on a
screen, depending on the capabilities of the device.

![Animation on Arduino](images/animation_on_arduino.gif)

The code has a small footprint (for example, around 22 kilobytes on a Cortex
M3) and only uses about 10 kilobytes of RAM for working memory, so it's able to
run on systems like an STM32F103 with only 20 kilobytes of total SRAM and 64
kilobytes of Flash.

## Table of contents

-   [Deploy to STM32F746](#deploy-to-STM32F746)
-   [Deploy to NXP FRDM K66F](#deploy-to-nxp-frdm-k66f)
-   [Deploy to CEVA BX1/SP500](#deploy-to-ceva-bx1)
-   [Run on macOS](#run-on-macos)
-   [Run the tests on a development machine](#run-the-tests-on-a-development-machine)
-   [Train your own model](#train-your-own-model)

## Deploy to STM32F746

The following instructions will help you build and deploy the example to the
[STM32F7 discovery kit](https://os.mbed.com/platforms/ST-Discovery-F746NG/)
using [ARM Mbed](https://github.com/ARMmbed/mbed-cli).

Before we begin, you'll need the following:

- STM32F7 discovery kit board
- Mini-USB cable
- ARM Mbed CLI ([installation instructions](https://os.mbed.com/docs/mbed-os/v6.9/quick-start/build-with-mbed-cli.html). Check it out for MacOS Catalina - [mbed-cli is broken on MacOS Catalina #930](https://github.com/ARMmbed/mbed-cli/issues/930#issuecomment-660550734))
- Python 3 and pip3

Since Mbed requires a special folder structure for projects, we'll first run a
command to generate a subfolder containing the required source files in this
structure:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=disco_f746ng OPTIMIZED_KERNEL_DIR=cmsis_nn generate_micro_speech_mbed_project
```

Running the make command will result in the creation of a new folder:

```
tensorflow/lite/micro/tools/make/gen/disco_f746ng_cortex-m4_default/prj/micro_speech/mbed
```

This folder contains all of the example's dependencies structured in the correct
way for Mbed to be able to build it.

Change into the directory and run the following commands.

First, tell Mbed that the current directory is the root of an Mbed project:

```
mbed config root .
```

Next, tell Mbed to download the dependencies and prepare to build:

```
mbed deploy
```

Older versions of Mbed will build the project using C++98. However, TensorFlow Lite
requires C++11. If needed, run the following Python snippet to modify the Mbed
configuration files so that it uses C++11:

```
python -c 'import fileinput, glob;
for filename in glob.glob("mbed-os/tools/profiles/*.json"):
  for line in fileinput.input(filename, inplace=True):
    print(line.replace("\"-std=gnu++98\"","\"-std=c++11\", \"-fpermissive\""))'
```

Note: Mbed has a dependency to an old version of arm_math.h and cmsis_gcc.h (adapted from the general [CMSIS-NN MBED example](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/cmsis_nn#example-2---mbed)). Therefore you need to copy the newer version as follows:
```bash
cp tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/DSP/Include/\
arm_math.h mbed-os/cmsis/TARGET_CORTEX_M/arm_math.h
cp tensorflow/lite/micro/tools/make/downloads/cmsis/CMSIS/Core/Include/\
cmsis_gcc.h mbed-os/cmsis/TARGET_CORTEX_M/cmsis_gcc.h
```

Finally, run the following command to compile:

```
mbed compile -m DISCO_F746NG -t GCC_ARM
```

This should result in a binary at the following path:

```
./BUILD/DISCO_F746NG/GCC_ARM/mbed.bin
```

To deploy, plug in your STM board and copy the file to it. On macOS, you can do
this with the following command:

```
cp ./BUILD/DISCO_F746NG/GCC_ARM/mbed.bin /Volumes/DIS_F746NG/
```

Copying the file will initiate the flashing process.

The inference results are logged by the board while the program is running.
To view it, establish a serial connection to the board
using a baud rate of `9600`. On OSX and Linux, the following command should
work, replacing `/dev/tty.devicename` with the name of your device as it appears
in `/dev`:

```
screen /dev/tty.devicename 9600
```

You will see a line output for every word that is detected:

```
Heard yes (201) @4056ms
Heard no (205) @6448ms
Heard unknown (201) @13696ms
Heard yes (205) @15000ms
```

The number after each detected word is its score. By default, the program only
considers matches as valid if their score is over 200, so all of the scores you
see will be at least 200.

To stop viewing the debug output with `screen`, hit `Ctrl+A`, immediately
followed by the `K` key, then hit the `Y` key.

## Deploy to NXP FRDM K66F

The following instructions will help you build and deploy the example to the
[NXP FRDM K66F](https://www.nxp.com/design/development-boards/freedom-development-boards/mcu-boards/freedom-development-platform-for-kinetis-k66-k65-and-k26-mcus:FRDM-K66F)
using [ARM Mbed](https://github.com/ARMmbed/mbed-cli).

1.  Download
    [the TensorFlow source code](https://github.com/tensorflow/tensorflow).
2.  Follow instructions from
    [mbed website](https://os.mbed.com/docs/mbed-os/v5.13/tools/installation-and-setup.html)
    to setup and install mbed CLI.
3.  Compile TensorFlow with the following command to generate mbed project:

    ```
    make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mbed TAGS="nxp_k66f" generate_micro_speech_mbed_project
    ```

4.  Change into the following directory that has been generated:
    `tensorflow/lite/micro/tools/make/gen/mbed_cortex-m4/prj/micro_speech/mbed`

5.  Create an Mbed project using the generated files, run ensuring your
    environment is using Python 2.7: `mbed config root .`

6.  Next, tell Mbed to download the dependencies and prepare to build: `mbed
    deploy`

7.  Finally, we can run the following command to compile the code: `mbed compile
    -m K66F -t GCC_ARM`

8.  For some Mbed compilers (such as GCC), you may get compile error in
    mbed_rtc_time.cpp. Go to `mbed-os/platform/mbed_rtc_time.h` and comment line
    32 and line 37:

    ```
    //#if !defined(__GNUC__) || defined(__CC_ARM) || defined(__clang__)
    struct timeval {
    time_t tv_sec;
    int32_t tv_usec;
    };
    //#endif
    ```

9.  If your system does not recognize the board with the `mbed detect` command.
    Follow the instructions for setting up
    [DAPLink](https://armmbed.github.io/DAPLink/?board=FRDM-K66F) for the
    [K66F](https://os.mbed.com/platforms/FRDM-K66F/).

10. Connect the USB cable to the micro USB port. When the Ethernet port is
    facing towards you, the micro USB port is left of the Ethernet port.

11. To compile and flash in a single step, add the `--flash` option:

    ```
    mbed compile -m K66F -t GCC_ARM --flash
    ```

12. Disconnect USB cable from the device to power down the device and connect
    back the power cable to start running the model.

13. Connect to serial port with baud rate of 9600 and correct serial device to
    view the output from the MCU. In linux, you can run the following screen
    command if the serial device is `/dev/ttyACM0`:

    ```
    sudo screen /dev/ttyACM0 9600
    ```

14. Saying "Yes" will print "Yes" and "No" will print "No" on the serial port.

15. A loopback path from microphone to headset jack is enabled. Headset jack is
    in black color. If there is no output on the serial port, you can connect
    headphone to headphone port to check if audio loopback path is working.

## Deploy to CEVA-BX1

The following instructions will help you build and deploy the sample to the
[CEVA-BX1](https://www.ceva-dsp.com/product/ceva-bx1-sound/) or [CEVA-SP500](https://www.ceva-dsp.com/product/ceva-senspro/)

1.  Contact CEVA at [sales@ceva-dsp.com](mailto:sales@ceva-dsp.com)
2.  For BX1:
2.1. Download and install CEVA-BX Toolbox v18.0.2
2.2.  Set the TARGET_TOOLCHAIN_ROOT variable in
    /tensorflow/lite/micro/tools/make/templates/ceva_bx1/ceva_app_makefile.tpl
    To your installation location. For example: TARGET_TOOLCHAIN_ROOT :=
    /home/myuser/work/CEVA-ToolBox/V18/BX
2.3.  Generate the Makefile for the project: /tensorflow$ make -f
    tensorflow/lite/micro/tools/make/Makefile TARGET=ceva TARGET_ARCH=CEVA_BX1
    generate_micro_speech_make_project
3. For SensPro (SP500):
3.1. Download and install CEVA-SP Toolbox v20
3.2. Set the TARGET_TOOLCHAIN_ROOT variable in
    /tensorflow/lite/micro/tools/make/templates/ceva_SP500/ceva_app_makefile.tpl
    To your installation location. For example: TARGET_TOOLCHAIN_ROOT :=
    /home/myuser/work/CEVA-ToolBox/V20/SensPro
3.3. Generate the Makefile for the project: /tensorflow$ make -f
    tensorflow/lite/micro/tools/make/Makefile TARGET=ceva TARGET_ARCH=CEVA_SP500
    generate_micro_speech_make_project 	
5.  Build the project:
    /tensorflow/lite/micro/tools/make/gen/ceva_bx1/prj/micro_speech/make$ make
6.  This should build the project and create a file called micro_speech.elf.
7.  The supplied configuration reads input from a files and expects a file
    called input.wav (easily changed in audio_provider.cc) to be placed in the
    same directory of the .elf file
8.  We used Google's speech command dataset: V0.0.2:
    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz V0.0.1:
    http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
9.  Follow CEVA Toolbox instructions for creating a debug target and running the
    project.
10. Output should look like: Heard silence (208) @352ms Heard no (201) @1696ms
    Heard yes (203) @3904ms

## Run on macOS

The example contains an audio provider compatible with macOS. If you have access
to a Mac, you can run the example on your development machine.

First, use the following command to build it:

```
make -f tensorflow/lite/micro/tools/make/Makefile micro_speech
```

Once the build completes, you can run the example with the following command:

```
tensorflow/lite/micro/tools/make/gen/osx_x86_64/bin/micro_speech
```

You might see a pop-up asking for microphone access. If so, grant it, and the
program will start.

Try saying "yes" and "no". You should see output that looks like the following:

```
Heard yes (201) @4056ms
Heard no (205) @6448ms
Heard unknown (201) @13696ms
Heard yes (205) @15000ms
Heard yes (205) @16856ms
Heard unknown (204) @18704ms
Heard no (206) @21000ms
```

The number after each detected word is its score. By default, the recognize
commands component only considers matches as valid if their score is over 200,
so all of the scores you see will be at least 200.

The number after the score is the number of milliseconds since the program was
started.

If you don't see any output, make sure your Mac's internal microphone is
selected in the Mac's *Sound* menu, and that its input volume is turned up high
enough.

## Run the tests on a development machine

To compile and test this example on a desktop Linux or macOS machine, download
[the TensorFlow source code](https://github.com/tensorflow/tensorflow), `cd`
into the source directory from a terminal, and then run the following command:

```
make -f tensorflow/lite/micro/tools/make/Makefile test_micro_speech_test
```

This will take a few minutes, and downloads frameworks the code uses like
[CMSIS](https://developer.arm.com/embedded/cmsis) and
[flatbuffers](https://google.github.io/flatbuffers/). Once that process has
finished, you should see a series of files get compiled, followed by some
logging output from a test, which should conclude with `~~~ALL TESTS PASSED~~~`.

If you see this, it means that a small program has been built and run that loads
the trained TensorFlow model, runs some example inputs through it, and got the
expected outputs.

To understand how TensorFlow Lite does this, you can look at the source in
[micro_speech_test.cc](micro_speech_test.cc).
It's a fairly small amount of code that creates an interpreter, gets a handle to
a model that's been compiled into the program, and then invokes the interpreter
with the model and sample inputs.

## Train your own model

So far you have used an existing trained model to run inference on
microcontrollers. If you wish to train your own model, follow the instructions
given in the [train/](train/) directory.
