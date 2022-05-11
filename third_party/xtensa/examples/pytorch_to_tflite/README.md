# Setup Xtensa Tools
$ set path = ( ~/xtensa/XtDevTools/install/tools/RI-2020.5-linux/XtensaTools/bin $path )

$ set path = ( ~/xtensa/XtDevTools/install/tools/RI-2020.5-linux/XtensaTools/Tools/bin $path )

$ setenv XTENSA_SYSTEM ~xtensa/XtDevTools/install/tools/RI-2020.5-linux/XtensaTools/config

$ setenv XTENSA_CORE AE_HiFi5_LE5_AO_FP_XC

$ setenv XTENSA_TOOLS_VERSION RI-2020.5-linux

$ setenv XTENSA_BASE ~/xtensa/XtDevTools/install/


# Clean and build mobilenet_v2 model on TFLM
$ make -f tensorflow/lite/micro/tools/make/Makefile clean

$ make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa OPTIMIZED_KERNEL_DIR=xtensa TARGET=xtensa TARGET_ARCH=hifi5 test_pytorch_to_tflite_test -j
