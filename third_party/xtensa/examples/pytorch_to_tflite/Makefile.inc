ifeq ($(OPTIMIZED_KERNEL_DIR), xtensa)
   EXAMPLE_NAME:=pytorch_to_tflite
   PYTORCH_TO_TFLITE_TEST_SRCS := \
    $(TENSORFLOW_ROOT)third_party/xtensa/examples/$(EXAMPLE_NAME)/$(EXAMPLE_NAME)_test.cc \
    $(TENSORFLOW_ROOT)third_party/xtensa/examples/$(EXAMPLE_NAME)/pytorch_images_dog_jpg.cc \
    $(TENSORFLOW_ROOT)third_party/xtensa/examples/pytorch_to_tflite/pytorch_op_resolver.cc \

   PYTORCH_TO_TFLITE_GENERATOR_INPUTS := \
    $(TENSORFLOW_ROOT)third_party/xtensa/examples/pytorch_to_tflite/mobilenet_v2_quantized_1x3x224x224.tflite

   PYTORCH_TO_TFLITE_HDRS := \
    $(TENSORFLOW_ROOT)third_party/xtensa/examples/$(EXAMPLE_NAME)/pytorch_images_dog_jpg.h \
    $(TENSORFLOW_ROOT)third_party/xtensa/examples/pytorch_to_tflite/pytorch_op_resolver.h \


   ## Tests loading and running a mobilenet v2 model.
   ifneq ($(TARGET_ARCH), hifi5)
     $(eval $(call microlite_test,pytorch_to_tflite_test,\
     $(PYTORCH_TO_TFLITE_TEST_SRCS),$(PYTORCH_TO_TFLITE_HDRS),$(PYTORCH_TO_TFLITE_GENERATOR_INPUTS)))
   endif
endif

