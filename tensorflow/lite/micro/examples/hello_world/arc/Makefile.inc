ifeq ($(TARGET_ARCH), arc)
ifneq ($(filter $(ARC_TAGS), mli20_experimental), )

# Patch of arc make project to adjust it specifically for hello_world example. 
# In particular:
# - Adapt the model to be used with MLI by transposing weights tensors

  HELLO_WORLD_HDRS += \
  hello_world_patch.txt
  
  HELLO_WORLD_TEST_HDRS += \
  hello_world_patch.txt
  
  MODEL_PATH = /tensorflow/lite/micro/examples/hello_world

# Apply changes in generated project files. 
# See related comment echoed (@echo <comment>) after each change 
# to get understanding on it's purpose.
%/hello_world_patch.txt: %$(MODEL_PATH)/model.cc

	@python tensorflow/lite/micro/tools/make/targets/arc/adaptation_tool.py $<
	@echo Makefile: Adapted the model to be used with MLI >> $@

endif
endif
