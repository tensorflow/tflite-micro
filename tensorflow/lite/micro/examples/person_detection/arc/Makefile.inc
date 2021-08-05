ifeq ($(TARGET_ARCH), arc)
ifneq ($(filter $(ARC_TAGS), mli20_experimental), )

# Patch of arc make project to adjust it specifically
# for experimental person detection example.
# In particular:
# - Adapt the model to be used with MLI by transposing weights tensors

  person_detection_HDRS += \
  person_detection_int8_patch.txt
  
  person_detection_TEST_HDRS += \
  person_detection_int8_patch.txt
  
  MODEL_PATH = /tensorflow/lite/micro/tools/make/downloads/person_model_int8

# Apply changes in generated project files. 
# See related comment echoed (@echo <comment>) after each change 
# to get understanding on it's purpose.
%/person_detection_int8_patch.txt: %$(MODEL_PATH)/person_detect_model_data.cc

	@python tensorflow/lite/micro/tools/make/targets/arc/adaptation_tool.py $<
	@echo Makefile: Adapted the model to be used with MLI >> $@

endif
endif
