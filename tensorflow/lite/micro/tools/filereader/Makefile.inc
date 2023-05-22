XTENSA_FILE_READER_ROOT_DIR := tensorflow/lite/micro/tools/filereader

XTENSA_FILE_READER_SRCS := $(XTENSA_FILE_READER_ROOT_DIR)/xtensa_file_reader.cc

XTENSA_FILE_READER_HDRS := $(DOWNLOADS_DIR)/flatbuffers/include/flatbuffers/util.h

$(eval $(call microlite_test,xtensa_file_reader,\
$(XTENSA_FILE_READER_SRCS),$(XTENSA_FILE_READER_HDRS),))