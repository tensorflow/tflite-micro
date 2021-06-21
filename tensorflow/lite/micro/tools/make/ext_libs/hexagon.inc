# Full path to the hexagon_tflm static library.
HEXAGON_TFLM_LIB :=

ifeq ($(HEXAGON_TFLM_LIB), )
  $(error HEXAGON_TFLM_LIB must be specified on the command line)
endif

MICROLITE_LIBS += $(HEXAGON_TFLM_LIB)
