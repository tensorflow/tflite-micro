LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_MODULE := correctness_meta_gemm
LOCAL_SRC_FILES := ../test/correctness_meta_gemm.cc

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_MODULE := benchmark_meta_gemm
LOCAL_CFLAGS := -DNDEBUG -DGEMMLOWP_USE_META_FASTPATH
LOCAL_SRC_FILES := ../test/benchmark_meta_gemm.cc ../eight_bit_int_gemm/eight_bit_int_gemm.cc

include $(BUILD_EXECUTABLE)

include $(CLEAR_VARS)

LOCAL_ARM_NEON := true
LOCAL_MODULE := benchmark
LOCAL_SRC_FILES := ../test/benchmark.cc

include $(BUILD_EXECUTABLE)
