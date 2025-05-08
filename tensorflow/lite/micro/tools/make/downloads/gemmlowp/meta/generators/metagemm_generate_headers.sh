#!/bin/bash
python streams_arm_32.py > ../streams_arm_32.h
python streams_arm_64.py > ../streams_arm_64.h
python quantized_mul_kernels_arm_32.py > ../quantized_mul_kernels_arm_32.h
python quantized_mul_kernels_arm_64.py > ../quantized_mul_kernels_arm_64.h
python transform_kernels_arm_32.py > ../transform_kernels_arm_32.h
python transform_kernels_arm_64.py > ../transform_kernels_arm_64.h

