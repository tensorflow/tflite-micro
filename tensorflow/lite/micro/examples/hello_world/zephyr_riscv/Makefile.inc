ifeq ($(TARGET), zephyr_vexriscv)
	export ZEPHYR_TOOLCHAIN_VARIANT?=zephyr
	export TOOLCHAIN_BASE=${ZEPHYR_SDK_INSTALL_DIR}/riscv64-zephyr-elf/riscv64-zephyr-elf
	export TOOLCHAIN_VERSION=9.2.0
	PROJECT_INCLUDES += ${CURDIR} ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION} ${TOOLCHAIN_BASE}/include ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION}/riscv64-zephyr-elf/rv32i/ilp32
	ZEPHYR_HELLO_WORLD_SRCS = \
tensorflow/lite/micro/examples/hello_world/zephyr_riscv/src/assert.cc \
tensorflow/lite/micro/examples/hello_world/main.cc \
tensorflow/lite/micro/examples/hello_world/main_functions.cc \
tensorflow/lite/micro/examples/hello_world/constants.cc \
tensorflow/lite/micro/examples/hello_world/output_handler.cc \
tensorflow/lite/micro/examples/hello_world/hello_world_model_data.cc \
prj.conf

$(eval $(call generate_project,cmake,zephyr_cmake_project.cmake,hello_world,$(MICROLITE_CC_SRCS) $(MICROLITE_CC_KERNEL_SRCS) $(THIRD_PARTY_CC_SRCS) $(ZEPHYR_HELLO_WORLD_SRCS) $(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(HELLO_WORLD_HDRS),,$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS),$(CCFLAGS),))

$(PRJDIR)hello_world/cmake/CMakeLists.txt: $(PRJDIR)hello_world/cmake/zephyr_cmake_project.cmake
	@sed -E 's#\%\{INCLUDE_DIRS\}\%#$(PROJECT_INCLUDES)#g' $< > $@

hello_world_pregen_arrays:
	@python3 tensorflow/lite/micro/tools/generate_cc_arrays.py ${PRJDIR}hello_world/cmake/ ${HELLO_WORLD_GENERATOR_INPUTS}
    $(info Finished hello_world_pregen_arrays)

#We are skipping here copy of `zephyr` third_party repository
#To compile standalone project ZEPHYR_BASE enviroment variable should be set
hello_world_generate_project:generate_hello_world_cmake_project $(PRJDIR)hello_world/cmake/CMakeLists.txt
	( \
	  . ${ZEPHYR_BASE}/venv-zephyr/bin/activate; \
	  cmake -B${GENDIR}hello_world/build -DBOARD="litex_vexriscv" -H${PRJDIR}hello_world/cmake/ -DPython_ROOT_DIR=${ZEPHYR_BASE}/venv-zephyr/bin/; \
	  make -C ${GENDIR}hello_world/build; \
	)

hello_world_bin:hello_world_pregen_arrays hello_world_generate_project

endif
