ifeq ($(TARGET), zephyr_vexriscv)
	export ZEPHYR_TOOLCHAIN_VARIANT?=zephyr
	export TOOLCHAIN_BASE=${ZEPHYR_SDK_INSTALL_DIR}/riscv64-zephyr-elf/riscv64-zephyr-elf
	export TOOLCHAIN_VERSION=9.2.0
	PROJECT_INCLUDES += ${CURDIR} ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION} ${TOOLCHAIN_BASE}/include ${TOOLCHAIN_BASE}/include/c++/${TOOLCHAIN_VERSION}/riscv64-zephyr-elf/rv32i/ilp32
	ZEPHYR_model_runner_SRCS = \
tensorflow/lite/micro/examples/model_runner/zephyr_riscv/src/assert.cc \
tensorflow/lite/micro/examples/model_runner/main.cc \
tensorflow/lite/micro/examples/model_runner/main_functions.cc \
tensorflow/lite/micro/examples/model_runner/model.cc \
tensorflow/lite/micro/examples/model_runner/output_handler.cc \
boards/litex_vexriscv.overlay \
prj.conf

$(eval $(call generate_project,cmake,zephyr_cmake_project.cmake,model_runner,$(MICROLITE_CC_SRCS) $(THIRD_PARTY_CC_SRCS) $(ZEPHYR_model_runner_SRCS) $(MICROLITE_CC_HDRS) $(THIRD_PARTY_CC_HDRS) $(model_runner_HDRS),,$(LDFLAGS) $(MICROLITE_LIBS),$(CXXFLAGS),$(CCFLAGS),))

$(PRJDIR)model_runner/cmake/CMakeLists.txt: $(PRJDIR)model_runner/cmake/zephyr_cmake_project.cmake
	@sed -E 's#\%\{INCLUDE_DIRS\}\%#$(PROJECT_INCLUDES)#g' $< > $@

#We are skipping here copy of `zephyr` third_party repository
#To compile standalone project ZEPHYR_BASE enviroment variable should be set
model_runner_bin: generate_model_runner_cmake_project $(PRJDIR)model_runner/cmake/CMakeLists.txt
	( \
	  . ${ZEPHYR_BASE}/venv-zephyr/bin/activate; \
	  cmake -B${GENDIR}model_runner/build -DBOARD="litex_vexriscv" -H${PRJDIR}model_runner/cmake/ -DPython_ROOT_DIR=${ZEPHYR_BASE}/venv-zephyr/bin/; \
	  make -C ${GENDIR}model_runner/build; \
	)


#We are skipping here copy of `zephyr` third_party repository
#To compile standalone project ZEPHYR_BASE enviroment variable should be set
test_model_runner: model_runner_bin
	( \
	  ${TENSORFLOW_ROOT}/tensorflow/lite/micro/testing/test_zephyr_vexriscv_binary.sh; \
	)
endif
