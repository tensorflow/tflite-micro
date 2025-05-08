NDK_TOOLCHAIN_VERSION := clang
APP_STL := gnustl_static
APP_ABI := armeabi-v7a
APP_CPPFLAGS := -std=c++11 -Wall -Wextra -pedantic -Wno-unused-variable -Wno-unused-parameter
APP_LDFLAGS := -L$(SYSROOT)/usr/lib -lstdc++ -latomic
APP_PIE := true
