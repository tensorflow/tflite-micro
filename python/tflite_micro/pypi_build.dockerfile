# Use the Python Packaging Authority's reference build environment
# for binary extensions. Binary extensions are typically built and distributed
# for each target Python version and OS platform. The reference build
# environment contains Python installations for each version, and a C/C++
# toolchain specified for maximum compatibility among x86_64 Linux paltforms.
FROM quay.io/pypa/manylinux_2_28_x86_64

# Install bazel (via bazelisk)
ENV BAZELISK=https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64
ENV BAZEL=/usr/local/bin/bazel
RUN curl --output $BAZEL --location $BAZELISK && chmod 755 $BAZEL

# Append the location of the C/C++ toolchain to the default PATH, where
# bazel expects to find it. The reference environment provides the location
# (typically somewhere under /opt) in DEVTOOLSET_ROOTPATH.
RUN echo "PATH="${PATH}:/${DEVTOOLSET_ROOTPATH}"" >>/etc/environment
