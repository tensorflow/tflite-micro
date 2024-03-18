# This docker container can be used to run all the TFLM CI checks.
#
# It is only used as part of the GitHub workflows to test for code-style. But
# the container is available and ready for use to run all the checks locally,
# in case that is useful for debugging. See all the versions at
# https://github.com/users/TFLM-bot/packages/container/tflm-ci/versions
#
# docker pull ghcr.io/tflm-bot/tflm-ci:<version>
#
# Build you own container with:
# docker build -f ci/Dockerfile.micro -t tflm-ci .
#
# Use a prebuilt Python image instead of base Ubuntu to speed up the build process,
# since it has all the build dependencies we need for Micro and downloads much faster
# than the install process.

# Using a multistage build so that the build tools required for stage 1 don't make the
# CI container unnecessarily large.
FROM python:3.10-bullseye AS qemu_builder
RUN apt-get update

RUN apt-get install -y ninja-build wget xz-utils
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y build-essential libcairo2-dev libpango1.0-dev libjpeg-dev libgif-dev librsvg2-dev

COPY ci/install_qemu.sh ./
# Installs all built files into /qemu_install rather than /usr/local default.
RUN ./install_qemu.sh /tmp/qemu_install

# This stage is the final CI container.
FROM python:3.10-bullseye as tflm-ci

RUN apt-get update

RUN apt-get install -y zip xxd sudo

RUN apt install -y lsb-release wget software-properties-common gnupg
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 16
RUN ln -s /usr/bin/clang-16 /usr/bin/clang
RUN ln -s /usr/bin/clang++-16 /usr/bin/clang++

RUN apt-get install clang-format-16
RUN ln -s /usr/bin/clang-format-16 /usr/bin/clang-format

# Needed when using the Dockerfile locally.
RUN git config --global --add safe.directory /opt/tflm

# Needed when the docker container is used with GitHub actions.
RUN git config --global --add safe.directory /github/workspace

# Install yapf to check for Python formatting as part of the TFLM continuous
# integration.
RUN pip install yapf==0.32.0

# Pillow was added first for the C array generation as a result of the following
# PRs:
# https://github.com/tensorflow/tflite-micro/pull/337
# https://github.com/tensorflow/tflite-micro/pull/410
RUN pip install Pillow

# necessary bits for create_size_log scripts
RUN pip install pandas
RUN pip install matplotlib
RUN pip install six

# Install Renode test dependencies
RUN pip install pyyaml requests psutil robotframework==4.0.1

# Install QEMU from build container qemu_builder into tflm-ci container.
# We're using a two stage build to keep the CI container smaller.
WORKDIR /usr/local
# Merge built files into /usr/local so that the path is already setup.
COPY --from=qemu_builder /tmp/qemu_install/. .

WORKDIR /
COPY ci/*.sh /install/
RUN /install/install_bazelisk.sh
RUN /install/install_buildifier.sh
