FROM  --platform=linux/amd64 ubuntu:22.04

SHELL ["/bin/bash", "-c"]

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && \
    apt-get -y install \
        build-essential \
        curl \
        gdb \
        gh \
        git \
        less \
        libncurses5 \
        libtinfo5 \
        llvm-15-tools \
        locales \
        nano \
        python3 \
        python3-pip \
        python-is-python3 \
        software-properties-common \
        sudo \
        unzip && \
    ln -s /usr/bin/FileCheck-15 /usr/bin/FileCheck

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y install libpython3.9

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen

RUN pip install \
        lit \
        python-matrix-runner

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN mkdir -p /workspaces && \
    chown $USER_UID:$USER_GID /workspaces

ADD vcpkg-configuration.json /home/
ADD postCreateCommand.sh /home/

RUN chmod +x /home/postCreateCommand.sh

USER $USERNAME
WORKDIR /home/$USERNAME

CMD ["/bin/bash"]
