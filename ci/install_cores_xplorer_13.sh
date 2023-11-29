#!/bin/bash

mkdir /opt/xtensa/licenses

##############
#  Fusion F1
##############
cd /opt/xtensa/
tar xvzf F1_190305_swupgrade_linux.tgz --dir /opt/xtensa/licenses/
cd /opt/xtensa/licenses/RI-2020.4-linux/F1_190305_swupgrade/

./install --xtensa-tools \
  /opt/xtensa/XtDevTools/install/tools/RI-2020.4-linux/XtensaTools/ \
  --no-default \
  --no-replace

##############
#  Vision P6
##############
cd /opt/xtensa/
tar xvzf P6_200528_linux.tgz --dir /opt/xtensa/licenses/
cd /opt/xtensa/licenses/RI-2020.4-linux/P6_200528/

./install --xtensa-tools \
  /opt/xtensa/XtDevTools/install/tools/RI-2020.4-linux/XtensaTools/ \
  --no-default \
  --no-replace

##############
#  Hifi3Z
##############
cd /opt/xtensa/
tar xvzf HIFI_190304_swupgrade_linux.tgz --dir /opt/xtensa/licenses/
cd /opt/xtensa/licenses/RI-2020.4-linux/HIFI_190304_swupgrade/

./install --xtensa-tools \
  /opt/xtensa/XtDevTools/install/tools/RI-2020.4-linux/XtensaTools/ \
  --no-default \
  --no-replace
