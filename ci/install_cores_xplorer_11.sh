#!/bin/bash

mkdir /opt/xtensa/licenses

mkdir -p /opt/xtensa/XtDevTools/install/tools/
tar xvzf XtensaTools_RI_2022_9_linux.tgz --dir /opt/xtensa/XtDevTools/install/tools/


###########
#  Hifimini
###########
cd /opt/xtensa/
tar xvzf mini1m1m_RI_2019_2_linux_w_keys.tgz --dir /opt/xtensa/licenses/
cd /opt/xtensa/licenses/RI-2019.2-linux/mini1m1m_RG/

./install --xtensa-tools \
  /opt/xtensa/XtDevTools/install/tools/RI-2019.2-linux/XtensaTools/ \
  --no-default \
  --no-replace
