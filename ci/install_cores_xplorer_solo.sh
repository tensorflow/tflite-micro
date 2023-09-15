#!/bin/bash

mkdir /opt/xtensa/licenses

mkdir -p /opt/xtensa/XtDevTools/install/tools/
tar xvzf XtensaTools_RI_2022_9_linux.tgz --dir /opt/xtensa/XtDevTools/install/tools/


###########
#  Hifi5
###########
cd /opt/xtensa/
tar xvzf PRD_H5_RDO_07_01_2022_linux.tgz --dir /opt/xtensa/licenses/
cd /opt/xtensa/licenses/RI-2022.9-linux/PRD_H5_RDO_07_01_2022/

./install --xtensa-tools \
  /opt/xtensa/XtDevTools/install/tools/RI-2022.9-linux/XtensaTools/ \
  --no-default \
  --no-replace
