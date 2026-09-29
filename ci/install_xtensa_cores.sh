#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

mkdir -p /opt/xtensa/licenses/RI-2022.9-linux
mkdir -p /opt/xtensa/XtDevTools/install/tools/

echo "Extracting XtensaTools RI-2022.9-linux..."
tar xzf XtensaTools_RI_2022_9_linux.tgz --dir /opt/xtensa/XtDevTools/install/tools/

XTENSA_TOOLS_DIR="/opt/xtensa/XtDevTools/install/tools/RI-2022.9-linux/XtensaTools"
mkdir -p "${XTENSA_TOOLS_DIR}/config"

install_core() {
  local core_archive=$1
  local core_name=$2
  local core_dir="/opt/xtensa/licenses/RI-2022.9-linux/${core_name}"

  if [[ -f "/opt/xtensa/${core_archive}" ]]; then
    echo "Installing Xtensa core: ${core_name} from ${core_archive}..."
    cd /opt/xtensa/
    tar xzf "${core_archive}" --dir /opt/xtensa/licenses/RI-2022.9-linux/
    if [[ -d "/opt/xtensa/licenses/RI-2022.9-linux/RI-2022.9-linux/${core_name}" ]]; then
      mv "/opt/xtensa/licenses/RI-2022.9-linux/RI-2022.9-linux/${core_name}" "${core_dir}"
      rmdir "/opt/xtensa/licenses/RI-2022.9-linux/RI-2022.9-linux" 2>/dev/null || true
    fi

    # Update installation paths in core parameter files (works for both fresh Cadence
    # core packages and pre-configured Google3 depot core packages).
    sed -i \
      -e "s|^install-prefix = .*|install-prefix = ${XTENSA_TOOLS_DIR}|g" \
      -e "s|^config-prefix = .*|config-prefix = ${core_dir}|g" \
      -e "s|^xtensa-tools = .*|xtensa-tools = ${XTENSA_TOOLS_DIR}/Tools|g" \
      -e "s|^tc-tools = .*|tc-tools = ${XTENSA_TOOLS_DIR}/TIE|g" \
      -e "s|\.\./\.\./\.\./\.\./\.\./unsupported_toolchains/xtensa/RI_2022_9/XtensaTools|${XTENSA_TOOLS_DIR}|g" \
      -e "s|\.\./\.\./${core_name}|${core_dir}|g" \
      -e "s|/usr/local/google/home/[^/]*/xtensa/[^/]*/install/builds/RI-2022.9-linux/${core_name}|${core_dir}|g" \
      -e "s|/usr/local/google/home/[^/]*/xtensa/[^/]*/install/tools/RI-2022.9-linux/XtensaTools|${XTENSA_TOOLS_DIR}|g" \
      "${core_dir}/config/default-params" \
      "${core_dir}/config/${core_name}-params"

    cp "${core_dir}/config/${core_name}-params" "${XTENSA_TOOLS_DIR}/config/${core_name}-params"

    # Remove unused RTL/SystemC examples and source trees to save ~1.1 GB in image size.
    rm -rf "${core_dir}/examples" "${core_dir}/src"
  else
    echo "Warning: Archive /opt/xtensa/${core_archive} not found, skipping."
  fi
}

install_core "P29A_FusionF1_linux.tgz" "P29A_FusionF1"
install_core "P29A_HiFi3z_linux.tgz" "P29A_HiFi3z"
install_core "P29A_VP6_linux.tgz" "P29A_VP6"
install_core "PRD_H5_RDO_07_01_2022_linux.tgz" "PRD_H5_RDO_07_01_2022"
