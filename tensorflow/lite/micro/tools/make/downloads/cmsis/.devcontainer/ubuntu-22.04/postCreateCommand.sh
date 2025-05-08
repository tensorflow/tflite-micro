#!/bin/bash

echo "Installing oh-my-bash ..."
bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)" --unattended
sed -i 's/OSH_THEME="font"/OSH_THEME="powerline"/' ~/.bashrc

echo "Bootstrapping vcpkg ..."
# shellcheck source=/dev/null
. <(curl -sL https://aka.ms/vcpkg-init.sh)
grep -q "vcpkg-init" ~/.bashrc || echo -e "\n# Initialize vcpkg\n. ~/.vcpkg/vcpkg-init" >> ~/.bashrc && \
pushd "$(dirname "$0")" || exit 
vcpkg x-update-registry --all
vcpkg activate
popd || exit 
