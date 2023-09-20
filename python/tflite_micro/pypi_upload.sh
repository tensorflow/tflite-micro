#!/bin/sh

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

set -e

USAGE="$(basename $0) [--test-pypi] <whl>...

Upload the given Python wheels to PyPI using the program twine. Requires an
authentication token in the environment variable TWINE_PASSWORD. TWINE_USERNAME
is set to \`__token__\` if not set in the environment.
"

die () {
    echo "$*" >&2
    exit 1
}

case "$1" in
    --test-pypi)
        export TWINE_REPOSITORY=testpypi
        shift
        ;;
    -h|--help)
        echo "$USAGE"
        exit
esac

if [ ! "$#" -ge 1 ]; then
    die "$USAGE"
fi

if [ ! -x $(command -v twine) ]; then
    die "error: twine not found. On Debian and derivatives, try \`apt install twine\`."
fi

if [ ! "$TWINE_PASSWORD" ]; then
    die "error: TWINE_PASSWORD is not set"
fi

: ${TWINE_USERNAME:="__token__"}

export TWINE_PASSWORD
export TWINE_USERNAME
twine upload "$@"
