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

OUT_DIR_DEFAULT=bazel-pypi-out

USAGE="$(basename $0) [<output-directory>]

Build a universal Python wheel (py3-none-manylinux) for public release to PyPI
using a special Docker build container. Uses bazel, but does not pollute the
WORKSPACE's default cache.

<output-directory> defaults to $OUT_DIR_DEFAULT.
"

OUTDIR=$(realpath ${1:-$OUT_DIR_DEFAULT})
mkdir -p $OUTDIR

SRCDIR=$(realpath .)
if ! test -f $SRCDIR/MODULE.bazel; then
    echo "error: must run from the top of the source tree" >&2
    exit 1
fi

# Remove Bazel's workspace symlinks so they'll be rewritten below, pointing into
# OUTDIR.
find . -maxdepth 1 -type l -name bazel-\* | xargs rm -f

# Build the Docker image from its source file. Don't pollute the public list of
# images by tagging; just use the image's ID.
DOCKERFILE=python/tflite_micro/pypi_build.dockerfile
IMAGE_ID_FILE=$OUTDIR/image-id
docker build - --iidfile $IMAGE_ID_FILE <$DOCKERFILE
IMAGE_ID=$(cat $IMAGE_ID_FILE)

# Build the Python package within an ephemeral container.
docker run \
    --init \
    --rm \
    --interactive \
    --mount type=bind,source=$SRCDIR,destination=$SRCDIR \
    --mount type=bind,source=$OUTDIR,destination=$OUTDIR \
    --workdir $SRCDIR \
    --env USER=$(id -un) \
    $IMAGE_ID \
    /bin/bash -s -e -x -u \
<<EOF
    # Setup the universal Python compatibility tags.
    PY_PLATFORM=\$AUDITWHEEL_PLAT
    PY_COMPATIBILITY=py3_none_\${PY_PLATFORM}

    # Bazelisk fails if it can't check HOME for a .rc file., and pip (in
    # :whl_test) installation of some dependencies expects HOME.
    export HOME=$OUTDIR

    # Bazelisk, bazel, and pip all need a writable cache directory.
    export XDG_CACHE_HOME=$OUTDIR/cache

    call_bazel() {
        setpriv --reuid=$(id -u) --regid=$(id -g) --clear-groups \
            bazel \
                --output_user_root=$OUTDIR/bazel-out \
                "\$@" \
                --repo_contents_cache= \
                --action_env=HOME \
                --action_env=USER \
                --action_env=XDG_CACHE_HOME
    }

    # Build universal wheel via bazel.
    call_bazel build //python/tflite_micro:whl.dist \
        --compilation_mode=opt \
        --//python/tflite_micro:compatibility_tag=\$PY_COMPATIBILITY \
        --//:with_compression=true

    # Test in the container environment.
    call_bazel test //python/tflite_micro:whl_test \
            --compilation_mode=opt \
            --//python/tflite_micro:compatibility_tag=\$PY_COMPATIBILITY \
            --//:with_compression=true
EOF

chmod -R +w $OUTDIR
cp bazel-bin/python/tflite_micro/whl_dist/*.whl $OUTDIR
echo "Output:\n$(ls $OUTDIR/*.whl)"
