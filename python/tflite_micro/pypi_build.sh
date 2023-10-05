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

USAGE="$(basename $0) <python-tag> [<output-directory>]

Build a Python wheel for public release to PyPI using a special Docker build
container. Uses bazel, but does not pollute the WORKSPACE's default cache.

<python-tag> must be one of the supported interpreters:
   cp310
   cp311

<output-directory> defaults to $OUT_DIR_DEFAULT.
"

case "$1" in
    cp310|cp311)
        PY_TAG=$1
        OUTDIR=$(realpath ${2:-$OUT_DIR_DEFAULT})
        mkdir -p $OUTDIR
        break
        ;;
    *)
        echo usage: "$USAGE" >&2
        exit 1
esac

SRCDIR=$(realpath .)
if ! test -f $SRCDIR/WORKSPACE; then
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
    --rm \
    --interactive \
    --mount type=bind,source=$SRCDIR,destination=$SRCDIR \
    --mount type=bind,source=$OUTDIR,destination=$OUTDIR \
    --workdir $SRCDIR \
    --env USER=$(id -un) \
    $IMAGE_ID \
    /bin/bash -s -e -x -u \
<<EOF
    # Setup the Python compatibility tags. The PY_ABI always matches the Python
    # interpreter tag. The platform tag is supplied by the build image in the
    # environment variable AUDITWHEEL_PLAT.
    PY_ABI=$PY_TAG
    PY_PLATFORM=\$AUDITWHEEL_PLAT
    PY_COMPATIBILITY=${PY_TAG}_\${PY_ABI}_\${PY_PLATFORM}

    # Link the desired Python version into the PATH, where bazel will find it.
    # The build image contains many different Python versions as options.
    ln -sf /opt/python/$PY_TAG-$PY_TAG/bin/* /usr/bin

    # Bazelisk fails if it can't check HOME for a .rc file., and pip (in
    # :whl_test) installation of some dependencies (e.g., wrapt) expects HOME.
    export HOME=$OUTDIR

    # Bazelisk, bazel, and pip all need a writable cache directory.
    export XDG_CACHE_HOME=$OUTDIR/cache

    # Relocate the bazel root to keep the cache used for each Python toolchain
    # separate. Drop root privledges and run as the invoking user.
    call_bazel() {
        setpriv --reuid=$(id -u) --regid=$(id -g) --clear-groups \
            bazel \
                --output_user_root=$OUTDIR/$PY_TAG-out \
                "\$@" \
                --action_env=HOME `# help setuptools find HOME in container` \
                --action_env=USER `# bazel reads USER via whoami` \
                --action_env=XDG_CACHE_HOME `# locate pip's cache inside OUTDIR`
    }

    # Build the wheel via bazel, using the Python compatibility tag matching the
    # build environment.
    call_bazel build //python/tflite_micro:whl.dist \
        --//python/tflite_micro:compatibility_tag=\$PY_COMPATIBILITY

    # Test, in the container environment.
    call_bazel test //python/tflite_micro:whl_test \
            --//python/tflite_micro:compatibility_tag=\$PY_COMPATIBILITY
EOF

# Make the output directory tree writable so it can be removed easily by the
# user with `rm -rf $OUTDIR`. Bazel leaves it write-protected.
chmod -R +w $OUTDIR

# Copy the generated wheel file to the root of the $OUTDIR.
cp bazel-bin/python/tflite_micro/whl_dist/*.whl $OUTDIR
echo "Output:\n$(ls $OUTDIR/*.whl)"
