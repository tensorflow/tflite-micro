# Add URLs and MD5 checksums for third-party libraries here.
# We use mirror.tensorflow.org to cache copies of third-party files,
# but this is just an optimization applied manually by TensorFlow
# engineers, so add non-mirrored URLs if you need to update this
# in a pull request and we'll periodically copy them and update
# the URL.

GEMMLOWP_URL := "https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip"
GEMMLOWP_MD5 := "7e8191b24853d75de2af87622ad293ba"

LEON_BCC2_URL := "http://mirror.tensorflow.org/www.gaisler.com/anonftp/bcc2/bin/bcc-2.0.7-gcc-linux64.tar.xz"
LEON_BCC2_MD5 := "cdf78082be4882da2a92c9baa82fe765"

TSIM_URL := "http://mirror.tensorflow.org/www.gaisler.com/anonftp/tsim/tsim-eval-2.0.63.tar.gz"
TSIM_MD5 := "afa0095d3ed989a949e1467f94e41d2f"

ifeq ($(HOST_OS),osx)
  RISCV_TOOLCHAIN_URL := "http://mirror.tensorflow.org/static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-8.1.0-2019.01.0-x86_64-apple-darwin.tar.gz"
  RISCV_TOOLCHAIN_MD5 := "2ac2fa00618b9ab7fa0c7d0ec173de94"
else
  RISCV_TOOLCHAIN_URL := "http://mirror.tensorflow.org/static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-20181030-x86_64-linux-ubuntu14.tar.gz"
  RISCV_TOOLCHAIN_MD5="2366b7afe36a54dc94fb0ff8a0830934"
endif

RUY_URL="https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip"
RUY_MD5="abf7a91eb90d195f016ebe0be885bb6e"

PERSON_MODEL_INT8_URL := "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_int8_grayscale_2020_12_1.zip"
PERSON_MODEL_INT8_MD5 := "e765cc76889db8640cfe876a37e4ec00"

ifneq ($(filter $(ARC_TAGS), mli20_experimental),)
EMBARC_MLI_URL := "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/archive/refs/tags/Release_2.0.zip"
EMBARC_MLI_MD5 := "13dcc1ea81ed836326a616e7e842ae4d"
else
EMBARC_MLI_URL := "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/archive/refs/tags/Release_1.1.zip"
EMBARC_MLI_MD5 := "22555d76097727b00e731563b42cb098"
endif

EMBARC_MLI_PRE_COMPILED_URL := "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/releases/download/Release_1.1/embARC_MLI_package.zip"
EMBARC_MLI_PRE_COMPILED_MD5 := "173990c2dde4efef6a2c95b92d1f0244"

# Skip md5sum-check since ethos-u-core-driver download link is non-deterministic, see  https://github.com/google/gitiles/issues/84
