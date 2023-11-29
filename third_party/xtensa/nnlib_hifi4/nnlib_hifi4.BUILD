constraint_setting(
    name = "compatible_constraint",
)

# Set this constraint_value on your platform to indicate compatiblity with this
# library.
constraint_value(
    name = "compatible",
    constraint_setting = ":compatible_constraint",
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "lib",
    srcs = glob(["xa_nnlib/algo/**/*.c"]),
    hdrs = glob([
        "xa_nnlib/algo/**/*.h",
        "xa_nnlib/include/**/*.h",
    ]),
    copts = ["-Wno-unused-parameter"],
    defines = [
        "NNLIB_V2=1",
        "MODEL_INT16=1",
        "EIGEN_NO_MALLOC=1",
        "hifi4=1",
    ],
    includes = [
        "xa_nnlib",
        "xa_nnlib/algo/common/include",
        "xa_nnlib/algo/kernels",
        "xa_nnlib/algo/ndsp/hifi4/include",
        "xa_nnlib/include",
        "xa_nnlib/include/nnlib",
    ],
    target_compatible_with = [
        ":compatible",
    ],
    visibility = [
        "//visibility:public",
    ],
)
