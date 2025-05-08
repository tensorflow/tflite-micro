# Android builds do not need to link in a separate pthread library.
LIB_COPTS = []

LIB_LINKOPTS = select({
    ":android": [],
    ":windows": [],
    "//conditions:default": ["-lpthread"],
})

BIN_LINKOPTS = LIB_LINKOPTS

