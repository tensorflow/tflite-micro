def if_not_windows(a):
    return select({
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": a,
    })

def clean_dep(dep):
    return str(Label(dep))

def get_compatible_with_portable():
    return []
