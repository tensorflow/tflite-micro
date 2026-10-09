def tflm_kernel_friends():
    return []

def tflm_audio_frontend_friends():
    return []

def tflm_application_friends():
    return []

def tflm_signal_friends():
    return []

def tflm_python_op_resolver_friends():
    return []

def xtensa_fusion_f1_config():
    """Config setting for all Fusion F1 based cores."""
    return "//tensorflow/lite/micro/kernels:xtensa_fusion_f1_default"

def xtensa_hifi_3_config():
    """Config setting for all HiFi 3 based cores."""
    return "//tensorflow/lite/micro/kernels:xtensa_hifi_3_default"

def xtensa_hifi_3z_config():
    """Config setting for all HiFi 3z based cores."""
    return "//tensorflow/lite/micro/kernels:xtensa_hifi_3z_default"

def xtensa_hifi_5_config():
    """Config setting for all HiFi 5 based cores."""
    return "//tensorflow/lite/micro/kernels:xtensa_hifi_5_default"

def xtensa_vision_p6_config():
    """Config setting for all Vision P6 based cores."""
    return "//tensorflow/lite/micro/kernels:xtensa_vision_p6_default"
