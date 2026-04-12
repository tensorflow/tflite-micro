import sys
import os
import types

repo_root = os.path.dirname(os.path.abspath(__file__))


def create_namespace_module(name, path):
    if not os.path.isdir(path):
        return None

    if name in sys.modules:
        return sys.modules[name]

    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules[name] = module
    return module


def inject_namespaces():
    for entry in os.listdir(repo_root):
        full_path = os.path.join(repo_root, entry)

        # chỉ lấy directory hợp lệ
        if not os.path.isdir(full_path):
            continue

        # bỏ folder ẩn / không cần thiết
        if entry.startswith("_") or entry in ("__pycache__",):
            continue

        full_name = f"{__name__}.{entry}"
        create_namespace_module(full_name, full_path)


inject_namespaces()
