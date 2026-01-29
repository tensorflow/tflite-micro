import sys
import os
import types

# Shim to expose 'tflite_micro' submodules pointing to the repo root directories.
# This allows legacy imports like 'from tflite_micro.tensorflow...' to work in Bzlmod runfiles structure.

repo_root = os.path.dirname(os.path.abspath(__file__))


def create_namespace_module(name, relative_path):
  m = types.ModuleType(name)
  m.__path__ = [os.path.join(repo_root, relative_path)]
  return m


# Inject submodules for all top-level directories
for name in ["tensorflow", "python", "tools", "signal", "third_party"]:
  full_name = __name__ + "." + name
  if full_name not in sys.modules:
    sys.modules[full_name] = create_namespace_module(full_name, name)