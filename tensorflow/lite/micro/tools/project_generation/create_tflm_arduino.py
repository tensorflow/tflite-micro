# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Script to generate TFLM Arduino examples ZIP file"""

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union

# parse args: {outdir}
# generate list of all source/header files and their src/dst locations
# create directory tree in {outdir}/tflm_arduino
# download third-party source
# copy templates: library.properties to {outdir}/tflm_arduino
# copy templates: TensorFlowLite.h to {outdir}/tflm_arduino/src
# copy transforms: examples
#   --platform=arduino
#   --third_party_headers=<output from make
#       list_{example_name}_example_headers
#       list_third_party_headers>
#   --is_example_source
# copy transforms: examples/{example_name}/main_functions.cc to
#  {example_name}.ino
#   --platform=arduino
#   --third_party_headers=<output from make
#       list_{example_name}_example_headers
#       list_third_party_headers>
#   --is_example_ino
# copy transforms: tensorflow, third_party
#   --platform=arduino
#   --third_party_headers=<output from make
#       list_third_party_headers>
# patch third_party/flatbuffers/include/flatbuffers/base.h with:
#   sed -E 's/utility\.h/utility/g'
# run fix_arduino_subfolders.py {outdir}/tflm_arduino
# remove all empty directories in {outdir}/tflm_arduino tree
# create ZIP file using shutil.make_archive()


def RunSedScripts(file_path: Path,
                  scripts: List[str],
                  args: Union[str, None] = None,
                  is_dry_run: bool = True) -> None:
  """
  Run SED scripts with specified arguments against the given file.
  The file is updated in place.

  Args:
    file_path: The full path to the input file
    scripts: A list of strings, each containing a single SED script
    args: a string containing all the SED arguments | None
    is_dry_run: if True, do not execute any commands
  """
  cmd = "sed"
  if args is not None and args != "":
    cmd += f" {args}"
  for scr in scripts:
    cmd += f" -E {scr}"
  cmd += f" < {file_path!s}"
  print(f"Running command: {cmd}")
  if not is_dry_run:
    result = subprocess.run(cmd,
                            shell=True,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    print(f"Saving output to: {file_path!s}")
    file_path.write_bytes(result.stdout)


def RunMakefileScript(path_to_makefile: Path,
                      args: str,
                      is_dry_run: bool = True) -> str:
  """
  Run a makefile script with specified arguments

  Args:
    path_to_makefile: The full path to the Makefile
    args: a string containing all the Makefile arguments
    is_dry_run: if True, do not execute any commands

  Returns:
    String containing Makefile output.  All line terminators
    in the string will be the newline character.
  """
  cmd = "make"
  if is_dry_run:
    cmd += " -n"
  cmd += f" -f {path_to_makefile!s}"
  cmd += f" {args}"
  print(f"Running command: {cmd}")
  result = subprocess.run(cmd,
                          shell=True,
                          check=True,
                          universal_newlines=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
  return result.stdout


def RemoveDirectories(paths: List[Path], is_dry_run: bool = True) -> None:
  """
  Remove directory tree(s) given list of pathnames

  Args:
    paths: A list of Path objects
    is_dry_run: if True, do not execute any commands
  """
  for dir_path in paths:
    print(f"Removing directory tree {dir_path!s}")
    if dir_path.exists() and not is_dry_run:
      shutil.rmtree(dir_path)


def RemoveEmptyDirectories(paths: Iterable[Path],
                           root: Path,
                           is_dry_run: bool = True) -> None:
  """
  Remove empty directories given list of pathnames, searching parent
  directories until reaching the root directory

  Args:
    paths: A list of Path objects
    root: The path at which to stop parent directory search
    is_dry_run: if True, do not execute any commands
  """
  empty_paths = list(filter(lambda p: list(p.glob("*")) == [], paths))
  parent_paths: Set[Path] = set()
  for dir_path in empty_paths:
    if dir_path == root:
      continue
    parent_paths.add(dir_path.parent)
    print(f"Removing empty directory {dir_path!s}")
    if not is_dry_run:
      dir_path.rmdir()
  if len(parent_paths) > 0:
    RemoveEmptyDirectories(parent_paths, root=root, is_dry_run=is_dry_run)


def RunPythonScript(path_to_script: Path,
                    args: str,
                    is_dry_run: bool = True) -> None:
  """
  Run a python script with specified arguments

  Args:
    path_to_script: The full path to the Python script
    args: a string containing all the script arguments
    is_dry_run: if True, do not execute any commands
  """
  cmd = f"python3 {path_to_script!s}"
  cmd += f" {args}"
  print(f"Running command: {cmd}")
  if not is_dry_run:
    _ = subprocess.run(cmd,
                       shell=True,
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)


def CreateDirectories(paths: List[Path], is_dry_run: bool = True) -> None:
  """
  Create directory tree(s) given list of pathnames

  Args:
    paths: A list of Path objects
    is_dry_run: if True, do not execute any commands
  """
  dir_path: Path
  for dir_path in paths:
    print(f"Creating directory tree {dir_path!s}")
    if not dir_path.is_dir() and not is_dry_run:
      dir_path.mkdir(mode=0o755, parents=True, exist_ok=True)


def CopyFiles(paths: Iterable[Tuple[Path, Path]],
              is_dry_run: bool = True) -> None:
  """
  Copy files given list of source and destination Path tuples

  Args:
    paths: A list of tuples of Path objects.
    Each tuple is of the form (source, destination)
    is_dry_run: if True, do not execute any commands
  """
  dir_path: Tuple[Path, Path]
  for dir_path in paths:
    print(f"Copying {dir_path[0]!s} to {dir_path[1]!s}")
    if not is_dry_run:
      shutil.copy2(dir_path[0], dir_path[1])


class ArduinoProjectGenerator:
  """
  Generate the TFLM Arduino library ZIP file
  """

  def __init__(self) -> None:
    args = self._ParseArguments().parse_args()
    self.makefile_options = args.makefile_options
    self.output_dir: Path = Path(args.output_dir)
    self.output_arduino_dir: Path = Path(
        self.output_dir) / args.output_file_basename
    self.is_dry_run: bool = args.is_dry_run
    self.examples = [
        "hello_world", "person_detection", "micro_speech", "magic_wand"
    ]
    self.tensorflow_path = Path("tensorflow/lite")
    self.examples_path = Path("tensorflow/lite/micro/examples")
    self.downloads_path = Path("tensorflow/lite/micro/tools/make/downloads")

  # private methods
  def _ParseArguments(self) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Script for TFLM Arduino project generation")
    parser.add_argument("--output_dir",
                        type=str,
                        default=tempfile.gettempdir(),
                        help="Output directory for generated TFLM tree")
    parser.add_argument("--output_file_basename",
                        type=str,
                        default="tflm_arduino",
                        help="Output ZIP file name")
    parser.add_argument("--is_dry_run",
                        default=False,
                        action="store_true",
                        help="Show commands only (no execution)")
    parser.add_argument("--makefile_options",
                        default="",
                        help="Additional TFLM Makefile options. For example: "
                        "--makefile_options=\"TARGET=<target> "
                        "OPTIMIZED_KERNEL_DIR=<optimized_kernel_dir> "
                        "TARGET_ARCH=corex-m4\"")
    return parser

  def _CleanOutputDirectory(self) -> None:
    dirs_to_remove = [self.output_arduino_dir]
    RemoveDirectories(dirs_to_remove, is_dry_run=self.is_dry_run)
    zip_path = self.output_arduino_dir.with_suffix(".zip")
    print(f"Removing ZIP file: {zip_path!s}")
    if zip_path.exists() and not self.is_dry_run:
      zip_path.unlink()

  def _DownloadThirdPartyFiles(self) -> None:
    makefile_path = Path("tensorflow/lite/micro/tools/make/Makefile")
    args = self.makefile_options
    args += " third_party_downloads"
    results = RunMakefileScript(makefile_path, args, is_dry_run=self.is_dry_run)
    print(results)

  def _CreateOutputArduinoDirectories(
      self, all_path_pairs: List[Tuple[Path, Path]]) -> None:
    # generate full list of source tree directories
    # collect relative destination paths and sort relative paths
    set_relative_subdirs: Set[Path] = {
        path[1].parent for path in all_path_pairs if path[1].parent != Path(".")
    }
    relative_subdirs = list(set_relative_subdirs)
    relative_subdirs.sort()

    # filter out common parents
    def _FilterFunc(pair: Tuple[int, Path]):
      index = pair[0]
      if index == len(relative_subdirs) - 1:
        return True
      elif pair[1] != relative_subdirs[index + 1].parent:
        return True
      else:
        return False

    filtered_subdirs: List[Tuple[int, Path]] = list(
        filter(_FilterFunc, enumerate(relative_subdirs)))
    # convert from enumerated tuples back into list of Path objects
    if filtered_subdirs != []:
      relative_subdirs = list(list(zip(*filtered_subdirs))[1])
    else:
      relative_subdirs = []

    # convert relative paths to full destination paths
    dst_subdirs = [self.output_arduino_dir / path for path in relative_subdirs]
    CreateDirectories(dst_subdirs, is_dry_run=self.is_dry_run)

  def _CopyTemplates(self) -> None:
    templates_dir = Path("tensorflow/lite/micro/tools/make/templates")
    relative_paths: List[Tuple[Path, str]] = [
        (templates_dir / "library.properties", "library.properties"),
        (templates_dir / "TensorFlowLite.h", "src/TensorFlowLite.h")
    ]
    full_paths: List[Tuple[Path, Path]] = [
        (item[0], self.output_arduino_dir / item[1]) for item in relative_paths
    ]
    CopyFiles(full_paths, is_dry_run=self.is_dry_run)

  def _CopyWithTransform(self, all_path_pairs: List[Tuple[Path, Path]]) -> None:
    script_path = Path("tensorflow/lite/micro/tools/make/transform_source.py")
    headers_dict: Dict[Union[str, None], str] = {
        example: " ".join(self._GenerateHeaderList(example))
        for example in self.examples
    }
    headers_dict[None] = " ".join(self._GenerateHeaderList(None))

    # generate set of all source and header files to transform
    src_suffixes = [".c", ".cc", ".cpp", ".h"]
    source_pairs: Set[Tuple[Path, Path]] = set(
        filter(lambda path: path[0].suffix in src_suffixes, all_path_pairs))

    # transform all source and header files
    for relative_paths in source_pairs:
      dst_path = self.output_arduino_dir / relative_paths[1]
      src_path = relative_paths[0]
      if relative_paths[1].parts[0] == "examples":
        third_party_headers = headers_dict[relative_paths[1].parts[1]]
        if relative_paths[1].suffix == ".ino":
          is_example_ino = True
          is_example_source = False
        else:
          is_example_ino = False
          is_example_source = True
      else:
        third_party_headers = headers_dict[None]
        is_example_source = False
        is_example_ino = False

      args = "--platform=arduino"
      if is_example_ino:
        args += " --is_example_ino"
      elif is_example_source:
        args += " --is_example_source"
      args += f' --third_party_headers="{third_party_headers}"'
      args += f" < {src_path!s} > {dst_path!s}"
      RunPythonScript(script_path, args=args, is_dry_run=self.is_dry_run)

    # generate set of files to copy without transform
    copy_pairs = set(all_path_pairs) - source_pairs
    copy_pairs = map(lambda pair: (pair[0], self.output_arduino_dir / pair[1]),
                     copy_pairs)
    CopyFiles(copy_pairs, is_dry_run=self.is_dry_run)

  def _PatchWithSed(self) -> None:
    patches: List[Tuple[str, List[str]]] = [
        ("src/third_party/flatbuffers/include/flatbuffers/base.h",
         [r"'s/utility\.h/utility/g'"])
    ]
    for file, scripts in patches:
      RunSedScripts(self.output_arduino_dir / file,
                    scripts=scripts,
                    is_dry_run=self.is_dry_run)

  def _FixSubDirectories(self) -> None:
    script_path = Path(
        "tensorflow/lite/micro/tools/make/fix_arduino_subfolders.py")
    args = str(self.output_arduino_dir)
    RunPythonScript(script_path, args, is_dry_run=self.is_dry_run)

  def _MakeZipFile(self) -> None:
    print(f"Creating ZIP file: {self.output_arduino_dir!s}.zip")
    shutil.make_archive(base_name=str(self.output_arduino_dir),
                        format="zip",
                        root_dir=self.output_arduino_dir.parent,
                        base_dir=self.output_arduino_dir.name,
                        dry_run=self.is_dry_run,
                        logger=logging.getLogger())

  def _GenerateHeaderList(self, example: Union[str, None]) -> List[str]:
    makefile_path = Path("tensorflow/lite/micro/tools/make/Makefile")
    args = self.makefile_options
    if example is not None:
      # need just the headers for this example
      args += f" list_{example}_example_headers"
    else:
      # need headers for all examples; will sort/filter below
      for example_name in self.examples:
        args += f" list_{example_name}_example_headers"
    args += " list_third_party_headers"
    result = RunMakefileScript(makefile_path, args, is_dry_run=self.is_dry_run)

    # change third-party paths
    result = result.replace(f"{self.downloads_path!s}", "third_party")
    # change example paths
    if example is not None:
      #result = result.replace(f"{self.examples_path!s}/{example}/", "")
      list_result = result.split()
    else:
      # remove duplicates
      set_result = set(result.split())
      # filter out examples headers
      list_result = list(
          filter(lambda s: self.examples_path not in Path(s).parents,
                 set_result))

    return list_result

  def _GenerateFilePathsRelative(self) -> List[Tuple[Path, Path]]:
    # generate set of all files
    makefile_path = Path("tensorflow/lite/micro/tools/make/Makefile")
    args = self.makefile_options
    for example in self.examples:
      args += f" list_{example}_example_headers"
      args += f" list_{example}_example_sources"
    args += " list_third_party_headers"
    args += " list_third_party_sources"
    args += " list_library_headers"
    args += " list_library_sources"
    result = RunMakefileScript(makefile_path, args, is_dry_run=self.is_dry_run)
    all_files = result.split()

    ino_files = ["main_functions.cc"]
    # paths that require special handling
    special_paths: Dict[Path, Path] = {
        Path(f"{self.downloads_path!s}"
             "/person_model_int8/person_detect_model_data.cc"):
            Path("examples/person_detection/person_detect_model_data.cpp")
    }

    # generate relative source/destination pairs
    relative_path_pairs: List[Tuple[Path, Path]] = []
    for file in all_files:
      src_path = Path(file)
      path_parents = src_path.parents
      if src_path in special_paths.keys():
        dst_path = special_paths[src_path]
      elif self.examples_path in path_parents:
        dst_path = "examples" / src_path.relative_to(self.examples_path)
        # check for .ino file rename
        if dst_path.name in ino_files:
          dst_path = dst_path.with_name(dst_path.parts[1] + ".ino")
      elif self.downloads_path in path_parents:
        dst_path = "src/third_party" / src_path.relative_to(self.downloads_path)
      elif self.tensorflow_path in path_parents:
        dst_path = "src" / src_path
      else:
        dst_path = src_path

      # check for .cc to .cpp rename
      if dst_path.suffix == ".cc":
        dst_path = dst_path.with_suffix(".cpp")

      # add new tuple(src,dst) to list
      relative_path_pairs.append(tuple([src_path, dst_path]))

    return relative_path_pairs

  def _RemoveEmptyDirectories(self) -> None:
    paths = self.output_arduino_dir.glob("**")
    RemoveEmptyDirectories(paths,
                           root=self.output_arduino_dir,
                           is_dry_run=self.is_dry_run)

  # public methods
  def CreateZip(self) -> None:
    """
    Execute all steps to create TFLM Arduino ZIP file
    """
    self._CleanOutputDirectory()
    self._DownloadThirdPartyFiles()
    all_files = self._GenerateFilePathsRelative()
    self._CreateOutputArduinoDirectories(all_files)
    self._CopyTemplates()
    self._CopyWithTransform(all_files)
    self._PatchWithSed()
    self._FixSubDirectories()
    self._RemoveEmptyDirectories()
    self._MakeZipFile()


if __name__ == "__main__":
  generator = ArduinoProjectGenerator()
  generator.CreateZip()
