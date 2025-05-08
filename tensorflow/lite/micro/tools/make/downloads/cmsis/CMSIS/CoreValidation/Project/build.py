#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from datetime import datetime
from enum import Enum
from glob import glob, iglob
from pathlib import Path

from lxml.etree import XMLSyntaxError
from zipfile import ZipFile

from matrix_runner import main, matrix_axis, matrix_action, matrix_command, matrix_filter, \
    ConsoleReport, CropReport, TransformReport, JUnitReport


@matrix_axis("device", "d", "Device(s) to be considered.")
class DeviceAxis(Enum):
    CM0 = ('Cortex-M0', 'CM0')
    CM0plus = ('Cortex-M0plus', 'CM0plus')
    CM3 = ('Cortex-M3', 'CM3')
    CM4 = ('Cortex-M4', 'CM4')
    CM7 = ('Cortex-M7', 'CM7')
    CM23 = ('Cortex-M23', 'CM23')
    CM23S = ('Cortex-M23S', 'CM23S')
    CM23NS = ('Cortex-M23NS', 'CM23NS')
    CM33 = ('Cortex-M33', 'CM33')
    CM33S = ('Cortex-M33S', 'CM33S')
    CM33NS = ('Cortex-M33NS', 'CM33NS')
    CM35P = ('Cortex-M35P', 'CM35P')
    CM35PS = ('Cortex-M35PS', 'CM35PS')
    CM35PNS = ('Cortex-M35PNS', 'CM35PNS')
    CM52 = ('Cortex-M52', 'CM52')
    CM52S = ('Cortex-M52S', 'CM52S')
    CM52NS = ('Cortex-M52NS', 'CM52NS')
    CM55 = ('Cortex-M55', 'CM55')
    CM55S = ('Cortex-M55S', 'CM55S')
    CM55NS = ('Cortex-M55NS', 'CM55NS')
    CM85 = ('Cortex-M85', 'CM85')
    CM85S = ('Cortex-M85S', 'CM85S')
    CM85NS = ('Cortex-M85NS', 'CM85NS')
    CA5 = ('Cortex-A5', 'CA5')
    CA7 = ('Cortex-A7', 'CA7')
    CA9 = ('Cortex-A9', 'CA9')
#    CA5NEON = ('Cortex-A5neon', 'CA5neon')
#    CA7NEON = ('Cortex-A7neon', 'CA7neon')
#    CA9NEON = ('Cortex-A9neon', 'CA9neon')

    def has_bl(self):
        return self in [
            DeviceAxis.CM23NS,
            DeviceAxis.CM33NS,
            DeviceAxis.CM35PNS,
            DeviceAxis.CM52NS,
            DeviceAxis.CM55NS,
            DeviceAxis.CM85NS
        ]

    @property
    def bl_device(self):
        bld = {
            DeviceAxis.CM23NS: 'CM23S',
            DeviceAxis.CM33NS: 'CM33S',
            DeviceAxis.CM35PNS: 'CM35PS',
            DeviceAxis.CM52NS: 'CM52S',
            DeviceAxis.CM55NS: 'CM55S',
            DeviceAxis.CM85NS: 'CM85S'
        }
        return bld[self]


@matrix_axis("compiler", "c", "Compiler(s) to be considered.")
class CompilerAxis(Enum):
    AC6 = ('AC6')
    GCC = ('GCC')
    IAR = ('IAR')
    CLANG = ('Clang')

    @property
    def image_ext(self):
        ext = {
            CompilerAxis.AC6: 'axf',
            CompilerAxis.GCC: 'elf',
            CompilerAxis.IAR: 'elf',
            CompilerAxis.CLANG: 'elf',
        }
        return ext[self]

    @property
    def toolchain(self):
        ext = {
            CompilerAxis.AC6: 'AC6',
            CompilerAxis.GCC: 'GCC',
            CompilerAxis.IAR: 'IAR',
            CompilerAxis.CLANG: 'CLANG'
        }
        return ext[self]


@matrix_axis("optimize", "o", "Optimization level(s) to be considered.")
class OptimizationAxis(Enum):
    NONE = ('none')
    BALANCED = ('balanced')
    SPEED = ('speed')
    SIZE = ('size')


MODEL_EXECUTABLE = {
    DeviceAxis.CM0: ("FVP_MPS2_Cortex-M0", []),
    DeviceAxis.CM0plus: ("FVP_MPS2_Cortex-M0plus", []),
    DeviceAxis.CM3: ("FVP_MPS2_Cortex-M3", []),
    DeviceAxis.CM4: ("FVP_MPS2_Cortex-M4", []),
    DeviceAxis.CM7: ("FVP_MPS2_Cortex-M7", []),
    DeviceAxis.CM23: ("FVP_MPS2_Cortex-M23", []),
    DeviceAxis.CM23S: ("FVP_MPS2_Cortex-M23", []),
    DeviceAxis.CM23NS: ("FVP_MPS2_Cortex-M23", []),
    DeviceAxis.CM33: ("FVP_MPS2_Cortex-M33", []),
    DeviceAxis.CM33S: ("FVP_MPS2_Cortex-M33", []),
    DeviceAxis.CM33NS: ("FVP_MPS2_Cortex-M33", []),
    DeviceAxis.CM35P: ("FVP_MPS2_Cortex-M35P", []),
    DeviceAxis.CM35PS: ("FVP_MPS2_Cortex-M35P", []),
    DeviceAxis.CM35PNS: ("FVP_MPS2_Cortex-M35P", []),
    DeviceAxis.CM52: ("FVP_MPS2_Cortex-M52", []),
    DeviceAxis.CM52S: ("FVP_MPS2_Cortex-M52", []),
    DeviceAxis.CM52NS: ("FVP_MPS2_Cortex-M52", []),
    DeviceAxis.CM55: ("FVP_MPS2_Cortex-M55", []),
    DeviceAxis.CM55S: ("FVP_MPS2_Cortex-M55", []),
    DeviceAxis.CM55NS: ("FVP_MPS2_Cortex-M55", []),
    DeviceAxis.CM85: ("FVP_MPS2_Cortex-M85", []),
    DeviceAxis.CM85S: ("FVP_MPS2_Cortex-M85", []),
    DeviceAxis.CM85NS: ("FVP_MPS2_Cortex-M85", []),
    DeviceAxis.CA5: ("FVP_VE_Cortex-A5x1", []),
    DeviceAxis.CA7: ("FVP_VE_Cortex-A7x1", []),
    DeviceAxis.CA9: ("FVP_VE_Cortex-A9x1", []),
#    DeviceAxis.CA5NEON: ("_VE_Cortex-A5x1", []),
#    DeviceAxis.CA7NEON: ("_VE_Cortex-A7x1", []),
#    DeviceAxis.CA9NEON: ("_VE_Cortex-A9x1", [])
}

QEMU_MACHINE = {
    DeviceAxis.CM3: ("mps2-an385"),
    DeviceAxis.CM4: ("mps2-an386"),
    DeviceAxis.CM7: ("mps2-an500"),
    DeviceAxis.CM33: ("mps2-an505"),
    DeviceAxis.CM55: ("mps3-an547")
}

def config_suffix(config, timestamp=True):
    suffix = f"{config.compiler[0]}-{config.optimize[0]}-{config.device[1]}"
    if timestamp:
        suffix += f"-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return suffix


def project_name(config):
    return f"Validation.{config.compiler}_{config.optimize}+{config.device[1]}"


def bl_project_name(config):
    return f"Bootloader.{config.compiler}_{config.optimize}+{config.device.bl_device}"


def output_dir(config):
    return f"Validation/outdir"


def bl_output_dir(config):
    return f"Bootloader/outdir"


def model_config(config):
    return f"../Layer/Target/{config.device[1]}/model_config.txt"


def build_dir(config):
    return f"build/{config.device[1]}/{config.compiler.toolchain}/{config.optimize}"


@matrix_action
def clean(config):
    """Build the selected configurations using CMSIS-Build."""
    yield cbuild_clean(f"{project_name(config)}/{project_name(config)}.cprj")


@matrix_action
def build(config, results):
    """Build the selected configurations using CMSIS-Build."""

    logging.info("Compiling Tests...")

    yield cbuild(config)

    if not all(r.success for r in results):
        return

    file = f"build/CoreValidation-{config_suffix(config)}.zip"
    logging.info("Archiving build output to %s...", file)
    with ZipFile(file, "w") as archive:
        for content in iglob(f"{build_dir(config)}/**/*", recursive=True):
            if Path(content).is_file():
                archive.write(content)


@matrix_action
def extract(config):
    """Extract the latest build archive."""
    archives = sorted(glob(f"build/CoreValidation-{config_suffix(config, timestamp=False)}-*.zip"), reverse=True)
    yield unzip(archives[0])


@matrix_action
def run(config, results):
    """Run the selected configurations."""
    logging.info("Running Core Validation on Arm model ...")
    yield model_exec(config)

    try:
        results[0].test_report.write(f"build/CoreValidation-{config_suffix(config)}.junit")
    except RuntimeError as ex:
        if isinstance(ex.__cause__, XMLSyntaxError):
            logging.error("No valid test report found in model output!")
        else:
            logging.exception(ex)


@matrix_action
def qemu(config, results):
    """Run the selected configurations."""
    if not config.device in QEMU_MACHINE:
        logging.error("Qemu doesn't support target device '%s'!", config.device)
        return
    logging.info("Running Core Validation on Qemu ...")
    yield qemu_exec(config)


@matrix_command()
def cbuild_clean(project):
    return ["cbuild", "-C", project]


@matrix_command()
def unzip(archive):
    return ["bash", "-c", f"unzip {archive}"]


@matrix_command()
def preprocess(infile, outfile):
    return ["arm-none-eabi-gcc", "-xc", "-E", infile, "-P", "-o", outfile]


@matrix_command()
def cbuild(config):
    return ["cbuild", "--toolchain", config.compiler.toolchain, "--update-rte", \
             "--context", f".{config.optimize}+{config.device[1]}", \
             "Validation.csolution.yml" ]


@matrix_command(test_report=ConsoleReport() |
                            CropReport('<\\?xml version="1.0"\\?>', '</report>') |
                            TransformReport('validation.xsl') |
                            JUnitReport(title=lambda title, result: f"{result.command.config.compiler}."
                                                                    f"{result.command.config.optimize}."
                                                                    f"{result.command.config.device}."
                                                                    f"{title}"))
def model_exec(config):
    cmdline = [MODEL_EXECUTABLE[config.device][0], "-q", "--simlimit", 100, "-f", model_config(config)]
    cmdline += MODEL_EXECUTABLE[config.device][1]
    cmdline += ["-a", f"{build_dir(config)}/{output_dir(config)}/Validation.{config.compiler.image_ext}"]
    if config.device.has_bl():
        cmdline += ["-a", f"{build_dir(config)}/{bl_output_dir(config)}/Bootloader.{config.compiler.image_ext}"]
    return cmdline


@matrix_command(test_report=ConsoleReport() |
                            CropReport('<\\?xml version="1.0"\\?>', '</report>') |
                            TransformReport('validation.xsl') |
                            JUnitReport(title=lambda title, result: f"{result.command.config.compiler}."
                                                                    f"{result.command.config.optimize}."
                                                                    f"{result.command.config.device}."
                                                                    f"{title}"))
def qemu_exec(config):
    cmdline = ["qemu-system-arm", "-semihosting-config", "enable=on", "-monitor", "none", "-serial", "none", "-nographic"]
    cmdline += ["-machine", QEMU_MACHINE[config.device]]
    cmdline += ["-kernel", f"{build_dir(config)}/{output_dir(config)}/Validation.{config.compiler.image_ext}"]
    return cmdline

@matrix_filter
def filter_iar(config):
    return config.compiler == CompilerAxis.IAR

@matrix_filter
def filter_cm52(config):
    device = config.device.match('CM52*')
    compiler = config.compiler != CompilerAxis.AC6
    return device and compiler

if __name__ == "__main__":
    main()
