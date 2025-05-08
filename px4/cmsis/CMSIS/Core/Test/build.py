#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from datetime import datetime
from enum import Enum

from matrix_runner import main, matrix_axis, matrix_action, matrix_command, matrix_filter, \
    FileReport, JUnitReport


@matrix_axis("device", "d", "Device(s) to be considered.")
class DeviceAxis(Enum):
    CM0 = ('Cortex-M0', 'CM0')
    CM0plus = ('Cortex-M0plus', 'CM0plus')
    CM3 = ('Cortex-M3', 'CM3')
    CM4 = ('Cortex-M4', 'CM4')
    CM4FP = ('Cortex-M4FP', 'CM4FP')
    CM7 = ('Cortex-M7', 'CM7')
    CM7SP = ('Cortex-M7SP', 'CM7SP')
    CM7DP = ('Cortex-M7DP', 'CM7DP')
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
    CA5NEON = ('Cortex-A5neon', 'CA5neon')
    CA7NEON = ('Cortex-A7neon', 'CA7neon')
    CA9NEON = ('Cortex-A9neon', 'CA9neon')


@matrix_axis("compiler", "c", "Compiler(s) to be considered.")
class CompilerAxis(Enum):
    AC6 = ('AC6')
    GCC = ('GCC')
    IAR = ('IAR')
    CLANG = ('Clang')


@matrix_axis("optimize", "o", "Optimization level(s) to be considered.")
class OptimizationAxis(Enum):
    NONE = ('none')
    BALANCED = ('balanced')
    SPEED = ('speed')
    SIZE = ('size')


def timestamp():
    return datetime.now().strftime('%Y%m%d%H%M%S')


@matrix_action
def lit(config, results):
    """Run tests for the selected configurations using llvm's lit."""
    yield run_lit(config.compiler[0], config.device[1], config.optimize[0])
    results[0].test_report.write(f"lit-{config.compiler[0]}-{config.optimize[0]}-{config.device[1]}-{timestamp()}.xunit")


def timestamp():
    return datetime.now().strftime('%Y%m%d%H%M%S')


@matrix_command(exit_code=[0, 1], test_report=FileReport(f"lit.xml") | JUnitReport())
def run_lit(toolchain, device, optimize):
    return ["lit", "--xunit-xml-output", f"lit.xml", "-D", f"toolchain={toolchain}", "-D", f"device={device}", "-D", f"optimize={optimize}", "src" ]


@matrix_filter
def filter_iar(config):
    return config.compiler == CompilerAxis.IAR


@matrix_filter
def filter_gcc_cm52(config):
    device = config.device.match('CM52*')
    compiler = config.compiler == CompilerAxis.GCC
    return device and compiler


if __name__ == "__main__":
    main()
