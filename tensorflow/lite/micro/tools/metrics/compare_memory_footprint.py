#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot memory footprint from stored Jenkins archifacts
"""
import argparse
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import zipfile

_STATUS_SUCCESS  = 200                       # Succesful read for requests.get(<url>)
_CSV_EXT         = '.csv'                    # File extension for CSV files
_DATA_LABELS     = ['text', 'data', 'bss']   # Data labels
_ALL_LABELS      = _DATA_LABELS + ['total']  # Same as above but including total data size
_DICT_KEYS       = _ALL_LABELS + ['sha']     # Same as above but including git SHA label
_SUBPLOT_ROWS    = int(np.ceil(np.sqrt(len(_ALL_LABELS))))
_SUBPLOT_COLS    = int(np.ceil(len(_ALL_LABELS)/_SUBPLOT_ROWS))
_LINEWIDTH       = 3
_MEDIUM_FONTSIZE = 16
_LARGE_FONTSIZE  = 18
_PLOT_OPACITY    = 1
_NBR_OF_XTICKS   = 10

def _download_missing_artifacts(top_dir:str, build_indx:list, jenkins_url:str, artifacts_folder:str):
    """
    Iterate through build_indx. Search for corresponding builds first locally as folders in top_dir.
    If they are not found, try downloading and extracting them from Jenkins
    """
    for indx in build_indx:
        indx = str(indx)
        local_path = os.path.join(top_dir, indx)
        if not os.path.exists(local_path):
            formatted_url = "{jenkins_url}/{build}/artifact/{artifacts_folder}/*zip*/{artifacts_folder}.zip". \
                format(jenkins_url=jenkins_url, build=indx, artifacts_folder=artifacts_folder)

            artifacts = requests.get(formatted_url)
            if artifacts.status_code == _STATUS_SUCCESS:
                os.mkdir(local_path)
                zipped = zipfile.ZipFile(io.BytesIO(artifacts.content))
                zipped.extractall(local_path)
                extracted_path = os.path.join(local_path, artifacts_folder)
                for csv_file in os.listdir(extracted_path):
                    os.rename(os.path.join(extracted_path, csv_file), os.path.join(local_path, csv_file))
                os.rmdir(extracted_path)


def _extract_memory_footprint(csv_file:str):
    """
    Parse CSV file and convert memory footprint information to dictionary
    If file is not found, or a specific key is not found albeit file is present, return corresponding value as NAN
    """
    retval =  {key: np.nan for key in _DICT_KEYS}

    if os.path.exists(csv_file):
        tmpdict = pd.read_csv(csv_file).to_dict()
        for key in tmpdict:
            retval[key] = tmpdict[key][0]
    return retval


def _unpack_local_artifacts(top_dir:str):
    """
    Compile all available artifact data in single dict, sorted by binary application or by architecture and build settings
    """
    # Extract all combinations of processor architectures, build types and binary applications
    all_binary_applications = []
    all_settings            = []
    all_build_indxs = [f for f in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, f))]

    for indx in all_build_indxs:
        subdir = os.path.join(top_dir, indx)
        settings_dir = [f for f in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, f))]
        for sdir in settings_dir:
            if sdir not in all_settings:
                all_settings.append(sdir)
            subdir2 = os.path.join(subdir, sdir)
            csv_files = [file for file in os.listdir(subdir2) if file.endswith(_CSV_EXT)]
            for cf in csv_files:
                ba = cf[:-len(_CSV_EXT)]
                if ba not in all_binary_applications:
                    all_binary_applications.append(ba)

    all_settings            = sorted(all_settings)
    all_binary_applications = sorted(all_binary_applications)
    all_build_indxs         = sorted(all_build_indxs)

    # Run through detected build combinations and compile data into dict
    application_setting = {}
    for app in all_binary_applications:
        application_setting[app] = {}
        for setting in all_settings:
            application_setting[app][setting] = {}
            for indx in all_build_indxs:
                csv_file = os.path.join(top_dir, indx, setting, app + _CSV_EXT)
                mem_footprint = _extract_memory_footprint(csv_file)
                for data_label in mem_footprint:
                    if data_label not in application_setting[app][setting]:
                        application_setting[app][setting][data_label] = []
                    application_setting[app][setting][data_label].append(mem_footprint[data_label])

    # Reorder data so that it can also be grouped based on setting
    setting_application = {}
    for setting in all_settings:
        setting_application[setting] = {}
        for app in all_binary_applications:
            setting_application[setting][app] = application_setting[app][setting]

    # Extract time for last commit
    datestr = [np.nan] * len(all_build_indxs)
    nan_values = [i for i in range(len(datestr))]

    for app in application_setting:
        if len(nan_values) == 0:
                break
        for setting in application_setting[app]:
            if len(nan_values) == 0:
                    break
            for i in nan_values:
                val = application_setting[app][setting]['date'][i]
                if pd.notna(val):
                    datestr[i] = val[2:10]
                    nan_values.remove(i)

    return application_setting, setting_application, datestr


def _set_xticklabels(ax, xticklabels):
    xdiff = int(np.ceil(len(xticklabels)/_NBR_OF_XTICKS))
    indx  = [i for i in range(0, len(xticklabels), xdiff)]
    ax.set_xticks([i+1 for i in indx])
    ax.set_xticklabels([xticklabels[i] for i in indx], rotation=-45)


def _format_subplot(ax, subplot_nbr, title, xticklabels, ylabel):
    if subplot_nbr == 1:
        ax.legend()
        ax.set_title(title, fontsize=_LARGE_FONTSIZE)
    if subplot_nbr > len(_ALL_LABELS) -_SUBPLOT_COLS:
        ax.set_xlabel('Commit date', fontsize=_MEDIUM_FONTSIZE)
    _set_xticklabels(ax, xticklabels)
    ax.set_ylabel(ylabel, fontsize=_MEDIUM_FONTSIZE)
    ax.grid(True)


def _plot_grouped_on_labels(plotdict, xticklabels, diff_from_start):
    x = [i+1 for i in range(len(xticklabels))]
    for key1 in plotdict:
        fig = plt.figure()
        for indx, label in enumerate(_ALL_LABELS):
            subplot_indx = indx+1
            ax = fig.add_subplot(_SUBPLOT_ROWS, _SUBPLOT_COLS, subplot_indx)
            for key2 in plotdict[key1]:
                plotdata = plotdict[key1][key2][label]
                if diff_from_start:
                    plotdata = [p - plotdata[0] for p in plotdata]
                ax.plot(x, plotdata, label=key2, linewidth=_LINEWIDTH, alpha=_PLOT_OPACITY)
            if diff_from_start:
                label += ", diff from start"
            _format_subplot(ax, subplot_indx, key1, xticklabels, label)


def _plot_extracted_data(mem_footprint:dict):
    """
    Plot memory footprint data
    """
    application_setting, setting_application, builds = _unpack_local_artifacts(top_dir=top_dir)
    x = np.array([i+1 for i in range(len(builds))])
    # Plot separately for each unique combination of architecture, build type and application
    for app in application_setting:
        for setting in application_setting[app]:
            plotdata = application_setting[app][setting]
            fig = plt.figure()
            ax = fig.add_subplot()
            bottom = np.zeros(np.size(builds))
            for label in _DATA_LABELS:
                y = np.array(plotdata[label])
                ax.bar(x, y, label=label, bottom=bottom, width=1)
                bottom = bottom + y
            ax.legend()
            ax.set_title(setting.replace("_", " ") + "; " + app)
            ax.set_ylabel('Memory')
            ax.set_ylim(ymin=0)
            _set_xticklabels(ax, builds)
            ax.set_xlabel('Commit date')
            ax.grid(True)

            plotdata = application_setting[app][setting]
            fig = plt.figure()
            ax = fig.add_subplot()
            for label in _DATA_LABELS:
                y = np.array(plotdata[label])
                ax.plot(x, y, label=label, linewidth=_LINEWIDTH, alpha=_PLOT_OPACITY)
            ax.legend()
            ax.set_title(setting.replace("_", " ") + "; " + app)
            ax.set_ylabel('Memory')
            ax.set_ylim(ymin=0)
            _set_xticklabels(ax, builds)
            ax.set_xlabel('Commit date')
            ax.grid(True)

    # Separate based on application, group based on architecture and build type
    _plot_grouped_on_labels(application_setting, builds, False)
    _plot_grouped_on_labels(application_setting, builds, True)

    # Group based on application, separate based on architecture and build type
    _plot_grouped_on_labels(setting_application, builds, False)
    _plot_grouped_on_labels(setting_application, builds, True)


def _purge_download_folders(top_dir:str, build_indx: list):
    """
    Delete previous download folders
    Only deletes folders contained in build_indx list
    """
    for build in build_indx:
        build = str(build)
        local_path = os.path.join(top_dir, build)
        if os.path.isdir(local_path):
            os.rmdir(local_path)


def _compare_memory_footprint(top_dir:str, jenkins_url:str, artifacts_folder:str, build_indx:list, purge:bool):
    """
    Parse through top_dir and plot memory footprint for builds
    """
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)

    if purge:
        _purge_download_folders(top_dir, build_indx)

    if jenkins_url is not None:
        _download_missing_artifacts(top_dir=top_dir, build_indx=build_indx, jenkins_url=jenkins_url, \
            artifacts_folder=artifacts_folder)

    mem_footprint = _unpack_local_artifacts(top_dir=top_dir)

    _plot_extracted_data(mem_footprint=mem_footprint)


if __name__ == '__main__':
    """
    Example input:
        top_dir             $(pwd)/local
        jenkins_url         'https://embed-ci.ml.arm.com/job/commit-regression-test-PLAYGROUND'
        artifacts_folder    reports
        build_start_at      1186
        build_end_at        1195
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--top_dir',
                    nargs='?',
                    default='local',
                    type=str,
                    help='Top directory for storing downloaded artifacts')

    parser.add_argument('--jenkins_url',
                    nargs='?',
                    type=str,
                    help='URL to Jenkins build')

    parser.add_argument('--artifacts_folder',
                    nargs='?',
                    default='report',
                    type=str,
                    help='Name of artifact subfolder / zip file on Jenkins')

    parser.add_argument('--build_start_at',
                    nargs='?',
                    default=1,
                    type=int)

    parser.add_argument('--build_end_at',
                    nargs='?',
                    default=100,
                    type=int)

    parser.add_argument('--purge',
                    action='store_true',
                    help='Remove all downloaded artifacts')

    args = parser.parse_args()

    top_dir          = args.top_dir or os.getcwd()
    jenkins_url      = args.jenkins_url
    artifacts_folder = args.artifacts_folder
    build_indx       = [indx for indx in range(args.build_start_at, args.build_end_at+1)]
    purge            = args.purge

    _compare_memory_footprint(top_dir=top_dir,
                            jenkins_url=jenkins_url,
                            artifacts_folder=artifacts_folder,
                            build_indx=build_indx,
                            purge=purge)

    plt.show()
