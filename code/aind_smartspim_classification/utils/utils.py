#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:23:13 2022

@author: nicholas.lusk
@Modified by: camilo.laiton
"""

import inspect
import json
import logging
import multiprocessing
import os
import platform
import struct
import subprocess
import time
from datetime import datetime
from multiprocessing.managers import BaseManager, NamespaceProxy
from pathlib import Path
from typing import List, Optional

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import psutil
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing)
from scipy import ndimage as ndi
from scipy.signal import argrelmin

from .._shared.types import PathLike


def find_good_blocks(img, counts, chunk, ds=3):
    """
    Function to Identify good blocks to process
    using downsampled zarr

    Parameters
    ----------
    img : da.array
        dask array of low resolution image
    counts : list[int]
        chunks per dimension of level 0 array.
    chunk : int
        chunk size used for cell detection
    ds : int
        the factor by which the array is downsampled

    Returns
    -------
    block_dict : dict
        dictionary with information on which blocks to
        process. key = block number and value = bool

    """
    if isinstance(img, dask.array.core.Array):
        img = np.asarray(img)

    img = ndi.gaussian_filter(img, sigma=5.0, mode="constant", cval=0)
    count, bin_count = np.histogram(
        img.astype("uint16"), bins=2**16, range=(0, 2**16), density=True
    )

    try:
        thresh = argrelmin(count, order=10)[0][0]
    except IndexError:
        thresh = 100

    img_binary = np.where(img >= thresh, 1, 0)

    cz = int(chunk / 2**ds)
    dims = list(img_binary.shape)

    b = 0
    block_dict = {}

    """ there are rare occasions where the level 0 and
    level 3 array disagree on chuncks so have some
    catches to account for that """
    for z in range(counts[0]):
        z_l, z_u = z * cz, (z + 1) * cz
        if z_l > dims[0] - 1:
            z_l = dims[0] - 2
        if z_u > dims[0] - 1:
            z_u = dims[0] - 1
        for y in range(counts[1]):
            y_l, y_u = y * cz, (y + 1) * cz
            if y_l > dims[1] - 1:
                y_l = dims[1] - 2
            if y_u > dims[1] - 1:
                y_u = dims[1] - 1
            for x in range(counts[2]):
                x_l, x_u = x * cz, (x + 1) * cz
                if x_l > dims[2] - 1:
                    x_l = dims[2] - 2
                if x_u > dims[2] - 1:
                    x_u = dims[2] - 1

                block = img_binary[
                    z_l:z_u,
                    y_l:y_u,
                    x_l:x_u,
                ]

                if np.sum(block) > 0:
                    block_dict[b] = True
                else:
                    block_dict[b] = False

                b += 1

    return block_dict


def execute_command_helper(
    command: str,
    print_command: bool = False,
):
    """
    Execute a shell command.

    Parameters
    ------------------------

    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------

    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def create_logger(output_log_path: PathLike):
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """

    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/classification_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def read_json_as_dict(filepath: str):
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.
    Parameters
    ------------------------
    dest_dir: PathLike
        Path where the folder will be created if it does not exist.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.
    Raises
    ------------------------
    OSError:
        if the folder exists.
    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def volume_orientation(acquisition_params: dict):
    """
    Uses the acquisition orientation to set the cross-section
    orientation in the neuroglancer links

    Parameters
    ----------
    acquisition_params : dict
        acquisition paramenters from the processing manifest

    Raises
    ------
    ValueError
        if a brain is aquired in a way other than those predifined here

    Returns
    -------
    orientation : list
        orientation values for the neuroglancer link

    """

    acquired = ["", "", ""]

    for axis in acquisition_params["axes"]:
        acquired[axis["dimension"]] = axis["direction"][0]

    acquired = "".join(acquired)

    if acquired in ["SPR", "SPL"]:
        orientation = [0.5, 0.5, 0.5, -0.5]
    elif acquired == "SAL":
        orientation = [0.5, 0.5, -0.5, 0.5]
    elif acquired == "IAR":
        orientation = [0.5, -0.5, 0.5, 0.5]
    elif acquired == "RAS":
        orientation = [np.cos(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]
    elif acquired == "RPI":
        orientation = [np.cos(np.pi / 4), 0.0, 0.0, -np.cos(np.pi / 4)]
    elif acquired == "LAI":
        orientation = [0.0, np.cos(np.pi / 4), -np.cos(np.pi / 4), 0.0]
    else:
        raise ValueError(
            "Acquisition orientation: {acquired} has unknown NG parameters"
        )

    return orientation


def wavelength_to_hex_alternate(wavelength: int) -> int:
    """
    Converts wavelengths to hex value, taking fpbase.org spectra viewer
    as a guide.
    Fluorescent proteins querried:
    mTFP1,
    EGFP,
    SYFP2,
    mbanana,
    morange,
    mtomato,
    mcherry,
    mraspberry,
    mplum

    Parameters
    ------------------------
    wavelength: int
        Integer value representing wavelength.

    Returns
    ------------------------
    int:
        Hex value color.
    """

    color_map = {
        500: 0x61ABFD,  # RUDDY BLUE, mTFP/mTurquoise
        530: 0x92FF42,  # CHARTREUSE,   EGFP
        540: 0xE4FE41,  # CHARTREUSE, SYFP2
        560: 0xF3D038,  # MUSTARD, mBanana
        580: 0xEAB032,  # XANTHOUS, mOrange
        600: 0xF15F22,  # GIANTS ORANGE, tdTomato/mScarlet
        630: 0xED1C24,  # RED, mCherry
        680: 0xC51E1F,  # FIRE ENGINE RED, mRaspberry
        700: 0xA81F1F,  # FIRE BRICK, mPlum
    }

    for ub, hex_val in color_map.items():
        if wavelength <= ub:  # Inclusive
            return hex_val
    return hex_val  # hex_val is set to the last color in for loop


def calculate_dynamic_range(image_path: PathLike, percentile: 99, level: 3):
    """
    Calculates the default dynamic range for teh neuroglancer link
    using a defined percentile from the downsampled zarr

    Parameters
    ----------
    image_path : PathLike
        location of the zarr used for classification
    percentile : 99
        The top percentile value for setting the dynamic range
    level : 3
        level of zarr to use for calculating percentile

    Returns
    -------
    dynamic_ranges : list
        The dynamic range and window range values for zarr

    """

    img = da.from_zarr(image_path, str(level)).squeeze()
    range_max = da.percentile(img.flatten(), percentile).compute()[0]
    window_max = int(range_max * 1.5)
    dynamic_ranges = [int(range_max), window_max]

    return dynamic_ranges


class ObjProxy(NamespaceProxy):
    """Returns a proxy instance for any user defined data-type. The proxy instance will have the namespace and
    functions of the data-type (except private/protected callables/attributes). Furthermore, the proxy will be
    pickable and can its state can be shared among different processes."""

    @classmethod
    def populate_obj_attributes(cls, real_cls):
        """
        Populates attributes of the proxy object
        """
        DISALLOWED = set(dir(cls))
        ALLOWED = [
            "__sizeof__",
            "__eq__",
            "__ne__",
            "__le__",
            "__repr__",
            "__dict__",
            "__lt__",
            "__gt__",
        ]
        DISALLOWED.add("__class__")
        new_dict = {}
        for attr, value in inspect.getmembers(real_cls, callable):
            if attr not in DISALLOWED or attr in ALLOWED:
                new_dict[attr] = cls._proxy_wrap(attr)
        return new_dict

    @staticmethod
    def _proxy_wrap(attr):
        """
        This method creates function that calls the proxified object's method.
        """

        def f(self, *args, **kwargs):
            """
            Function that calls the proxified object's method.
            """
            return self._callmethod(attr, args, kwargs)

        return f


def buf_builder(x, y, z, buf_):
    """builds the buffer"""
    pt_buf = struct.pack("<3f", x, y, z)
    buf_.extend(pt_buf)


attributes = ObjProxy.populate_obj_attributes(bytearray)
bytearrayProxy = type("bytearrayProxy", (ObjProxy,), attributes)


def generate_precomputed_cells(cells, precompute_path, configs):
    """
    Function for saving precomputed annotation layer

    Parameters
    -----------------

    cells: dict
        output of the xmltodict function for importing cell locations
    precomputed_path: str
        path to where you want to save the precomputed files
    comfigs: dict
        data on the space that the data will be viewed

    """

    BaseManager.register(
        "bytearray",
        bytearray,
        bytearrayProxy,
        exposed=tuple(dir(bytearrayProxy)),
    )
    manager = BaseManager()
    manager.start()

    buf = manager.bytearray()

    cell_list = []
    for idx, cell in cells.iterrows():
        cell_list.append([int(cell["z"]), int(cell["y"]), int(cell["x"])])

    l_bounds = np.min(cell_list, axis=0)
    u_bounds = np.max(cell_list, axis=0)

    output_path = os.path.join(precompute_path, "spatial0")
    create_folder(output_path)

    metadata = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": dict(
            (key, configs["dimensions"][key]) for key in ("z", "y", "x")
        ),
        "lower_bound": [float(x) for x in l_bounds],
        "upper_bound": [float(x) for x in u_bounds],
        "annotation_type": "point",
        "properties": [],
        "relationships": [],
        "by_id": {
            "key": "by_id",
        },
        "spatial": [
            {
                "key": "spatial0",
                "grid_shape": [1] * configs["rank"],
                "chunk_size": [max(1, float(x)) for x in u_bounds - l_bounds],
                "limit": len(cell_list),
            },
        ],
    }

    with open(os.path.join(precompute_path, "info"), "w") as f:
        f.write(json.dumps(metadata))

    with open(os.path.join(output_path, "0_0_0"), "wb") as outfile:
        start_t = time.time()

        total_count = len(cell_list)  # coordinates is a list of tuples (x,y,z)

        print("Running multiprocessing")

        if not isinstance(buf, type(None)):
            buf.extend(struct.pack("<Q", total_count))

            with multiprocessing.Pool(processes=os.cpu_count()) as p:
                p.starmap(buf_builder, [(x, y, z, buf) for (x, y, z) in cell_list])

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack("<%sQ" % len(cell_list), *range(len(cell_list)))
            buf.extend(id_buf)
        else:
            buf = struct.pack("<Q", total_count)

            for x, y, z in cell_list:
                pt_buf = struct.pack("<3f", x, y, z)
                buf += pt_buf

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack("<%sQ" % len(cell_list), *range(len(cell_list)))
            buf += id_buf

        print("Building file took {0} minutes".format((time.time() - start_t) / 60))

        outfile.write(bytes(buf))


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: PathLike,
    processor_full_name: str,
    pipeline_version: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        note="Metadata for classification step",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata of cell segmentation \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(output_directory=dest_processing)


def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_points[:min_len], memory_usages[:min_len], label="Memory Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight")


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    memory = get_memory_limit_bytes()

    if memory:
        memory = int(memory)
        memory = get_size(memory)

    slurm_id = os.environ.get("SLURM_JOBID")
    # System info
    sep = "=" * 20
    logger.info(f"{sep} Machine Information {sep}")
    logger.info(f"Assigned cores: {get_cpu_limit()}")
    logger.info(f"Assigned memory: {memory} GBs")
    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}")
    logger.info(f"Is pipeline execution in SLURM?: {bool(slurm_id)}")
    logger.info(f"SLURM ID: {slurm_id}")
    logger.info(f"SLURM GPUs: {os.environ.get('SLURM_JOB_GPUS')}")
    logger.info(f"SLURM CPUs: {os.environ.get('SLURM_JOB_CPUS_PER_NODE')}")
    logger.info(
        f"SLURM variables {[( k, v ) for k, v in os.environ.items() if 'SLURM' in k]}"
    )

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(
        f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}"
    )

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


def get_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    # Trying to get CPU cores from Code Ocean
    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1

    # Trying to get CPU cores from SLURM
    slurm_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")

    # Total cpus in node SLURM_CPUS_ON_NODE
    if slurm_cpus:
        return slurm_cpus

    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())

        container_cpus = cfs_quota_us // cfs_period_us

    except FileNotFoundError as e:
        container_cpus = 0

    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def get_memory_limit_bytes():
    """
    Gets the best estimate of the memory limit (in bytes) for the current job.
    Order of precedence:
    1. CO_MEMORY environment variable (assumed in GB)
    2. Cgroup memory limit (from /sys/fs/cgroup/)
    3. SLURM environment variables
    4. psutil system memory (total)
    """
    # 1. CO_MEMORY (in GB)
    memory_env = os.environ.get("CO_MEMORY")
    if memory_env:
        try:
            return int(memory_env)  # Convert GB → bytes
        except ValueError:
            pass  # Invalid format, fallback

    # 2. cgroup memory limit (in bytes)
    cgroup_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    try:
        with open(cgroup_path, "r") as f:
            mem_bytes = int(f.read().strip())
            # Some systems report a huge number when no limit is set
            if mem_bytes < 1 << 50:  # Filter out values >1PB
                return mem_bytes
    except FileNotFoundError:
        pass

    # 3. SLURM memory allocation
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")  # in MB
    if mem_per_node:
        try:
            return int(mem_per_node) * 1024**2  # MB → bytes
        except ValueError:
            pass

    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")  # in MB
    cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if mem_per_cpu and cpus:
        try:
            return int(mem_per_cpu) * int(cpus) * 1024**2  # MB → bytes
        except ValueError:
            pass

    # 4. Fallback: system-wide total memory
    return psutil.virtual_memory().total


def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.

    Parameters
    ------------------------

    obj: object
        Object that wants to be validated.

    Returns
    ------------------------

    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """

    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True

    return False


def save_dict_as_json(
    filename: str, dictionary: dict, verbose: Optional[bool] = False
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------

    filename: str
        Name of the json file.

    dictionary: dict
        Dictionary that will be saved as json.

    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)

    if verbose:
        print(f"- Json file saved: {filename}")
