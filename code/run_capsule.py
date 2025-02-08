"""
Main file to execute the smartspim classification
in code ocean
"""

import os
import shutil
import sys
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from aind_smartspim_classification import classification
from aind_smartspim_classification.params import get_yaml
from aind_smartspim_classification.utils import utils


def parse_cell_xml(xml_path: str) -> np.array:
    """
    Parses a XML with cell proposals coming from
    the aind-smartspim-segmentation capsule.

    Returns
    -------
    np.array
        Array with the cell proposals in order ZYX
    """

    # Load and parse the XML file
    tree = ET.parse(xml_path)  # Replace 'file.xml' with your file path
    root = tree.getroot()

    # Extract image filename
    # image_filename = root.find("./Image_Properties/Image_Filename").text

    # Extract marker data
    marker_data = []
    for marker in root.findall("./Marker_Data/Marker_Type/Marker"):
        marker_x = int(marker.find("MarkerX").text)
        marker_y = int(marker.find("MarkerY").text)
        marker_z = int(marker.find("MarkerZ").text)
        marker_data.append([marker_z, marker_y, marker_x])

    return np.array(marker_data, dtype=np.uint32)


def get_data_config(
    data_folder: str,
    processing_manifest_path: str = "segmentation_processing_manifest*",
    data_description_path: str = "data_description.json",
) -> Tuple:
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: str
        Path for the processing manifest

    data_description_path: str
        Path for the data description

    Returns
    -----------
    Tuple[Dict, str]
        Dict: Empty dictionary if the path does not exist,
        dictionary with the data otherwise.

        Str: Empty string if the processing manifest
        was not found
    """

    # Returning first smartspim dataset found
    # Doing this because of Code Ocean, ideally we would have
    # a single dataset in the pipeline

    print(f"Manifest Path: {data_folder}/{processing_manifest_path}")

    try:
        derivatives_dict = utils.read_json_as_dict(
            glob(f"{data_folder}/{processing_manifest_path}")[0]
        )
    except:
        derivatives_dict = utils.read_json_as_dict(
            glob(f"{data_folder}/processing_manifest_*")[0]
        )
    data_description_dict = utils.read_json_as_dict(
        f"{data_folder}/{data_description_path}"
    )

    smartspim_dataset = data_description_dict["name"]

    return derivatives_dict, smartspim_dataset


def set_up_pipeline_parameters(pipeline_config: dict, default_config: dict):
    """
    Sets up smartspim stitching parameters that come from the
    pipeline configuration

    Parameters
    -----------
    smartspim_dataset: str
        String with the smartspim dataset name

    pipeline_config: dict
        Dictionary that comes with the parameters
        for the pipeline described in the
        processing_manifest.json

    default_config: dict
        Dictionary that has all the default
        parameters to execute this capsule with
        smartspim data

    Returns
    -----------
    Dict
        Dictionary with the combined parameters
    """

    default_config["input_channel"] = (
        f"{pipeline_config['segmentation']['channel']}.zarr"
    )
    default_config["background_channel"] = (
        f"{pipeline_config['segmentation']['background_channel']}.zarr"
    )
    default_config["channel"] = pipeline_config["segmentation"]["channel"]
    default_config["input_scale"] = pipeline_config["segmentation"]["input_scale"]
    default_config["chunk_size"] = int(pipeline_config["segmentation"]["chunksize"])

    return default_config


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def get_detection_data(results_folder, dataset, channel, bucket="aind-open-data"):
    """
    Gets the detection data from the bucket
    """

    s3_path = f"s3://{bucket}/{dataset}/image_cell_segmentation/{channel}/"

    # Copying final processing manifest
    for out in utils.execute_command_helper(
        f"aws s3 cp {s3_path} {results_folder}/cell_{channel}/ --recursive"
    ):
        print(out)


def downsample_cell_locations(coordinates: np.ndarray, downscale_factors: list):
    """
    Adjusts ZYX coordinates to match the resolution of a lower level in a multiscale image.

    Parameters
    ----------
    coordinates : np.ndarray
        A 2D array of shape (N, 3), where each row contains the ZYX coordinates
        in the original resolution.

    downscale_factors : list or tuple
        A tuple (z_steps, y_steps, x_steps) where each value is the number of
        downsampling steps (base 2) applied to the Z, Y, and X dimensions.

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, 3) containing the downscaled ZYX coordinates.
    """
    coordinates = np.asarray(coordinates)
    downscale_factors = np.asarray(downscale_factors)

    # Compute downscale factors as 2^steps for each dimension
    downscale_factors = np.power(2, downscale_factors)

    downscaled_coordinates = coordinates / downscale_factors

    # Rounding to closest integer
    # Might move the center of cell by a couple voxels (max 3?)
    downscaled_coordinates = np.floor(downscaled_coordinates).astype(int)

    return downscaled_coordinates


def copy_detection_files(
    data_folder: str,
    results_folder: str,
    proposal_folder: str,
):
    """
    The detection files contain metadata about the
    identification of cell proposals, runtimes and
    visualization files.

    Parameters
    ----------
    data_folder: str
        Folder where the data is stored.

    results_folder: str
        Path where we want to store the results.

    proposal_folder: str
        Folder where the cell proposals are stored.

    """
    detected_metadata_path = f"{data_folder}/{proposal_folder}/metadata"
    detected_visualization_path = f"{data_folder}/{proposal_folder}/visualization"

    dest_detected_metadata_path = (
        f"{results_folder}/{proposal_folder}/proposals/metadata"
    )
    dest_detected_visualization_path = (
        f"{results_folder}/{proposal_folder}/proposals/visualization"
    )

    # If detected metadata exists, we should copy it
    if os.path.exists(detected_metadata_path):
        utils.create_folder(dest_dir=os.path.dirname(dest_detected_metadata_path))
        shutil.copytree(
            detected_metadata_path, dest_detected_metadata_path, dirs_exist_ok=True
        )
        print(f"Copied detection metadata to {dest_detected_metadata_path}")

    else:
        print(f"Detected metadata path not provided: {detected_metadata_path}")

    # If detected visualization exists, we should copy it
    if os.path.exists(detected_visualization_path):
        utils.create_folder(dest_dir=os.path.dirname(dest_detected_visualization_path))

        shutil.copytree(
            detected_visualization_path,
            dest_detected_visualization_path,
            dirs_exist_ok=True,
        )
        print(f"Copied detection visualization to {dest_detected_visualization_path}")

    else:
        print(
            f"Detected visualization path not provided: {detected_visualization_path}"
        )


def run():
    """
    Main function to execute the smartspim segmentation
    in code ocean
    """

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")
    # scratch_folder = os.path.abspath("../scratch")

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = []

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    pipeline_config, smartspim_dataset_name = get_data_config(
        data_folder=data_folder,
    )

    segmentation_info = pipeline_config.get("segmentation")

    if segmentation_info is None:
        raise ValueError("Please, provide segmentation channels.")

    channel_to_process = segmentation_info.get("channel")

    # Note: The dispatcher capsule creates a single config with
    # the channels. If the channel key does not exist, it means
    # there are no segmentation channels splitted
    if channel_to_process is not None:

        # Folder where the detection files are stored from the previous step
        proposal_folder = f"cell_{channel_to_process}"

        # get default configs
        mode = str(sys.argv[1:])
        mode = mode.replace("[", "").replace("]", "").casefold()

        print(f"Classification mode: {mode}")

        if "nuclei" in mode:
            default_config = get_yaml(
                os.path.abspath(
                    "aind_smartspim_classification/params/default_classify_config.yml"
                )
            )
            default_config["cellfinder_params"][
                "trained_model"
            ] = f"{data_folder}/smartspim_18_model/smartspim_18_model.h5"

        elif "cytosolic":
            default_config = get_yaml(
                os.path.abspath(
                    "aind_smartspim_classification/params/cytosolic_classify_config.yml"
                )
            )

            default_config["cellfinder_params"][
                "trained_model"
            ] = f"{data_folder}/cytosolic_model/2024_10_09_smartspim_18_cytosolic.h5"
        else:
            raise NotImplementedError(f"The mode {mode} has not been implemented")

        # add paths to default_config
        default_config["input_data"] = os.path.abspath(
            pipeline_config["segmentation"]["input_data"]
        )
        print("Files in path: ", os.listdir(default_config["input_data"]))

        default_config["save_path"] = f"{results_folder}/{proposal_folder}"

        # want to shutil segmentation data to results folder if detection was run
        default_config["metadata_path"] = f"{results_folder}/{proposal_folder}/metadata"

        if "classify" in mode:
            get_detection_data(
                results_folder=results_folder,
                dataset=smartspim_dataset_name,
                channel=pipeline_config["segmentation"]["channel"],
            )

        if "detect" in mode:
            shutil.copytree(
                f"{data_folder}/{proposal_folder}/",
                f"{results_folder}/{proposal_folder}/",
            )

        print("Initial cell classification config: ", default_config)

        # combine configs
        smartspim_config = set_up_pipeline_parameters(
            pipeline_config=pipeline_config, default_config=default_config
        )

        smartspim_config["name"] = smartspim_dataset_name

        print("Final cell classification config: ", smartspim_config)

        # Remove comment when new detection is deployed
        # cell_proposals = np.load(f"{data_folder}/spots.npy")
        cell_proposals = parse_cell_xml(
            f"{data_folder}/{proposal_folder}/detected_cells.xml"
        )

        # Copying detection files
        copy_detection_files(
            data_folder=data_folder,
            results_folder=results_folder,
            proposal_folder=proposal_folder,
        )

        # Downsample cells to the prediction scale
        cell_proposals = downsample_cell_locations(
            coordinates=cell_proposals,
            downscale_factors=[int(smartspim_config["downsample"])] * 3,
        )

        print("Spots proposals: ", cell_proposals.shape)
        print("Cellfinder params: ", smartspim_config["cellfinder_params"])

        classification.main(
            smartspim_config=smartspim_config,
            cell_proposals=cell_proposals,
        )

    else:
        print(f"No segmentation channel, pipeline config: {pipeline_config}")
        utils.save_dict_as_json(
            filename=f"{results_folder}/segmentation_processing_manifest_no_class.json",
            dictionary=pipeline_config,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    run()
