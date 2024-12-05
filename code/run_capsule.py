"""
Main file to execute the smartspim classification
in code ocean
"""

import os
import sys
import shutil
from glob import glob
from pathlib import Path
from typing import List, Tuple

from aind_smartspim_classification import classification
from aind_smartspim_classification.params import get_yaml
from aind_smartspim_classification.utils import utils


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

    default_config[
        "input_channel"
    ] = f"{pipeline_config['segmentation']['channel']}.zarr"
    default_config[
        "background_channel"
    ] = f"{pipeline_config['segmentation']['background_channel']}.zarr"
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

def get_detection_data(results_folder, dataset, channel, bucket = 'aind-open-data'):

    s3_path = f"s3://{bucket}/{dataset}/image_cell_segmentation/{channel}/"

    # Copying final processing manifest
    for out in utils.execute_command_helper(
        f"aws s3 cp {s3_path} {results_folder}/cell_{channel}/ --recursive"
    ):
        print(out)


def run():
    """
    Main function to execute the smartspim segmentation
    in code ocean
    """

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")
    scratch_folder = os.path.abspath("../scratch")

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = []

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    pipeline_config, smartspim_dataset_name = get_data_config(data_folder=data_folder)

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
        ] = f"{data_folder}/resnet_smartspim_18_test.keras"
        #f"{data_folder}/smartspim_18_model/smartspim_18_model.h5"

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

    default_config[
        "save_path"
    ] = f"{results_folder}/cell_{pipeline_config['segmentation']['channel']}"

    # want to shutil segmentation data to results folder if detection was run
    default_config[
        "metadata_path"
    ] = f"{results_folder}/cell_{pipeline_config['segmentation']['channel']}/metadata"


    if 'classify' in mode:
        get_detection_data(
            results_folder=results_folder,
            dataset=smartspim_dataset_name,
            channel=pipeline_config['segmentation']['channel']
        )

    if 'detect' in mode:
        shutil.copytree(
            f"{data_folder}/cell_{pipeline_config['segmentation']['channel']}/",
            f"{results_folder}/cell_{pipeline_config['segmentation']['channel']}/",
        )

    print("Initial cell classification config: ", default_config)

    # combine configs
    smartspim_config = set_up_pipeline_parameters(
        pipeline_config=pipeline_config, default_config=default_config
    )

    smartspim_config["name"] = smartspim_dataset_name

    print("Final cell classification config: ", smartspim_config)

    classification.main(
        data_folder=Path(data_folder),
        output_segmented_folder=Path(results_folder),
        intermediate_segmented_folder=Path(scratch_folder),
        smartspim_config=smartspim_config,
    )


if __name__ == "__main__":
    run()
