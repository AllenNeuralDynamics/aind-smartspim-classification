"""
Module for the classification of smartspim datasets
"""

import json
import logging
import multiprocessing
import os
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import pandas as pd
import torch
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    concatenate_lazy_data, recover_global_position, unpad_global_coords)
from aind_large_scale_prediction.io import ImageReaderFactory
from natsort import natsorted
from ng_link import NgState

from .__init__ import __maintainers__, __pipeline_version__, __version__
from ._shared.types import PathLike
from .utils import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def extract_centered_3d_block(
    big_block: np.array, center: Tuple, size: Tuple, pad_value: Optional[int] = 0
):
    """
    Extract a centered 3D block around a specified center and pad it if needed.

    Parameters
    ----------
    big_block: np.array
        numpy array of shape (Z, Y, X) representing
        the larger 3D block.

    center: Tuple
        Tuple (z_center, y_center, x_center),
        the center of the block to extract.

    size: Tuple
        (z_size, y_size, x_size), the size of the block
        to extract.
    pad_value: Optional[int]
        Value to use for padding if the block is out of bounds.

    Returns
    -------
        np.array
            Padded block of shape (z_size, y_size, x_size).
    """
    z_center, y_center, x_center = center
    z_size, y_size, x_size = size

    # Dimensions of the larger block
    D, Z, Y, X = big_block.shape

    # Compute start and end indices
    z_start = z_center - z_size // 2
    y_start = y_center - y_size // 2
    x_start = x_center - x_size // 2

    z_end = z_start + z_size
    y_end = y_start + y_size
    x_end = x_start + x_size

    # Ensure the indices are within bounds
    z_start_valid = max(z_start, 0)
    y_start_valid = max(y_start, 0)
    x_start_valid = max(x_start, 0)

    z_end_valid = min(z_end, Z)
    y_end_valid = min(y_end, Y)
    x_end_valid = min(x_end, X)

    # Extract the valid part of the block
    extracted_block = big_block[
        :,
        z_start_valid:z_end_valid,
        y_start_valid:y_end_valid,
        x_start_valid:x_end_valid,
    ]

    # Compute padding widths for each dimension
    z_pad_before = max(0, -z_start)
    y_pad_before = max(0, -y_start)
    x_pad_before = max(0, -x_start)

    z_pad_after = max(0, z_end - Z)
    y_pad_after = max(0, y_end - Y)
    x_pad_after = max(0, x_end - X)

    pad_width = (
        (0, 0),
        (z_pad_before, z_pad_after),
        (y_pad_before, y_pad_after),
        (x_pad_before, x_pad_after),
    )
    # print("Pad width: ", pad_width)

    # Pad the extracted block to achieve the requested size
    padded_block = np.pad(
        extracted_block,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    )

    return padded_block


def upsample_position(position: List[int], downsample_factor: Tuple[int]):
    """
    Upsample a ZYX position from a downsampled image
    to the original resolution.

    Parameters
    ----------
    position: List[int]
        Tuple or list containing (z, y, x) coordinates in the downsampled image
    downsample_factor: Tuple[int]
        Number of times the image was downsampled by 2 in each axis

    Returns
    -------
    List[int]
        Upsampled (z, y, x) coordinates in the original image resolution
    """
    z, y, x = position

    # Upsample each coordinate by multiplying by 2^downsample_factor
    upsampled_z = z * (2**downsample_factor)
    upsampled_y = y * (2**downsample_factor)
    upsampled_x = x * (2**downsample_factor)

    upsampled_z = upsampled_z.astype(np.uint32)
    upsampled_y = upsampled_y.astype(np.uint32)
    upsampled_x = upsampled_x.astype(np.uint32)
    return upsampled_z, upsampled_y, upsampled_x


def cell_classification(
    smartspim_config: Dict,
    logger: logging.Logger,
    cell_proposals: np.array,
    prediction_chunksize: Optional[Tuple] = (128, 128, 128),
    target_size_mb: Optional[int] = 2048,
    n_workers: Optional[int] = 0,
    super_chunksize: Optional[Tuple] = None,
):
    """
    Runs cell classification based on a set of proposals
    in a whole lightsheet brain.

    Parameters
    ----------
    smartspim_config: Dict
        Dictionary with the configuration to process
        a whole smartspim brain with cell proposals.

    logger: logging.Logger,
        Logging object

    cell_proposals: np.array
        Proposals in the whole brain. These should be
        ZYX coordinates.

    prediction_chunksize: Optional[Tuple]
        Chunksize that will be used to run predictions on
        blocks of data. This means that the large-scale
        prediction package will pull a superchunk and
        then small chunks will be pulled.
        Default: (128, 128, 128)

    target_size_mb: Optional[int] = 2048
        Target size in MB for the shared memory compartment.
        This is used of the superchunksize is not provided.

    n_workers: Optional[int] = 0
        Number of workers that will be processing data.
        Leave this as 0, but if more GPUs are available,
        there's still work to do to allow this.

    super_chunksize: Optional[Tuple] = None
        Chunksize that will be pulled from the cloud in
        a single call. prediction_chunksize > super_chunksize.

    """
    start_date_time = datetime.now()

    data_processes = []

    image_path = Path(smartspim_config["input_data"]).joinpath(
        f"{smartspim_config['input_channel']}"
    )

    background_path = Path(smartspim_config["input_data"]).joinpath(
        f"{smartspim_config['background_channel']}"
    )

    mask_path = Path(smartspim_config["input_data"]).joinpath(
        f"{smartspim_config['input_channel']}/{smartspim_config['mask_scale']}"
    )

    print(
        f" Image Path: {image_path} -- mask path: {mask_path} - scale: {smartspim_config['downsample']}"
    )

    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    axis_pad = 6
    overlap_prediction_chunksize = (axis_pad, axis_pad, axis_pad)

    if background_path:
        logger.info(f"Using background path in {background_path} with {image_path}")
        lazy_data = concatenate_lazy_data(
            dataset_paths=[image_path, background_path],
            multiscales=[
                smartspim_config["downsample"],
                smartspim_config["downsample"],
            ],
            concat_axis=-4,
        )
        overlap_prediction_chunksize = (0, axis_pad, axis_pad, axis_pad)
        prediction_chunksize = (lazy_data.shape[-4],) + prediction_chunksize

        logger.info(
            f"Background path provided! New prediction chunksize: {prediction_chunksize} - New overlap: {overlap_prediction_chunksize}"
        )

    else:
        # No segmentation mask
        lazy_data = (
            ImageReaderFactory()
            .create(
                data_path=str(image_path),
                parse_path=False,
                multiscale=smartspim_config["downsample"],
            )
            .as_dask_array()
        )

    print("Loaded lazy data: ", lazy_data)
    batch_size = 1
    dtype = np.float32
    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=dtype,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=None,  # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        override_suggested_cpus=False,
        drop_last=True,
        locked_array=False,
    )

    logger.info(
        f"Running cell classification in chunked data. Prediction chunksize: {prediction_chunksize} - Overlap chunksize: {overlap_prediction_chunksize}"
    )

    cube_width = smartspim_config["cellfinder_params"]["cube_width"]
    cube_height = smartspim_config["cellfinder_params"]["cube_height"]
    cube_depth = smartspim_config["cellfinder_params"]["cube_depth"]

    # Load model defaults to inference mode
    model = keras.models.load_model(
        smartspim_config["cellfinder_params"]["trained_model"]
    )
    model.trainable = False
    ORIG_AXIS_ORDER = ["Z", "Y", "X"]

    total_batches = sum(zarr_dataset.internal_slice_sum) / batch_size
    logger.info(
        f"Total batches: {total_batches} - cell proposals: {cell_proposals.shape[0]}"
    )

    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = int(0.80 * total_memory)

    logger.info(f"GPU total memory: {total_memory} - Target memory: {target_memory}")

    block_size_bytes = (
        np.prod((cube_depth, cube_height, cube_width, 2)) * np.dtype(dtype).itemsize
    )
    # Estimate the number of blocks that fit within 80% memory
    max_blocks = target_memory // block_size_bytes
    logger.info(f"Maximum blocks: {max_blocks}")

    curr_blocks = 0
    blocks_to_classify = []
    picked_proposals = []
    processed_cells = 0
    # Zarr at a downsampled resolution
    # Cell locations should be at this level
    for i, sample in enumerate(zarr_data_loader):
        logger.info(
            f"Batch [{i} | {total_batches}]: processed_cells {processed_cells} blocks: {curr_blocks} - Max blocks: {max_blocks} {sample.batch_tensor.shape} - Pinned?: {sample.batch_tensor.is_pinned()} - dtype: {sample.batch_tensor.dtype} - device: {sample.batch_tensor.device}"
        )

        data_block = sample.batch_tensor[0, ...]  # .permute(-1, -2, -3, -4)
        batch_super_chunk = sample.batch_super_chunk[0]
        batch_internal_slice = sample.batch_internal_slice

        (
            global_coord_pos,
            global_coord_positions_start,
            global_coord_positions_end,
        ) = recover_global_position(
            super_chunk_slice=batch_super_chunk,
            internal_slices=batch_internal_slice,
        )

        unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
            global_coord_pos=global_coord_pos[-3:],
            block_shape=data_block.shape[-3:],
            overlap_prediction_chunksize=overlap_prediction_chunksize[-3:],
            dataset_shape=zarr_dataset.lazy_data.shape[
                -3:
            ],  # zarr_dataset.lazy_data.shape,
        )
        # print("Global pos: ", global_coord_pos, unpadded_global_slice, data_block.shape)

        proposals_in_block = cell_proposals[
            (
                cell_proposals[:, 0] >= unpadded_global_slice[0].start
            )  # within Z boundaries
            & (cell_proposals[:, 0] < unpadded_global_slice[0].stop)
            & (
                cell_proposals[:, 1] >= unpadded_global_slice[1].start
            )  # Within Y boundaries
            & (cell_proposals[:, 1] < unpadded_global_slice[1].stop)
            & (
                cell_proposals[:, 2] >= unpadded_global_slice[2].start
            )  # Within X boundaries
            & (cell_proposals[:, 2] < unpadded_global_slice[2].stop)
        ]

        global_pos_name = "_".join(
            [
                f"{ORIG_AXIS_ORDER[idx]}_{sl.start}_{sl.stop}"
                for idx, sl in enumerate(unpadded_global_slice)
            ]
        )

        if proposals_in_block.shape[0]:

            logger.info(
                f"{proposals_in_block.shape[0]} proposals found in {global_pos_name}!"
            )

            for proposal in proposals_in_block:
                local_coord_proposal = proposal[:3] - np.array(
                    global_coord_positions_start[0][1:]
                )

                # ZYX coord order
                local_coord_proposal = local_coord_proposal.astype(np.int32)

                extracted_block = extract_centered_3d_block(
                    big_block=data_block,
                    center=local_coord_proposal,
                    size=(cube_depth, cube_height, cube_width),
                    pad_value=0,
                )

                # Comparing shapes starting pos 1, since we have two channels
                if extracted_block.shape[1:] != (cube_depth, cube_height, cube_width):
                    error = (
                        "Shapes between CellFinder and extracted cube don't match."
                        f"Block {extracted_block.shape} - CellFinder: {(cube_depth, cube_height, cube_width)}"
                    )
                    raise ValueError(error)

                # Changing orientation from DZYX to XYZD for cellfinder
                extracted_block = extracted_block.transpose(-1, -2, -3, -4)
                blocks_to_classify.append(extracted_block)
                picked_proposals.append(proposal)
                curr_blocks += 1
        else:
            logger.info(f"No proposals found in {global_pos_name}!")

        if (
            curr_blocks >= max_blocks
        ):  # and len(blocks_to_classify) == len(picked_proposals)
            blocks_to_classify = np.array(blocks_to_classify, dtype=np.float32)
            picked_proposals = np.array(picked_proposals, dtype=np.uint32)

            if blocks_to_classify.shape[0] != picked_proposals.shape[0]:
                error = (
                    "Shapes between blocks and proposals are not the same:"
                    f"blocks: {blocks_to_classify.shape} - Proposals: {picked_proposals.shape}"
                )
                ValueError(error)

            previous_cell_count = processed_cells
            processed_cells += picked_proposals.shape[0]
            curr_cell_count = processed_cells - previous_cell_count

            predictions_raw = model.predict(blocks_to_classify)
            predictions = predictions_raw.round()
            predictions = predictions.astype("uint16")

            predictions = np.argmax(predictions, axis=1)

            if predictions.shape[0] != blocks_to_classify.shape[0]:
                error = (
                    "Shapes between blocks and predictions are not the same:"
                    f"blocks: {blocks_to_classify.shape} - Proposals: {predictions.shape}"
                )
                ValueError(error)

            cell_likelihood = []
            for idx, proposal in enumerate(picked_proposals):
                cell_type = predictions[idx] + 1

                cell_z, cell_y, cell_x = upsample_position(
                    proposal[:3], downsample_factor=smartspim_config["downsample"]
                )

                cell_likelihood.append(
                    [cell_x, cell_y, cell_z, cell_type, predictions_raw[idx][1]]
                )

            cell_likelihood = np.array(cell_likelihood)

            all_cells_df = pd.DataFrame(
                cell_likelihood, columns=["x", "y", "z", "Class", "Cell Likelihood"]
            )

            all_cells_df.to_csv(
                os.path.join(
                    smartspim_config["metadata_path"],
                    f"classified_block_count_{curr_cell_count}_end_block_{global_pos_name}.csv",
                )
            )
            curr_blocks = 0
            picked_proposals = []
            blocks_to_classify = []
            logger.info(
                f"[PROGRESS] Total of cells at this point: {processed_cells} - Restarted vars - blocks: {len(blocks_to_classify)} proposals: {len(picked_proposals)}"
            )

    logger.info(f"Number of blocks left to process: {curr_blocks}")

    if curr_blocks:
        blocks_to_classify = np.array(blocks_to_classify, dtype=np.float32)
        picked_proposals = np.array(picked_proposals, dtype=np.uint32)

        if blocks_to_classify.shape[0] != picked_proposals.shape[0]:
            error = (
                "Shapes between blocks and proposals are not the same:"
                f"blocks: {blocks_to_classify.shape} - Proposals: {picked_proposals.shape}"
            )
            ValueError(error)

        previous_cell_count = processed_cells
        processed_cells += picked_proposals.shape[0]
        curr_cell_count = processed_cells - previous_cell_count

        predictions_raw = model.predict(blocks_to_classify)
        predictions = predictions_raw.round()
        predictions = predictions.astype("uint16")

        predictions = np.argmax(predictions, axis=1)

        if predictions.shape[0] != blocks_to_classify.shape[0]:
            error = (
                "Shapes between blocks and predictions are not the same:"
                f"blocks: {blocks_to_classify.shape} - Proposals: {predictions.shape}"
            )
            ValueError(error)

        cell_likelihood = []
        for idx, proposal in enumerate(picked_proposals):
            cell_type = predictions[idx] + 1

            cell_z, cell_y, cell_x = upsample_position(
                proposal[:3], downsample_factor=smartspim_config["downsample"]
            )

            cell_likelihood.append(
                [cell_x, cell_y, cell_z, cell_type, predictions_raw[idx][1]]
            )

        cell_likelihood = np.array(cell_likelihood)

        all_cells_df = pd.DataFrame(
            cell_likelihood, columns=["x", "y", "z", "Class", "Cell Likelihood"]
        )

        all_cells_df.to_csv(
            os.path.join(
                smartspim_config["metadata_path"],
                f"classified_block_count_{curr_cell_count}_end_block_{global_pos_name}.csv",
            )
        )
        curr_blocks = 0
        picked_proposals = []
        blocks_to_classify = []

    end_date_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_CELL_SEGMENTATION,
            software_version=__version__,
            start_date_time=start_date_time,
            end_date_time=end_date_time,
            input_location=str(image_path),
            output_location=str(smartspim_config["metadata_path"]),
            outputs={},
            code_url="https://github.com/AllenNeuralDynamics/aind-smartspim-classification",
            code_version=__version__,
            parameters={
                "image_path": str(image_path),
                "background_path": str(background_path),
                "mask_path": str(mask_path),
                "smartspim_cell_config": smartspim_config,
                "target_size_mb": target_size_mb,
                "prediction_chunksize": prediction_chunksize,
                "overlap_prediction_chunksize": overlap_prediction_chunksize,
            },
            notes=f"Classifying channel in path: {image_path}",
        )
    )

    logger.info(
        f"Classification time : {end_date_time - start_date_time} minutes! Total cells: {processed_cells} - Proposals: {cell_proposals.shape}"
    )

    return str(image_path), data_processes


def merge_csv(metadata_path: PathLike, save_path: PathLike, logger: logging.Logger):
    """
    Saves list of all cell locations and likelihoods to CSV

    Returns
    -------
    str
        Path where the merged csv was stored
    """

    # load temporary files and save to a single list
    logger.info(f"Reading CSVs from cells path: {metadata_path}")
    cells = []
    tmp_files = glob(metadata_path + "/classified_block_*.csv")
    logger.info(f"Merging files: {tmp_files}")

    for f in natsorted(tmp_files):
        try:
            cells.append(pd.read_csv(f, index_col=0))
        except:
            pass

    # save list of all cells
    df = pd.concat(cells)
    df = df.reset_index(drop=True)
    output_csv = os.path.join(save_path, "cell_likelihoods.csv")
    df.to_csv(output_csv)
    return output_csv


def cumulative_likelihoods(save_path: PathLike, logger: logging.Logger):
    """
    Takes the cell_likelihoods.csv and creates a cumulative metric
    """

    logger.info(f"Reading cell likelihood CSV from cells path: {save_path}")

    df = pd.read_csv(os.path.join(save_path, "cell_likelihoods.csv"), index_col=0)

    df_cells = df.loc[df["Class"] == 2, :]
    df_non_cells = df.loc[df["Class"] == 1, :]

    likelihood_metrics = {
        "Cell Counts": len(df_cells),
        "Cell Likelihood Mean": df_cells["Cell Likelihood"].mean(),
        "Cell Likelihood STD": df_cells["Cell Likelihood"].std(),
        "Noncell Counts": len(df_non_cells),
        "Noncell Likelihood Mean": df_non_cells["Cell Likelihood"].mean(),
        "Noncell Likelihood STD": df_non_cells["Cell Likelihood"].std(),
    }

    df_out = pd.DataFrame(likelihood_metrics, index=["Metrics"])
    df_out.to_csv(os.path.join(save_path, "cell_likelihood_metrics.csv"))


def generate_neuroglancer_link(
    image_path: str,
    dataset_name: str,
    channel_name: str,
    classified_cells_path: str,
    output: str,
    voxel_sizes: list,
    logger: logging.Logger,
):
    """
    Generates neuroglancer link with the cell location
    for a specific dataset

    Parameters
    -----------
    image_path: str
        Path to the zarr file

    dataset_name: str
        Dataset name where the data will be stored
        in the cloud. Follows SmartSPIM_***_stitched_***

    channel_name: str
        Channel name that was processed

    classified_cells_path: str
        Path to the detected cells

    output: str
        Output path of the neuroglancer
        config and precomputed format

    voxel_sizes: list
        list of um per voxel along each dimension
        ordered [z, y, x]
    """

    logger.info(f"Reading cells from {classified_cells_path}")
    df_cells = pd.read_csv(classified_cells_path)
    df_cells = df_cells.loc[df_cells["Class"] == 2, :]
    df_cells = df_cells[["x", "y", "z"]]

    cells = df_cells.to_dict(orient="records")

    output_precomputed = os.path.join(output, "visualization/classified_precomputed")
    json_name = os.path.join(output, "visualization/neuroglancer_config.json")
    utils.create_folder(output_precomputed)
    print(f"Output cells precomputed: {output_precomputed}")

    logger.info(f"Image path in {image_path}")
    example_data = {
        "dimensions": {
            # check the order
            "z": {"voxel_size": voxel_sizes[0], "unit": "microns"},
            "y": {"voxel_size": voxel_sizes[1], "unit": "microns"},
            "x": {"voxel_size": voxel_sizes[2], "unit": "microns"},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": [
            {
                "source": image_path,
                "type": "image",
                "channel": 0,
                "shader": {"color": "gray", "emitter": "RGB", "vec": "vec3"},
                "shaderControls": {"normalized": {"range": [0, 500]}},  # Optional
            },
            {
                "type": "annotation",
                "source": f"precomputed://{output_precomputed}",
                "tool": "annotatePoint",
                "name": "classified cells",
                "annotations": cells,
            },
        ],
    }
    bucket_path = "aind-open-data"
    neuroglancer_link = NgState(
        input_config=example_data,
        base_url="https://aind-neuroglancer-sauujisjxq-uw.a.run.app",
        mount_service="s3",
        bucket_path=bucket_path,
        output_json=os.path.join(output, "visualization"),
        json_name=json_name,
    )

    json_state = neuroglancer_link.state
    json_state["ng_link"] = (
        f"https://aind-neuroglancer-sauujisjxq-uw.a.run.app#!s3://{bucket_path}/{dataset_name}/image_cell_segmentation/{channel_name}/visualization/neuroglancer_config.json"
    )

    json_state["layers"][0][
        "source"
    ] = f"zarr://s3://{bucket_path}/{dataset_name}/image_tile_fusing/OMEZarr/{channel_name}.zarr"

    json_state["layers"][1][
        "source"
    ] = f"precomputed://s3://{bucket_path}/{dataset_name}/image_cell_segmentation/{channel_name}/visualization/classified_precomputed"

    logger.info(f"Visualization link: {json_state['ng_link']}")
    output_path = os.path.join(output, json_name)

    with open(output_path, "w") as outfile:
        json.dump(json_state, outfile, indent=2)


def main(
    smartspim_config: dict,
    cell_proposals: np.array,
):
    """
    This function detects cells

    Parameters
    -----------

    smartspim_config: dict
        Dictionary with the smartspim configuration
        for that dataset

    cell_proposals: np.array
        Cell proposals from the previous step.

    """

    utils.create_folder(smartspim_config["metadata_path"])

    # Logger pointing everything to the metadata path
    logger = utils.create_logger(output_log_path=smartspim_config["metadata_path"])
    utils.print_system_information(logger)

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    # run cell detection
    image_path, data_processes = cell_classification(
        smartspim_config=smartspim_config, logger=logger, cell_proposals=cell_proposals
    )

    # merge block .xmls and .csvs into single file
    # merge_xml(smartspim_config["metadata_path"], smartspim_config["save_path"], logger)
    classified_cells_path = merge_csv(
        smartspim_config["metadata_path"], smartspim_config["save_path"], logger
    )

    # generate cumulative metrics
    cumulative_likelihoods(smartspim_config["save_path"], logger)

    image_path = os.path.abspath(
        f"{smartspim_config['input_data']}/{smartspim_config['input_channel']}"
    )

    # create neuroglancer link
    generate_neuroglancer_link(
        image_path,
        smartspim_config["name"],
        smartspim_config["channel"],
        classified_cells_path,
        smartspim_config["save_path"],
        smartspim_config["ng_voxel_sizes"],
        logger,
    )

    utils.generate_processing(
        data_processes=data_processes,
        dest_processing=str(smartspim_config["metadata_path"]),
        processor_full_name=__maintainers__[-1],
        pipeline_version=__pipeline_version__,
    )

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            smartspim_config["metadata_path"],
            "smartspim_classification",
        )


if __name__ == "__main__":
    main()
