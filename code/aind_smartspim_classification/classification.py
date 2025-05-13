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

from .__init__ import __maintainers__, __pipeline_version__, __version__
from ._shared.types import PathLike
from .utils import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
import tensorflow as tf
import tensorflow.keras.backend as K

@keras.saving.register_keras_serializable()
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_pred = K.cast(y_pred, tf.float32)
    y_true = K.cast(y_true, tf.float32)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

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
    standardize = False,
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

    standardize: bool
        If you want to normalize and standardize your data before
        passing to the model

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
        f"{smartspim_config['input_channel']}/{smartspim_config['model_config']['parameters']['mask_scale']}"
    )
    downsample = smartspim_config["model_config"]["parameters"]["downsample"]

    print(f" Image Path: {image_path} -- mask path: {mask_path} - scale: {downsample}")

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
                downsample,
                downsample,
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
                multiscale=downsample,
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

    model_config = smartspim_config.get("model_config")

    if model_config is None:
        raise ValueError(f"Please, provide a model configuration: {smartspim_config}")

    cube_width = model_config["parameters"]["cube_width"]
    cube_height = model_config["parameters"]["cube_height"]
    cube_depth = model_config["parameters"]["cube_depth"]
    model_path = model_config["default_model"]

    # Load model defaults to inference mode
    model = keras.models.load_model(model_path)
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

            if standardize:
                print(f"blocks shape: {blocks_to_classify.shape}")
                means = [
                    np.mean(blocks_to_classify[:, :, :, 0]),
                    np.mean(blocks_to_classify[:, :, :, 1])
                ]

                stds = [
                    np.std(blocks_to_classify[:, :, :, 0]),
                    np.std(blocks_to_classify[:, :, :, 1])
                ]

                for i in range(2):
                    blocks_to_classify[:, :, :, i] -= means[i]
                    blocks_to_classify[:, :, :, i] /= (stds[i] + 1e-7)

                print(f"Signal mean: {np.mean(blocks_to_classify[:, :, :, 0])}")
                print(f"Background mean: {np.std(blocks_to_classify[:, :, :, 1])}")

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
                cell_type = predictions[idx]

                cell_z, cell_y, cell_x = upsample_position(
                    proposal[:3], downsample_factor=downsample
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
            cell_type = predictions[idx]

            cell_z, cell_y, cell_x = upsample_position(
                proposal[:3], downsample_factor=downsample
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

    utils.create_folder(f"{save_path}/proposals")

    # save list of all cells
    df = pd.concat(cells)
    df = df.reset_index(drop=True)
    output_csv = os.path.join(save_path, "proposals/cell_likelihoods.csv")

    df.to_csv(output_csv)

    # Saving detected cells
    df_cells = df.copy()
    df_cells = df_cells.loc[df_cells["Class"] == 1, :]
    df_cells = df_cells[["x", "y", "z"]]
    df_cells = df_cells.reset_index(drop=True)
    output_csv = os.path.join(save_path, "detected_cells.csv")
    df_cells.to_csv(output_csv)

    return output_csv, df_cells


def cumulative_likelihoods(save_path: PathLike, logger: logging.Logger):
    """
    Takes the cell_likelihoods.csv and creates a cumulative metric
    """

    logger.info(f"Reading cell likelihood CSV from cells path: {save_path}")

    df = pd.read_csv(
        os.path.join(save_path, "proposals/cell_likelihoods.csv"), index_col=0
    )

    df_cells = df.loc[df["Class"] == 1, :]
    df_non_cells = df.loc[df["Class"] == 0, :]

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
    cells_df: pd.DataFrame,
    ng_configs: dict,
    smartspim_config: dict,
    dynamic_range: list,
    logger: logging.Logger,
    bucket="aind-open-data",
):
    """
    Creates the json state dictionary for the neuroglancer link

    Parameters
    ----------
    cells_df: pd.DataFrame
        the location of all the cells from proposal phase
    ng_configs : dict
        Parameters for creating neuroglancer link defined in run_capsule.py
    smartspim_config : dict
        Dataset specific parameters from processing_manifest
    dynamic_range : list
        The intensity range calculated from the zarr
    logger: logging.Logger
    bucket: str
        Location on AWS where the data lives

    Returns
    -------
    json_state : dict
        fully configured JSON for neuroglancer visualization
    """

    output_precomputed = os.path.join(
        smartspim_config["save_path"], "visualization/detected_precomputed"
    )
    utils.create_folder(output_precomputed)
    print(f"Output cells precomputed: {output_precomputed}")

    utils.generate_precomputed_cells(
        cells_df, precompute_path=output_precomputed, configs=ng_configs
    )

    ng_path = f"s3://{bucket}/{smartspim_config['name']}/image_cell_segmentation/{smartspim_config['channel']}/visualization/neuroglancer_config.json"

    if isinstance(ng_configs["orientation"], dict):
        crossSectionOrientation = utils.volume_orientation(ng_configs["orientation"])
    else:
        crossSectionOrientation = [np.cos(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]

    json_state = {
        "ng_link": f"{ng_configs['base_url']}{ng_path}",
        "title": smartspim_config["channel"],
        "dimensions": ng_configs["dimensions"],
        "crossSectionOrientation": crossSectionOrientation,
        "crossSectionScale": ng_configs["crossSectionScale"],
        "projectionScale": ng_configs["projectionScale"],
        "layers": [
            {
                "source": f"zarr://s3://{bucket}/{smartspim_config['name']}/image_tile_fusing/OMEZarr/{smartspim_config['channel']}.zarr",
                "type": "image",
                "tab": "rendering",
                "shader": '#uicontrol vec3 color color(default="#ffffff")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}',
                "shaderControls": {
                    "normalized": {
                        "range": [0, dynamic_range[0]],
                        "window": [0, dynamic_range[1]],
                    },
                },
                "name": f"Channel: {smartspim_config['channel']}",
            },
            {
                "source": f"precomputed://s3://{bucket}/{smartspim_config['name']}/image_cell_segmentation/{smartspim_config['channel']}/visualization/detected_precomputed",
                "type": "annotation",
                "tool": "annotatePoint",
                "tab": "annotations",
                "crossSectionAnnotationSpacing": 1.0,
                "name": "Classified Cells",
            },
        ],
        "gpuMemoryLimit": ng_configs["gpuMemoryLimit"],
        "selectedLayer": {
            "visible": True,
            "layer": f"Channel: {smartspim_config['channel']}",
        },
        "layout": "4panel",
    }

    logger.info(f"Visualization link: {json_state['ng_link']}")
    output_path = os.path.join(
        smartspim_config["save_path"], "visualization/neuroglancer_config.json"
    )

    with open(output_path, "w") as outfile:
        json.dump(json_state, outfile, indent=2)


def main(
    smartspim_config: dict,
    neuroglancer_config: dict,
    cell_proposals: np.array,
    ng_voxel_sizes: List[float] = [2.0, 1.8, 1.8],
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

    ng_voxel_sizes: List[float]
        Default voxel sizes for SmartSPIM.
        Default: [2.0, 1.8, 1.8]
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
    classified_cells_path, cells_df = merge_csv(
        smartspim_config["metadata_path"], smartspim_config["save_path"], logger
    )

    # generate cumulative metrics
    cumulative_likelihoods(smartspim_config["save_path"], logger)

    image_path = os.path.abspath(
        f"{smartspim_config['input_data']}/{smartspim_config['input_channel']}"
    )

    dynamic_range = utils.calculate_dynamic_range(image_path, 99, 3)

    generate_neuroglancer_link(
        cells_df, neuroglancer_config, smartspim_config, dynamic_range, logger
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
