#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from report_guided_annotation import extract_lesion_candidates

from picai_baseline.nnunet.softmax_export import \
    save_softmax_nifti_from_softmax
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample, crop_or_pad


class MissingSequenceError(Exception):
    """Exception raised when a sequence is missing."""

    def __init__(self, name, folder):
        message = f"Could not find scan for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


class MultipleScansSameSequencesError(Exception):
    """Exception raised when multiple scans of the same sequences are provided."""

    def __init__(self, name, folder):
        message = f"Found multiple scans for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


def convert_to_original_extent(pred: np.ndarray, pkl_path: Union[Path, str], dst_path: Union[Path, str]):
    # convert to nnUNet's internal softmax format
    pred = np.array([1-pred, pred])

    # read physical properties of current case
    with open(pkl_path, "rb") as fp:
        properties = pickle.load(fp)

    # let nnUNet resample to original physical space
    save_softmax_nifti_from_softmax(
        segmentation_softmax=pred,
        out_fname=str(dst_path),
        properties_dict=properties,
    )


def extract_lesion_candidates_cropped(pred: np.ndarray, threshold: Union[str, float]):
    size = pred.shape
    pred = crop_or_pad(pred, (20, 384, 384))
    pred = crop_or_pad(pred, size)
    return extract_lesion_candidates(pred, threshold=threshold)[0]


class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline nnU-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # input / output paths for algorithm
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri",
            "/input/images/transverse-adc-prostate-mri",
            "/input/images/transverse-hbv-prostate-mri",
        ]
        self.scan_paths = []
        self.cspca_detection_map_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.case_confidence_path = Path("/output/cspca-case-level-likelihood.json")

        # input / output paths for nnUNet
        self.nnunet_inp_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_out_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_results = Path("/opt/algorithm/results")

        # ensure required folders exist
        self.nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        self.cspca_detection_map_path.parent.mkdir(exist_ok=True, parents=True)

        # input validation for multiple inputs
        scan_glob_format = "*.mha"
        for folder in self.image_input_dirs:
            file_paths = list(Path(folder).glob(scan_glob_format))
            if len(file_paths) == 0:
                raise MissingSequenceError(name=folder.split("/")[-1], folder=folder)
            elif len(file_paths) >= 2:
                raise MultipleScansSameSequencesError(name=folder.split("/")[-1], folder=folder)
            else:
                # append scan path to algorithm input paths
                self.scan_paths += [file_paths[0]]

    def preprocess_input(self):
        """Preprocess input images to nnUNet Raw Data Archive format"""
        # set up Sample
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.scan_paths
            ],
        )

        # perform preprocessing
        sample.preprocess()

        # write preprocessed scans to nnUNet input directory
        for i, scan in enumerate(sample.scans):
            path = self.nnunet_inp_dir / f"scan_{i:04d}.nii.gz"
            atomic_image_write(scan, path)

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load bpMRI scans and generate detection map for clinically significant prostate cancer
        """
        # perform preprocessing
        self.preprocess_input()

        # perform inference using nnUNet
        pred_ensemble = None
        ensemble_count = 0
        for trainer in [
            "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
        ]:
            # predict sample
            self.predict(
                task="Task2201_picai_baseline",
                trainer=trainer,
                checkpoint="model_best",
            )

            # read softmax prediction
            pred_path = str(self.nnunet_out_dir / "scan.npz")
            pred = np.array(np.load(pred_path)['softmax'][1]).astype('float32')
            os.remove(pred_path)
            if pred_ensemble is None:
                pred_ensemble = pred
            else:
                pred_ensemble += pred
            ensemble_count += 1

        # average the accumulated confidence scores
        pred_ensemble /= ensemble_count

        # the prediction is currently at the size and location of the nnU-Net preprocessed
        # scan, so we need to convert it to the original extent before we continue
        convert_to_original_extent(
            pred=pred_ensemble,
            pkl_path=self.nnunet_out_dir / "scan.pkl",
            dst_path=self.nnunet_out_dir / "softmax.nii.gz",
        )

        # now each voxel in softmax.nii.gz corresponds to the same voxel in the original (T2-weighted) scan
        pred_ensemble = sitk.ReadImage(str(self.nnunet_out_dir / "softmax.nii.gz"))

        # extract lesion candidates from softmax prediction
        # note: we set predictions outside the central 81 x 192 x 192 mm to zero, as this is far outside the prostate
        detection_map = extract_lesion_candidates_cropped(
            pred=sitk.GetArrayFromImage(pred_ensemble),
            threshold="dynamic"
        )

        # convert detection map to a SimpleITK image and infuse the physical metadata of original T2-weighted scan
        reference_scan_original_path = str(self.scan_paths[0])
        reference_scan_original = sitk.ReadImage(reference_scan_original_path)
        detection_map: sitk.Image = sitk.GetImageFromArray(detection_map)
        detection_map.CopyInformation(reference_scan_original)

        # save prediction to output folder
        atomic_image_write(detection_map, str(self.cspca_detection_map_path))

        # save case-level likelihood
        with open(self.case_confidence_path, 'w') as fp:
            json.dump(float(np.max(sitk.GetArrayFromImage(detection_map))), fp)

    def predict(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
                checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=True,
                disable_augmentation=False, disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_results)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(self.nnunet_inp_dir),
            '-o', str(self.nnunet_out_dir),
            '-m', network,
            '-tr', trainer,
            '--num_threads_preprocessing', '2',
            '--num_threads_nifti_save', '1'
        ]

        if folds:
            cmd.append('-f')
            cmd.extend(folds.split(','))

        if checkpoint:
            cmd.append('-chk')
            cmd.append(checkpoint)

        if store_probability_maps:
            cmd.append('--save_npz')

        if disable_augmentation:
            cmd.append('--disable_tta')

        if disable_patch_overlap:
            cmd.extend(['--step_size', '1'])

        subprocess.check_call(cmd)


if __name__ == "__main__":
    csPCaAlgorithm().process()
