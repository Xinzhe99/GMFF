from typing import Sequence, Dict, Union, Mapping, Any, Optional
import time
import io
import random
import importlib
import os
import numpy as np
from PIL import Image
import torch.utils.data as data
try:
    from utils import load_file_list, center_crop_arr, random_crop_arr
except ImportError:
    from .utils import load_file_list, center_crop_arr, random_crop_arr

def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    # Check if 'target' key exists in config
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    # Instantiate object from config
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string: str, reload: bool = False) -> Any:
    # Split module and class name
    module, cls = string.rsplit(".", 1)
    # Reload module if needed
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # Return class from module
    return getattr(importlib.import_module(module, package=None), cls)


class GMFF_Dataset(data.Dataset):

    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        # blur_kernel_size: int,  # Unused parameter
        # kernel_list: Sequence[str],  # Unused parameter
        # kernel_prob: Sequence[float],  # Unused parameter
        # blur_sigma: Sequence[float],  # Unused parameter
        # downsample_range: Sequence[float],  # Unused parameter
        # noise_range: Sequence[float],  # Unused parameter
        # jpeg_range: Sequence[int],  # Unused parameter
    ) -> None:
        super(GMFF_Dataset, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]

    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        # Load image bytes with retry mechanism
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Crop image if needed
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return np.array(image, dtype=np.uint8)

    def load_aif_missing_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        # Replace "AiF" with "AiF_missing" in path and change extension to png
        aif_missing_path = image_path.replace("AiF", "AiF_missing")
        # Load image bytes with retry mechanism
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(aif_missing_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Crop image if needed
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return np.array(image, dtype=np.uint8)

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # Load ground truth images
        img_aif = None
        img_aif_missing = None
        prompt = ""  # Initialize prompt variable
        while img_aif is None or img_aif_missing is None:
            # Load meta file
            image_file = self.image_files[index]
            aif_path = image_file["image_path"]
            prompt = image_file.get("prompt", "")  # Provide default value
            img_aif = self.load_gt_image(aif_path)
            img_aif_missing = self.load_aif_missing_image(aif_path)
            if img_aif is None or img_aif_missing is None:
                print(f"failed to load {aif_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_aif = (img_aif[..., ::-1] / 255.0).astype(np.float32)
        img_aif_missing = (img_aif_missing[..., ::-1] / 255.0).astype(np.float32)
        if np.random.uniform() < 0.5:
            prompt = ""

        # BGR to RGB, [-1, 1]
        aif = (img_aif[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        aif_missing = (img_aif_missing[..., ::-1]).astype(np.float32)
        # prompt is not needed for GMFF training.
        return {"aif": aif, "aif_missing": aif_missing, "prompt": prompt}
    

    def __len__(self) -> int:
        return len(self.image_files)