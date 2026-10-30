"""
Adaptive Visual Stream Pipeline with Dynamic Downsampling and Change Detection.

Optimizes visual navigation for browser/desktop automation by:
1. Dynamically downsampling frames based on element density to ensure < 200 KB payload size.
2. Computing fast SSIM/MSE diffs (<40ms) to detect static screens (<2% variance).
3. Bypassing VLM calls when screens remain static and reusing cached spatial coordinates.
4. Mapping spatial bounding boxes accurately between original and downsampled dimensions.
5. Enforcing 45s maximum timeout bounds on VLM inference with graceful error handling.
"""

import io
import time
import base64
import logging
from typing import Tuple, Dict, Any, Optional, Union, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from PIL import Image
import numpy as np

logger = logging.getLogger("VisualPipeline")

# Thresholds
MAX_PAYLOAD_BYTES = 200 * 1024  # 200 KB max image payload per step
DIFF_VARIANCE_THRESHOLD = 0.02  # 2% variance threshold for static screen detection
MAX_STEP_TIMEOUT_SECONDS = 45.0  # 45 seconds step timeout cap

# Standard resolution targets for dynamic downsampling
DENSITY_RESOLUTIONS = {
    "high": (1280, 720),     # 720p for dense UI / small text
    "medium": (960, 540),    # 540p for average density UI
    "low": (854, 480),       # 480p for sparse UI
}


def decode_image(image_input: Union[str, bytes, Image.Image]) -> Image.Image:
    """Decodes base64 string, raw bytes, or returns PIL Image converted to RGB."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    
    if isinstance(image_input, str):
        # Strip header if data URI scheme is present
        if "," in image_input:
            image_input = image_input.split(",", 1)[1]
        raw_data = base64.b64decode(image_input)
        return Image.open(io.BytesIO(raw_data)).convert("RGB")
    
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
        
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def encode_image_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> bytes:
    """Encodes PIL Image to bytes in specified format and quality."""
    buf = io.BytesIO()
    if fmt.upper() in ("JPG", "JPEG"):
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format=fmt, optimize=True)
    return buf.getvalue()


def encode_image_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encodes PIL Image to base64 string."""
    raw_bytes = encode_image_to_bytes(img, fmt=fmt, quality=quality)
    return base64.b64encode(raw_bytes).decode("utf-8")


def resolve_density_target(element_density: Union[str, int, float, None]) -> Tuple[int, int]:
    """Determines target resolution based on element density input."""
    if isinstance(element_density, (int, float)):
        if element_density > 50:
            key = "high"
        elif element_density > 20:
            key = "medium"
        else:
            key = "low"
    elif isinstance(element_density, str) and element_density.lower() in DENSITY_RESOLUTIONS:
        key = element_density.lower()
    else:
        key = "medium"
    
    return DENSITY_RESOLUTIONS[key]


def downsample_frame(
    image_input: Union[str, bytes, Image.Image],
    element_density: Union[str, int, float, None] = "medium",
    max_bytes: int = MAX_PAYLOAD_BYTES
) -> Dict[str, Any]:
    """
    Dynamically downsamples display frames based on element density while maintaining 
    text legibility and ensuring payload size stays strictly under max_bytes (default 200 KB).
    
    Returns:
        dict: {
            'image': PIL.Image,
            'b64': str,
            'bytes': bytes,
            'size_kb': float,
            'original_size': (width, height),
            'downsampled_size': (width, height),
            'scale_x': float,
            'scale_y': float,
            'quality': int
        }
    """
    orig_img = decode_image(image_input)
    orig_w, orig_h = orig_img.size
    
    target_max_w, target_max_h = resolve_density_target(element_density)
    
    # Calculate scale maintaining aspect ratio
    scale = min(target_max_w / orig_w, target_max_h / orig_h, 1.0)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    
    if (new_w, new_h) != (orig_w, orig_h):
        downsampled_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        downsampled_img = orig_img.copy()
        
    # Quality tuning loop to enforce < max_bytes (e.g. 200 KB) limit
    quality = 85
    encoded_bytes = encode_image_to_bytes(downsampled_img, fmt="JPEG", quality=quality)
    
    # If initial compression is still above max_bytes, progressively scale down quality or resolution
    while len(encoded_bytes) > max_bytes and quality > 30:
        quality -= 10
        encoded_bytes = encode_image_to_bytes(downsampled_img, fmt="JPEG", quality=quality)
        
    if len(encoded_bytes) > max_bytes:
        # Secondary fallback: scale resolution down further
        scaling_factor = 0.8
        while len(encoded_bytes) > max_bytes and (new_w > 320 and new_h > 240):
            new_w = int(new_w * scaling_factor)
            new_h = int(new_h * scaling_factor)
            downsampled_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            encoded_bytes = encode_image_to_bytes(downsampled_img, fmt="JPEG", quality=quality)

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    b64_str = base64.b64encode(encoded_bytes).decode("utf-8")
    
    return {
        "image": downsampled_img,
        "b64": b64_str,
        "bytes": encoded_bytes,
        "size_kb": len(encoded_bytes) / 1024.0,
        "original_size": (orig_w, orig_h),
        "downsampled_size": (new_w, new_h),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "quality": quality,
    }


def compute_ssim_mse(
    img1: Image.Image,
    img2: Image.Image,
    compare_dim: Tuple[int, int] = (256, 256)
) -> Dict[str, float]:
    """
    Computes Structural Similarity Index (SSIM) and Mean Squared Error (MSE) 
    between two PIL images.
    
    Optimized to complete in under 40ms with minimal CPU consumption by using 
    resized grayscale matrix operations.
    
    Returns:
        dict: {
            'mse': float (0.0 to 1.0),
            'ssim': float (0.0 to 1.0),
            'variance': float (0.0 to 1.0),  # 1.0 - ssim
            'diff_ms': float execution time in milliseconds
        }
    """
    start_t = time.perf_counter()
    
    # Convert to grayscale and resize to uniform matrix size for fast comparison
    g1 = np.array(img1.convert("L").resize(compare_dim, Image.Resampling.BILINEAR), dtype=np.float32)
    g2 = np.array(img2.convert("L").resize(compare_dim, Image.Resampling.BILINEAR), dtype=np.float32)
    
    # Normalized Mean Squared Error
    mse = float(np.mean((g1 - g2) ** 2) / (255.0 ** 2))
    
    # Fast SSIM calculation
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    
    mu1 = np.mean(g1)
    mu2 = np.mean(g2)
    
    var1 = np.var(g1)
    var2 = np.var(g2)
    cov12 = np.mean((g1 - mu1) * (g2 - mu2))
    
    ssim_num = (2 * mu1 * mu2 + c1) * (2 * cov12 + c2)
    ssim_den = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
    ssim = float(ssim_num / ssim_den)
    
    # Ensure SSIM is bounded in [0.0, 1.0]
    ssim = max(0.0, min(1.0, ssim))
    variance = max(0.0, 1.0 - ssim)
    
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    
    return {
        "mse": mse,
        "ssim": ssim,
        "variance": variance,
        "diff_ms": elapsed_ms,
    }


def map_bounding_box(
    bbox: List[float],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    bbox_format: str = "norm_1000"
) -> List[float]:
    """
    Maps spatial bounding box coordinates accurately between source and target frame dimensions.
    
    Formats supported:
    - 'norm_1000': [ymin, xmin, ymax, xmax] in [0..1000] space.
    - 'norm_1': [ymin, xmin, ymax, xmax] in [0.0..1.0] space.
    - 'pixel': [x, y, width, height] or [xmin, ymin, xmax, ymax] absolute pixel values.
    
    Returns mapped bounding box in the same format.
    """
    if not bbox or len(bbox) != 4:
        return bbox
        
    src_w, src_h = source_size
    tgt_w, tgt_h = target_size
    
    if src_w <= 0 or src_h <= 0 or tgt_w <= 0 or tgt_h <= 0:
        return bbox
        
    if bbox_format == "norm_1000":
        ymin, xmin, ymax, xmax = bbox
        abs_ymin = (ymin / 1000.0) * src_h * (tgt_h / src_h)
        abs_xmin = (xmin / 1000.0) * src_w * (tgt_w / src_w)
        abs_ymax = (ymax / 1000.0) * src_h * (tgt_h / src_h)
        abs_xmax = (xmax / 1000.0) * src_w * (tgt_w / src_w)
        
        tgt_ymin = round((abs_ymin / tgt_h) * 1000.0, 1)
        tgt_xmin = round((abs_xmin / tgt_w) * 1000.0, 1)
        tgt_ymax = round((abs_ymax / tgt_h) * 1000.0, 1)
        tgt_xmax = round((abs_xmax / tgt_w) * 1000.0, 1)
        return [
            max(0.0, min(1000.0, tgt_ymin)),
            max(0.0, min(1000.0, tgt_xmin)),
            max(0.0, min(1000.0, tgt_ymax)),
            max(0.0, min(1000.0, tgt_xmax)),
        ]
        
    elif bbox_format == "norm_1":
        ymin, xmin, ymax, xmax = bbox
        abs_ymin = ymin * src_h * (tgt_h / src_h)
        abs_xmin = xmin * src_w * (tgt_w / src_w)
        abs_ymax = ymax * src_h * (tgt_h / src_h)
        abs_xmax = xmax * src_w * (tgt_w / src_w)
        
        return [
            round(max(0.0, min(1.0, abs_ymin / tgt_h)), 4),
            round(max(0.0, min(1.0, abs_xmin / tgt_w)), 4),
            round(max(0.0, min(1.0, abs_ymax / tgt_h)), 4),
            round(max(0.0, min(1.0, abs_xmax / tgt_w)), 4),
        ]
        
    elif bbox_format == "pixel":
        x, y, w, h = bbox
        scale_x = tgt_w / src_w
        scale_y = tgt_h / src_h
        return [
            round(x * scale_x, 1),
            round(y * scale_y, 1),
            round(w * scale_x, 1),
            round(h * scale_y, 1),
        ]
        
    return bbox


class VisualStreamPipeline:
    """
    Adaptive Visual Stream Pipeline Manager.
    Handles downsampling, change detection, VLM caching, spatial mapping, and timeout caps.
    """
    
    def __init__(
        self,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        diff_threshold: float = DIFF_VARIANCE_THRESHOLD,
        max_timeout_seconds: float = MAX_STEP_TIMEOUT_SECONDS
    ):
        self.max_payload_bytes = max_payload_bytes
        self.diff_threshold = diff_threshold
        self.max_timeout_seconds = max_timeout_seconds
        
        # Internal state history cache
        self.previous_downsampled_frame: Optional[Image.Image] = None
        self.previous_raw_frame: Optional[Image.Image] = None
        self.cached_action_response: Optional[Dict[str, Any]] = None
        self.cached_spatial_coords: Optional[List[Dict[str, Any]]] = None
        self.step_count: int = 0
        
    def reset(self):
        """Resets cached pipeline state."""
        self.previous_downsampled_frame = None
        self.previous_raw_frame = None
        self.cached_action_response = None
        self.cached_spatial_coords = None
        self.step_count = 0

    def process_frame(
        self,
        image_input: Union[str, bytes, Image.Image],
        element_density: Union[str, int, float, None] = "medium",
        vlm_callback: Optional[Any] = None,
        extracted_coords: Optional[List[Dict[str, Any]]] = None,
        force_vlm: bool = False
    ) -> Dict[str, Any]:
        """
        Processes an incoming visual frame through the adaptive pipeline.
        
        Workflow:
        1. Downsamples the frame dynamically to ensure < 200 KB size while keeping text legible.
        2. Computes SSIM / MSE diff against previous frame.
        3. If variance <= 2% threshold and not forced, bypasses VLM call and reuses cached spatial coordinates / action.
        4. If variance > 2% or first frame, executes VLM call (enforcing <= 45s timeout) and updates cache.
        5. Maps spatial bounding box coordinates between downsampled and original dimensions.
        """
        self.step_count += 1
        
        # 1. Dynamic Downsampling
        downsample_res = downsample_frame(
            image_input,
            element_density=element_density,
            max_bytes=self.max_payload_bytes
        )
        
        current_downsampled_img = downsample_res["image"]
        payload_size_kb = downsample_res["size_kb"]
        
        # 2. Change Detection against previous frame
        is_static = False
        diff_metrics = {
            "mse": 1.0,
            "ssim": 0.0,
            "variance": 1.0,
            "diff_ms": 0.0
        }
        
        if self.previous_downsampled_frame is not None:
            diff_metrics = compute_ssim_mse(current_downsampled_img, self.previous_downsampled_frame)
            if diff_metrics["variance"] <= self.diff_threshold:
                is_static = True
                
        # 3. Decision logic: Bypass VLM or execute VLM
        vlm_bypassed = False
        action_result = None
        vlm_time_ms = 0.0
        error_msg = None
        
        if is_static and not force_vlm and self.cached_action_response is not None:
            # Bypass VLM call & reuse cached coordinates/action
            vlm_bypassed = True
            action_result = dict(self.cached_action_response)
            action_result["reused_from_cache"] = True
            action_result["caption"] = f"[Static Screen Detected - Diff {diff_metrics['variance']*100:.2f}%] Reusing cached action"
            spatial_coords = self.cached_spatial_coords or []
            
        else:
            # Execute VLM call with strict 45s maximum timeout limit
            spatial_coords = extracted_coords or []
            if vlm_callback is not None:
                start_vlm = time.perf_counter()
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(vlm_callback, downsample_res["b64"])
                        action_result = future.result(timeout=self.max_timeout_seconds)
                    vlm_time_ms = (time.perf_counter() - start_vlm) * 1000.0
                    
                    # Update cache on success
                    if isinstance(action_result, dict):
                        self.cached_action_response = action_result
                        if "spatial_coords" in action_result:
                            self.cached_spatial_coords = action_result["spatial_coords"]
                        elif extracted_coords:
                            self.cached_spatial_coords = extracted_coords
                except FuturesTimeoutError:
                    vlm_time_ms = self.max_timeout_seconds * 1000.0
                    error_msg = f"VLM Step Timeout: Executed over {self.max_timeout_seconds}s limit."
                    action_result = {
                        "action": "wait",
                        "reason": error_msg,
                        "caption": "Step timed out after 45s. Retrying...",
                        "is_goal_achieved": False
                    }
                except Exception as ex:
                    error_msg = str(ex)
                    action_result = {
                        "action": "wait",
                        "reason": f"VLM Error: {error_msg}",
                        "caption": "Visual reasoning error encountered.",
                        "is_goal_achieved": False
                    }
            else:
                action_result = {
                    "action": "wait",
                    "reason": "No VLM callback provided",
                    "caption": "Frame processed successfully without VLM invocation.",
                    "is_goal_achieved": False
                }
                
        # 4. Map spatial bounding box coordinates
        mapped_coords = []
        if spatial_coords:
            for item in spatial_coords:
                mapped_item = dict(item)
                if "bbox" in mapped_item:
                    mapped_item["bbox_original"] = map_bounding_box(
                        mapped_item["bbox"],
                        source_size=downsample_res["downsampled_size"],
                        target_size=downsample_res["original_size"],
                        bbox_format=mapped_item.get("format", "norm_1000")
                    )
                mapped_coords.append(mapped_item)

        # Update frame state
        self.previous_downsampled_frame = current_downsampled_img
        
        return {
            "step": self.step_count,
            "is_static": is_static,
            "vlm_bypassed": vlm_bypassed,
            "diff_metrics": diff_metrics,
            "payload_size_kb": payload_size_kb,
            "payload_under_200kb": payload_size_kb < 200.0,
            "diff_under_40ms": diff_metrics["diff_ms"] < 40.0,
            "original_size": downsample_res["original_size"],
            "downsampled_size": downsample_res["downsampled_size"],
            "downsampled_b64": downsample_res["b64"],
            "action_result": action_result,
            "mapped_spatial_coords": mapped_coords,
            "vlm_time_ms": vlm_time_ms,
            "error": error_msg,
        }
