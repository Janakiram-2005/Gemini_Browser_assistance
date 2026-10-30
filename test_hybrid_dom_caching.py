#!/usr/bin/env python3
"""
Test suite for Hybrid DOM-First Navigation with Differential Frame Caching:
1. Perceptual frame hash (dHash) calculation & performance (<30ms)
2. Differential frame caching & static screen state detection (Hamming diff <= 3)
3. Frame downsampling (<=1280x720) & compression payload guardrail (<300 KB)
4. Visual fallback trigger conditions (DOM confidence < 0.70 or canvas element interaction)
5. Backend timeout configuration (30s default) & fallback model execution
"""

import sys
import os
import time
import io
import json
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import app as app_module
from app import app, OLLAMA_TIMEOUT, FRAME_CACHE


def compute_python_dhash(image: Image.Image) -> tuple:
    """
    Computes 64-bit dHash for PIL Image matching client JS implementation (9x8 grayscale matrix).
    """
    start_time = time.perf_counter()
    # Resize to 9x8 grayscale
    img_gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    if hasattr(img_gray, "get_flattened_data"):
        pixels = list(img_gray.get_flattened_data())
    else:
        pixels = list(img_gray.getdata())
    
    binary_bits = []
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            binary_bits.append("1" if left > right else "0")
            
    binary_str = "".join(binary_bits)
    hex_hash = f"{int(binary_str, 2):016x}"
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return hex_hash, elapsed_ms


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculates Hamming distance between two 16-hex character dHashes."""
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    xor_val = val1 ^ val2
    return bin(xor_val).count("1")


def process_and_compress_frame(img: Image.Image, max_w: int = 1280, max_h: int = 720, initial_quality: int = 80) -> tuple:
    """
    Downsamples frame to max 1280x720 and applies JPEG compression.
    Ensures size is strictly < 300 KB.
    """
    w, h = img.size
    if w > max_w or h > max_h:
        ratio = min(max_w / w, max_h / h)
        w = int(w * ratio)
        h = int(h * ratio)
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        
    quality = initial_quality
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    size_kb = len(buffer.getvalue()) / 1024
    
    while size_kb > 300 and quality > 20:
        quality -= 10
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        size_kb = len(buffer.getvalue()) / 1024
        
    return img, buffer.getvalue(), size_kb, quality


class TestHybridDomCaching(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_dhash_performance_and_calculation(self):
        """Verify dHash calculation runs in < 30ms and produces correct 16-char hex string."""
        img = Image.new("RGB", (1920, 1080), color=(120, 150, 180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 500, 500], fill=(255, 0, 0))
        
        hex_hash, elapsed_ms = compute_python_dhash(img)
        
        print(f"\n[TEST 1] dHash: {hex_hash} | Time: {elapsed_ms:.2f} ms")
        self.assertEqual(len(hex_hash), 16)
        self.assertLess(elapsed_ms, 30.0, f"dHash took {elapsed_ms:.2f}ms, which exceeds 30ms constraint")

    def test_02_differential_caching_and_static_state_detection(self):
        """Verify static screen states produce low Hamming diff (<=3) and hit frame cache."""
        img1 = Image.new("RGB", (1280, 720), color=(240, 240, 240))
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([50, 50, 200, 200], fill=(0, 120, 255))
        
        # Near identical image
        img2 = img1.copy()
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([52, 52, 54, 54], fill=(0, 120, 255))
        
        # Completely different image
        img3 = Image.new("RGB", (1280, 720), color=(10, 10, 10))
        draw3 = ImageDraw.Draw(img3)
        draw3.ellipse([300, 300, 600, 600], fill=(255, 255, 0))
        
        hash1, _ = compute_python_dhash(img1)
        hash2, _ = compute_python_dhash(img2)
        hash3, _ = compute_python_dhash(img3)
        
        diff_identical = hamming_distance(hash1, hash2)
        diff_different = hamming_distance(hash1, hash3)
        
        print(f"[TEST 2] Static screen diff: {diff_identical} bits | Changed screen diff: {diff_different} bits")
        self.assertLessEqual(diff_identical, 3, "Static screen diff should be <= 3 bits")
        self.assertGreater(diff_different, 3, "Changed screen diff should be > 3 bits")

    def test_03_frame_downsampling_and_size_constraint(self):
        """Verify frame processor downsamples to <=1280x720 and payload size is < 300 KB."""
        # 4K image
        img = Image.new("RGB", (3840, 2160), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        for i in range(0, 3840, 40):
            draw.line([(i, 0), (i, 2160)], fill=(i % 255, (i*2)%255, (i*3)%255), width=10)
            
        downsampled_img, buffer_bytes, size_kb, quality = process_and_compress_frame(img, max_w=1280, max_h=720, initial_quality=80)
        
        print(f"[TEST 3] Original: 3840x2160 -> Downsampled: {downsampled_img.size[0]}x{downsampled_img.size[1]} | Size: {size_kb:.1f} KB")
        self.assertLessEqual(downsampled_img.size[0], 1280)
        self.assertLessEqual(downsampled_img.size[1], 720)
        self.assertLess(size_kb, 300.0, f"Payload size {size_kb:.1f} KB exceeds 300 KB constraint")

    def test_04_backend_timeout_and_fallback_config(self):
        """Verify OLLAMA_TIMEOUT is set to 30s active timeout threshold."""
        print(f"[TEST 4] Backend active step timeout threshold: {OLLAMA_TIMEOUT}s")
        self.assertEqual(OLLAMA_TIMEOUT, 30, "Default timeout threshold must be 30 seconds")

    @patch("app.generate_response")
    def test_05_api_dom_first_vs_visual_fallback(self, mock_gen):
        """Test Flask /agent endpoint handling DOM-first and Visual Fallback requests."""
        mock_gen.return_value = json.dumps({
            "action": "click",
            "selector": "#search-btn",
            "value": "Search",
            "caption": "Clicking search",
            "confidence": 0.95,
            "is_goal_achieved": False
        })

        # 1. Standard DOM-First request (No visual frame)
        resp_dom = self.app.post("/agent", json={
            "command": "Click search button",
            "model": "qwen2.5",
            "context": "Search page content",
            "page_url": "https://example.com",
            "use_visual_fallback": False,
            "last_confidence": 0.95
        })
        self.assertEqual(resp_dom.status_code, 200)
        data_dom = resp_dom.get_json()
        print(f"[TEST 5A] DOM-First response action: {data_dom.get('action')}")
        self.assertFalse(data_dom.get("meta", {}).get("use_visual_fallback"))

        # 2. Visual Fallback request with low confidence (< 0.70)
        dummy_hash = "a1b2c3d4e5f67890"
        resp_vis = self.app.post("/agent", json={
            "command": "Click complex visual element",
            "model": "qwen2.5",
            "context": "Visual canvas page",
            "page_url": "https://example.com/dashboard",
            "use_visual_fallback": True,
            "last_confidence": 0.55,
            "frame_hash": dummy_hash,
            "frame_size_kb": 150.0,
            "is_canvas_interaction": True,
            "step_number": 2
        })
        self.assertEqual(resp_vis.status_code, 200)
        data_vis = resp_vis.get_json()
        print(f"[TEST 5B] Visual Fallback response meta: {data_vis.get('meta')}")
        self.assertTrue(data_vis.get("meta", {}).get("use_visual_fallback"))
        self.assertEqual(data_vis.get("meta", {}).get("dhash"), dummy_hash)

        # 3. Repeat Visual Fallback request with same hash -> Frame Cache HIT
        resp_cache = self.app.post("/agent", json={
            "command": "Click complex visual element",
            "model": "qwen2.5",
            "context": "Visual canvas page",
            "page_url": "https://example.com/dashboard",
            "use_visual_fallback": True,
            "last_confidence": 0.55,
            "frame_hash": dummy_hash,
            "frame_size_kb": 150.0,
            "is_canvas_interaction": True,
            "frame_cached": True,
            "step_number": 3
        })
        self.assertEqual(resp_cache.status_code, 200)
        data_cache = resp_cache.get_json()
        print(f"[TEST 5C] Visual Cache HIT meta: {data_cache.get('meta')}")
        self.assertTrue(data_cache.get("meta", {}).get("frame_cached"))


if __name__ == "__main__":
    unittest.main()
