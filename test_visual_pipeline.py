import unittest
import time
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from visual_pipeline import (
    VisualStreamPipeline,
    downsample_frame,
    compute_ssim_mse,
    map_bounding_box,
    MAX_PAYLOAD_BYTES,
    DIFF_VARIANCE_THRESHOLD,
    MAX_STEP_TIMEOUT_SECONDS
)

def create_synthetic_frame(text="Dashboard Overview", bg_color=(240, 240, 240), box_coords=None):
    """Generates a synthetic 1080p desktop / web application image with text and UI elements."""
    img = Image.new("RGB", (1920, 1080), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw header bar
    draw.rectangle([0, 0, 1920, 80], fill=(40, 60, 90))
    draw.text((30, 25), text, fill=(255, 255, 255))
    
    # Draw body card
    draw.rectangle([100, 150, 1800, 950], fill=(255, 255, 255), outline=(200, 200, 200), width=2)
    draw.text((130, 180), "Main Viewport Application Canvas", fill=(30, 30, 30))
    
    # Draw button if coords provided
    if box_coords:
        draw.rectangle(box_coords, fill=(50, 120, 220))
        draw.text((box_coords[0]+10, box_coords[1]+10), "Click Me", fill=(255, 255, 255))
        
    return img


class TestVisualStreamPipeline(unittest.TestCase):

    def test_downsampling_payload_size_under_200kb(self):
        """Requirement 1 & Acceptance Criteria: Image payload size reduced to under 200 KB per step."""
        img_1080p = create_synthetic_frame("Test Payload Size")
        
        # Downsample across high, medium, low density
        for density in ["high", "medium", "low"]:
            result = downsample_frame(img_1080p, element_density=density, max_bytes=MAX_PAYLOAD_BYTES)
            size_kb = result["size_kb"]
            self.assertLess(size_kb, 200.0, f"Payload size {size_kb:.2f} KB exceeds 200 KB for density {density}")
            self.assertIn("b64", result)
            self.assertGreater(len(result["b64"]), 0)
            
    def test_ssim_mse_diffing_under_40ms(self):
        """Requirement 2 & Constraint: Local SSIM image diffing completes in under 40ms."""
        img1 = create_synthetic_frame("Frame 1")
        img2 = create_synthetic_frame("Frame 2")
        
        diff_res = compute_ssim_mse(img1, img2)
        diff_ms = diff_res["diff_ms"]
        
        self.assertLess(diff_ms, 40.0, f"Diffing time {diff_ms:.2f} ms exceeds 40ms limit")
        self.assertIn("ssim", diff_res)
        self.assertIn("variance", diff_res)
        
    def test_static_screen_detection_and_vlm_bypass(self):
        """Requirement 3 & Acceptance Criteria: SSIM detects static screen (<2% variance) and bypasses VLM."""
        pipeline = VisualStreamPipeline()
        
        img_frame = create_synthetic_frame("Static Screen Test")
        
        # Step 1: Initial frame, VLM called
        vlm_calls = []
        def dummy_vlm(b64):
            vlm_calls.append(1)
            return {"action": "click", "selector": "#btn-1", "caption": "Clicking button 1"}
            
        res1 = pipeline.process_frame(img_frame, vlm_callback=dummy_vlm)
        self.assertFalse(res1["vlm_bypassed"], "First frame should not bypass VLM")
        self.assertEqual(len(vlm_calls), 1)
        
        # Step 2: Identical frame (0% difference)
        res2 = pipeline.process_frame(img_frame, vlm_callback=dummy_vlm)
        self.assertTrue(res2["is_static"], "Identical frame should be marked static")
        self.assertTrue(res2["vlm_bypassed"], "Identical frame should bypass VLM call")
        self.assertEqual(len(vlm_calls), 1, "VLM call count should remain 1 (bypassed)")
        self.assertEqual(res2["action_result"]["action"], "click")

    def test_spatial_bounding_box_mapping(self):
        """Acceptance Criteria: Spatial bounding box coordinates map accurately on downsampled frames."""
        source_dim = (960, 540)
        target_dim = (1920, 1080)
        
        # Normalized 0-1000 bounding box [ymin, xmin, ymax, xmax]
        norm_bbox = [100.0, 200.0, 300.0, 400.0]
        mapped = map_bounding_box(norm_bbox, source_size=source_dim, target_size=target_dim, bbox_format="norm_1000")
        
        self.assertEqual(len(mapped), 4)
        self.assertAlmostEqual(mapped[0], 100.0, delta=1.0)
        self.assertAlmostEqual(mapped[1], 200.0, delta=1.0)

        # Pixel format bbox [x, y, w, h]
        pixel_bbox = [100, 50, 200, 100]
        mapped_pixel = map_bounding_box(pixel_bbox, source_size=source_dim, target_size=target_dim, bbox_format="pixel")
        self.assertEqual(mapped_pixel, [200.0, 100.0, 400.0, 200.0])

    def test_timeout_bounds_at_45s_maximum(self):
        """Requirement 4 & Acceptance Criteria: Enforce timeout bounds at 45s maximum with graceful handling."""
        pipeline = VisualStreamPipeline(max_timeout_seconds=1.0)  # Short 1.0s timeout for test
        
        img = create_synthetic_frame("Timeout Test")
        
        def slow_vlm(b64):
            time.sleep(2.0)  # Exceeds 1.0s timeout
            return {"action": "click"}
            
        res = pipeline.process_frame(img, vlm_callback=slow_vlm)
        self.assertIn("error", res)
        self.assertIsNotNone(res["error"])
        self.assertIn("Timeout", res["error"])
        self.assertEqual(res["action_result"]["action"], "wait")

    def test_flask_api_visual_process(self):
        """Tests /api/visual_process endpoint integration with Flask test client."""
        from app import app
        client = app.test_client()
        
        img = create_synthetic_frame("API Test Frame")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64_frame = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        resp = client.post("/api/visual_process", json={
            "image": b64_frame,
            "element_density": "high",
            "command": "Click search button"
        })
        
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("payload_size_kb", data)
        self.assertLess(data["payload_size_kb"], 200.0)
        self.assertIn("diff_metrics", data)
        self.assertIn("action_result", data)


if __name__ == "__main__":
    unittest.main()
