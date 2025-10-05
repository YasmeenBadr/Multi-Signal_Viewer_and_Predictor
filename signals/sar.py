from flask import Blueprint, request, jsonify
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
from skimage.filters import threshold_minimum

bp = Blueprint('sar', __name__, url_prefix="/sar")

@bp.route('/view', methods=['POST'])
def sar_view():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.tif', '.tiff')):
        upload_path = 'temp.tiff'
        file.save(upload_path)

        with rasterio.open(upload_path) as src:
            scale_factor = 0.1
            new_height = int(src.height * scale_factor)
            new_width = int(src.width * scale_factor)
            band = src.read(
                1,
                out_shape=(1, new_height, new_width),
                resampling=rasterio.enums.Resampling.average
            ).astype(np.float32)

        # Speckle reduction
        band = gaussian_filter(band, sigma=1)

        # Correct dB conversion with approximate calibration
        calibration_offset = 49.0  # Based on typical sigmaNought LUT ~270-320
        band_db = 20 * np.log10(np.maximum(band, 1e-5)) - calibration_offset

        mean = np.mean(band_db)
        std = np.std(band_db)
        min_val = np.min(band_db)
        max_val = np.max(band_db)

        # Automatic threshold using threshold_minimum for bimodal distribution
        try:
            threshold = threshold_minimum(band_db.ravel())
        except ValueError:
            threshold = -14.0  # Fallback to empirical value for VV

        low_backscatter_mask = band_db < threshold

        # Morphology to clean mask
        low_backscatter_mask = binary_opening(low_backscatter_mask, structure=np.ones((3, 3)))
        low_backscatter_mask = binary_closing(low_backscatter_mask, structure=np.ones((3, 3)))

        low_backscatter_ratio = np.sum(low_backscatter_mask) / band_db.size

        stats = {
            'mean': float(mean),
            'std': float(std),
            'min': float(min_val),
            'max': float(max_val),
            'threshold': float(threshold),
            'low_backscatter_ratio': float(low_backscatter_ratio)
        }

        output_dir = 'static/generated/sar/'
        os.makedirs(output_dir, exist_ok=True)

        p2, p98 = np.percentile(band_db, (2, 98))
        band_scaled = np.clip((band_db - p2) / (p98 - p2), 0, 1)
        img_main = Image.fromarray((band_scaled * 255).astype(np.uint8))
        main_path = os.path.join(output_dir, 'main.png')
        img_main.save(main_path)

        plt.figure(figsize=(8, 6))
        plt.hist(band_db.ravel(), bins=100, color='blue', alpha=0.7)
        plt.title('Backscatter Distribution (dB)')
        plt.xlabel('dB Value')
        plt.ylabel('Frequency')
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1)
        hist_path = os.path.join(output_dir, 'histogram.png')
        plt.savefig(hist_path)
        plt.close()

        img_mask = Image.fromarray((low_backscatter_mask * 255).astype(np.uint8))
        mask_path = os.path.join(output_dir, 'mask.png')
        img_mask.save(mask_path)

        band_scaled_rgb = np.stack([band_scaled, band_scaled, band_scaled], axis=-1) * 255
        overlay = np.zeros_like(band_scaled_rgb)
        overlay[low_backscatter_mask] = [255, 0, 0]
        alpha = 0.5
        overlaid = (band_scaled_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)
        img_overlay = Image.fromarray(overlaid)
        overlay_path = os.path.join(output_dir, 'overlay.png')
        img_overlay.save(overlay_path)

        os.remove(upload_path)

        results = {
            'main_image': '/static/generated/sar/main.png',
            'histogram': '/static/generated/sar/histogram.png',
            'mask': '/static/generated/sar/mask.png',
            'overlay': '/static/generated/sar/overlay.png',
            'stats': stats
        }

        return jsonify(results)

    else:
        return jsonify({'error': 'Invalid file type'}), 400