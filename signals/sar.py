from flask import Blueprint, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

bp = Blueprint('sar', __name__)

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

@bp.route("/view", methods=["POST"])
def view_sar():
    """Process and visualize SAR GeoTIFF file"""
    
    print("=== SAR View Endpoint Called ===")
    
    if 'file' not in request.files:
        print("ERROR: No file in request")
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("ERROR: Empty filename")
        return jsonify({"error": "No file selected"}), 400
    
    print(f"Received file: {file.filename}")
    
    # Save file temporarily
    temp_path = os.path.join('uploads', 'temp_sar.tif')
    os.makedirs('uploads', exist_ok=True)
    
    try:
        file.save(temp_path)
        print(f"File saved to: {temp_path}")
        
        # Try importing rasterio here to catch import errors
        try:
            import rasterio
            print("Rasterio imported successfully")
        except ImportError as e:
            print(f"ERROR: Cannot import rasterio: {e}")
            return jsonify({"error": "rasterio library not installed. Run: pip install rasterio"}), 500
        
        # Read the GeoTIFF file
        print("Opening GeoTIFF file...")
        with rasterio.open(temp_path) as src:
            print(f"File opened. Bands: {src.count}, Shape: {src.shape}")
            
            # Downsample large images to reduce memory usage
            height, width = src.shape
            max_dimension = 2000  # Maximum dimension for processing
            
            if max(height, width) > max_dimension:
                scale_factor = max_dimension / max(height, width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                print(f"Downsampling from {height}x{width} to {new_height}x{new_width}")
                
                # Read with downsampling
                data = src.read(1, 
                               out_shape=(new_height, new_width),
                               resampling=rasterio.enums.Resampling.average)
            else:
                data = src.read(1)  # Read at full resolution
            
            print(f"Data shape: {data.shape}, dtype: {data.dtype}")
        
        # Handle invalid values - use float32 instead of float64 to save memory
        data = data.astype(np.float32)
        print(f"Data range before filtering: {np.min(data)} to {np.max(data)}")
        
        data[data <= 0] = np.nan  # Avoid log of zero/negative
        
        # Convert to dB scale
        data_db = 10 * np.log10(data)
        print(f"dB range: {np.nanmin(data_db)} to {np.nanmax(data_db)}")
        
        # Remove NaN/Inf for statistics
        valid_data = data_db[np.isfinite(data_db)]
        
        if len(valid_data) == 0:
            print("ERROR: No valid data after filtering")
            return jsonify({"error": "No valid data in file"}), 400
        
        print(f"Valid pixels: {len(valid_data)}")
        
        # Calculate statistics
        mean_db = np.mean(valid_data)
        std_db = np.std(valid_data)
        min_db = np.min(valid_data)
        max_db = np.max(valid_data)
        
        print(f"Stats - Mean: {mean_db:.2f}, Std: {std_db:.2f}, Min: {min_db:.2f}, Max: {max_db:.2f}")
        
        # Adaptive threshold for low-backscatter detection
        threshold = mean_db - 1.5 * std_db
        
        # Calculate low-backscatter ratio
        low_backscatter_mask = data_db < threshold
        low_backscatter_ratio = np.sum(low_backscatter_mask) / low_backscatter_mask.size
        
        print(f"Threshold: {threshold:.2f}, Low-backscatter ratio: {low_backscatter_ratio:.2%}")
        
        # --- 1. Main Display (2-98% percentile scaling) ---
        print("Generating main display...")
        p2, p98 = np.nanpercentile(valid_data, [2, 98])
        data_display = np.clip(data_db, p2, p98)
        data_display = (data_display - p2) / (p98 - p2)  # Normalize to [0, 1]
        
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        im1 = ax1.imshow(data_display, cmap='gray', interpolation='nearest')
        ax1.set_title('SAR Backscatter (2-98% Scaled)', fontsize=14, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Normalized Intensity')
        main_image = fig_to_base64(fig1)
        print("Main display generated")
        
        # --- 2. Histogram ---
        print("Generating histogram...")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(valid_data, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f} dB')
        ax2.set_xlabel('Backscatter (dB)', fontsize=12)
        ax2.set_ylabel('Pixel Count', fontsize=12)
        ax2.set_title('Backscatter Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        histogram = fig_to_base64(fig2)
        print("Histogram generated")
        
        # --- 3. Water/Low-Backscatter Mask ---
        print("Generating mask...")
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        mask_display = np.where(low_backscatter_mask, 1, 0)
        ax3.imshow(mask_display, cmap='binary', interpolation='nearest')
        ax3.set_title('Low-Backscatter Mask (White = Potential Water)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        mask_image = fig_to_base64(fig3)
        print("Mask generated")
        
        # --- 4. Overlay (Red highlight on low-backscatter) ---
        print("Generating overlay...")
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        
        # Convert grayscale to RGB
        rgb_base = np.stack([data_display]*3, axis=-1)
        
        # Add red overlay where backscatter is low
        rgb_overlay = rgb_base.copy()
        rgb_overlay[low_backscatter_mask] = [1, 0, 0]  # Red for low backscatter
        
        # Blend original with overlay
        alpha = 0.5
        rgb_blended = (1 - alpha) * rgb_base + alpha * rgb_overlay
        
        ax4.imshow(rgb_blended, interpolation='nearest')
        ax4.set_title('Overlay (Low-Backscatter in Red)', fontsize=14, fontweight='bold')
        ax4.axis('off')
        overlay_image = fig_to_base64(fig4)
        print("Overlay generated")
        
        # Prepare response
        response = {
            "main_image": main_image,
            "histogram": histogram,
            "mask": mask_image,
            "overlay": overlay_image,
            "stats": {
                "mean": float(mean_db),
                "std": float(std_db),
                "min": float(min_db),
                "max": float(max_db),
                "threshold": float(threshold),
                "low_backscatter_ratio": float(low_backscatter_ratio)
            }
        }
        
        print("Response prepared successfully")
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"ERROR occurred:\n{error_msg}")
        return jsonify({"error": f"Error processing SAR file: {str(e)}"}), 500
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temporary file cleaned up")