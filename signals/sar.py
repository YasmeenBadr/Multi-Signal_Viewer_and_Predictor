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
    # Create in-memory file buffer
    buf = BytesIO()  
    # Save figure to buffer
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    # Reset buffer position to start
    buf.seek(0)
    # Convert to base64 string
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    # Close figure to free memory
    plt.close(fig)
    # Return browser-ready data URL
    return f"data:image/png;base64,{img_base64}"


# Define endpoint: POST /sar/view
@bp.route("/view", methods=["POST"])
def view_sar():
    """Process and visualize SAR GeoTIFF file"""
    
    print("=== SAR View Endpoint Called ===")  # Debug log
    
     # Check if file exists in request
    if 'file' not in request.files:
        print("ERROR: No file in request")
        return jsonify({"error": "No file uploaded"}), 400

     # Get uploaded file 
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        print("ERROR: Empty filename")
        return jsonify({"error": "No file selected"}), 400
    
    print(f"Received file: {file.filename}")
    
     # Create temp file path
    temp_path = os.path.join('uploads', 'temp_sar.tif')
    # Create uploads folder if doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    try:
        # Save uploaded file to disk
        file.save(temp_path)
        print(f"File saved to: {temp_path}")
        
        # Try importing rasterio here to catch import errors
        try:
            import rasterio
            print("Rasterio imported successfully")
        except ImportError as e:
            print(f"ERROR: Cannot import rasterio: {e}")
            return jsonify({"error": "rasterio library not installed. Run: pip install rasterio"}), 500
        
        # Open and Read the GeoTIFF file
        print("Opening GeoTIFF file...")
        with rasterio.open(temp_path) as src:
            print(f"File opened. Bands: {src.count}, Shape: {src.shape}")
            
            # Get image dimensions
            height, width = src.shape
            # Maximum dimension for processing to reduce memory usage
            max_dimension = 2000  
            
            # Downsample if image is too large
            if max(height, width) > max_dimension:
                scale_factor = max_dimension / max(height, width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                print(f"Downsampling from {height}x{width} to {new_height}x{new_width}")
                
                # Read with minimizing the dimensions 
                data = src.read(1, 
                               out_shape=(new_height, new_width),
                               resampling=rasterio.enums.Resampling.average)
            else:
                # Read at full resolution
                data = src.read(1)  
            
            print(f"Data shape: {data.shape}, dtype: {data.dtype}")
        
        # Convert to float32 (saves memory vs float64)
        data = data.astype(np.float32)
        
        # Replace zero/negative values with NaN (can't take log)
        data[data <= 0] = np.nan  
        
        # Convert to dB scale (standard radar scale)
        data_db = 10 * np.log10(data)
        
        
        # Remove NaN/Inf for statistics
         # Get only valid (non-NaN, non-Inf) data for statistics
        valid_data = data_db[np.isfinite(data_db)]
        
        # Check if we have any valid data
        if len(valid_data) == 0:
            print("ERROR: No valid data after filtering")
            return jsonify({"error": "No valid data in file"}), 400
        
        
        # Calculate statistics for processing
        mean_db = np.mean(valid_data)
        std_db = np.std(valid_data)
        
        # Adaptive threshold for low-backscatter detection
        threshold = mean_db - 1.5 * std_db
        
        # Create mask for pixels below threshold
        low_backscatter_mask = data_db < threshold
        low_backscatter_ratio = np.sum(low_backscatter_mask) / low_backscatter_mask.size
        
        
        
        # --- 1. Main Display (contrast stretched for better visibility) ---
        print("Generating main display...")
        # Get 2nd and 98th percentiles
        p2, p98 = np.nanpercentile(valid_data, [2, 98])
        # Clip values to this range
        data_display = np.clip(data_db, p2, p98)
         # Normalize to [0, 1]
        data_display = (data_display - p2) / (p98 - p2) 
        

         # Create figure
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        # Display as grayscale
        im1 = ax1.imshow(data_display, cmap='gray', interpolation='nearest')
        ax1.set_title('SAR Backscatter (2-98% Scaled)', fontsize=14, fontweight='bold')
        # Hide axes
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Normalized Intensity')
        # Convert to base64
        main_image = fig_to_base64(fig1)
        print("Main display generated")
        

        # --- 2. Histogram ---
        print("Generating histogram...")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(valid_data, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        # Add threshold line
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f} dB')
        ax2.set_xlabel('Backscatter (dB)', fontsize=12)
        ax2.set_ylabel('Pixel Count', fontsize=12)
        ax2.set_title('Backscatter Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        histogram = fig_to_base64(fig2)
        print("Histogram generated")
        
        # --- 3. Overlay (Red highlight on low-backscatter) ---
        print("Generating overlay...")
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        
        # Convert grayscale to RGB (3 channels)
        rgb_base = np.stack([data_display]*3, axis=-1)
        
        #Create red overlay for low-backscatter areas
        rgb_overlay = rgb_base.copy()
        rgb_overlay[low_backscatter_mask] = [1, 0, 0]  # Red for low backscatter
        
        # Blend original with overlay (50% transparency)
        alpha = 0.5
        rgb_blended = (1 - alpha) * rgb_base + alpha * rgb_overlay
       # Display blended image  
        ax3.imshow(rgb_blended, interpolation='nearest')
        ax3.set_title('Overlay (Low-Backscatter in Red)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        overlay_image = fig_to_base64(fig3)
        print("Overlay generated")
        
        # Package all images into response
        response = {
            "main_image": main_image,
            "histogram": histogram,
            "overlay": overlay_image
        }
        
        print("Response prepared successfully")
        return jsonify(response) # Send JSON response to frontend
    
    except Exception as e:
        import traceback
        # Get full error traceback
        error_msg = traceback.format_exc()
        print(f"ERROR occurred:\n{error_msg}")
        return jsonify({"error": f"Error processing SAR file: {str(e)}"}), 500
    
    finally:
        # Clean up temporary file  (runs even if error occurs)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temporary file cleaned up")