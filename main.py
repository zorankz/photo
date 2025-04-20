# main.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import io
import time # To measure processing time
import numpy as np # Needed for OpenCV processing
import cv2 # OpenCV for image processing and Guided Filter
from werkzeug.utils import secure_filename
from rembg import remove, new_session
import logging

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# --- MODELO DE ALTA PRECISIÓN ---
MODEL_NAME = "isnet-general-use" # Still using a high-quality model

# --- Guided Filter Parameters ---
# Ajusta estos valores para controlar el refinamiento del borde
GUIDED_FILTER_RADIUS = 15 # Tamaño del vecindario del filtro (más grande = más suave)
GUIDED_FILTER_EPS = 1e-4  # Regularización (más pequeño = más sensible a los bordes)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Pre-load the rembg session ---
logging.info(f"Attempting to load rembg model: {MODEL_NAME}...")
try:
    session = new_session(MODEL_NAME)
    logging.info(f"Successfully loaded rembg model: {MODEL_NAME}")
except Exception as e:
    logging.error(f"FATAL: Failed to load rembg model '{MODEL_NAME}': {e}", exc_info=True)
    session = None

# --- Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handles image upload, processing (rembg + Guided Filter), and returns result."""
    start_time = time.time()

    if not session:
         logging.error("Processing request failed: Rembg session not available.")
         return jsonify({'error': f"El modelo de IA para quitar fondos ('{MODEL_NAME}') no está disponible. Contacta al administrador."}), 500

    if 'image' not in request.files:
        logging.warning("Upload attempt failed: 'image' part missing in request.")
        return jsonify({'error': 'No se encontró el archivo de imagen en la solicitud.'}), 400

    file = request.files['image']

    if file.filename == '':
        logging.warning("Upload attempt failed: No filename provided.")
        return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"no-bg-refined-{filename.rsplit('.', 1)[0]}.png" # Indicate refinement
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

        input_image_bytes = None
        output_data_rembg = None

        try:
            # Read the uploaded file into memory first
            input_image_bytes = file.read()
            # Save the original bytes temporarily if needed for debugging or direct rembg file input
            # with open(input_path, 'wb') as f:
            #     f.write(input_image_bytes)
            # logging.info(f"File received: {filename}. Temporarily saved.")

            # --- Step 1: Initial background removal with rembg ---
            logging.info(f"Starting initial background removal for {filename} using model {MODEL_NAME}...")
            rembg_start_time = time.time()
            output_data_rembg = remove(
                input_image_bytes, # Process directly from bytes
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=245,
                alpha_matting_background_threshold=5,
                alpha_matting_erode_size=5, # Keep erode size moderate for initial mask
                alpha_matting_iterations=3, # Reduce iterations slightly, Guided Filter will refine
                post_process_mask=True,
            )
            rembg_time = time.time() - rembg_start_time
            logging.info(f"Rembg processing for {filename} completed in {rembg_time:.2f} seconds.")

            # --- Step 2: Refine mask with Guided Filter using OpenCV ---
            logging.info(f"Starting Guided Filter refinement for {filename}...")
            refine_start_time = time.time()

            # Decode original image (for guide) and rembg output (for mask)
            # Use cv2.imdecode to read from bytes
            original_img_np = cv2.imdecode(np.frombuffer(input_image_bytes, np.uint8), cv2.IMREAD_COLOR)
            rembg_output_np = cv2.imdecode(np.frombuffer(output_data_rembg, np.uint8), cv2.IMREAD_UNCHANGED)

            if original_img_np is None or rembg_output_np is None:
                raise ValueError("Could not decode images using OpenCV.")

            # Ensure rembg output has 4 channels (RGBA)
            if rembg_output_np.shape[2] != 4:
                 raise ValueError("Rembg output does not have an alpha channel.")

            # Extract the initial alpha mask from rembg output
            alpha_mask_initial = rembg_output_np[:, :, 3].astype(np.float32) / 255.0 # Normalize to 0.0-1.0

            # Apply Guided Filter
            # The guide image should be the original color image
            # The source image to filter is the initial alpha mask
            # Requires opencv-contrib-python
            try:
                alpha_mask_refined = cv2.ximgproc.guidedFilter(
                    guide=original_img_np,
                    src=alpha_mask_initial,
                    radius=GUIDED_FILTER_RADIUS,
                    eps=GUIDED_FILTER_EPS,
                    # dDepth=-1 means output has same depth as src
                )
            except AttributeError:
                 logging.error("cv2.ximgproc.guidedFilter not found. Is opencv-contrib-python installed?")
                 raise ImportError("Guided Filter requires opencv-contrib-python package.")


            # Clip values to be strictly within [0, 1] range after filtering
            alpha_mask_refined = np.clip(alpha_mask_refined, 0, 1)

            # Convert refined mask back to 8-bit integer format (0-255)
            alpha_mask_refined_8bit = (alpha_mask_refined * 255).astype(np.uint8)

            # Create the final RGBA image: merge original RGB with the *refined* alpha mask
            final_rgba_image = cv2.merge((original_img_np[:, :, 0],  # Blue
                                          original_img_np[:, :, 1],  # Green
                                          original_img_np[:, :, 2],  # Red
                                          alpha_mask_refined_8bit)) # Refined Alpha

            refine_time = time.time() - refine_start_time
            logging.info(f"Guided Filter refinement completed in {refine_time:.2f} seconds.")

            # --- Step 3: Encode final image and save ---
            # Encode the final RGBA image to PNG format in memory
            encode_success, output_buffer = cv2.imencode('.png', final_rgba_image)
            if not encode_success:
                raise ValueError("Could not encode final image to PNG format.")

            final_output_data = output_buffer.tobytes()

            # Save the final processed image
            with open(output_path, 'wb') as o:
                o.write(final_output_data)
            logging.info(f"Refined result saved to: {output_path}")

            # Return the URL
            result_url = f'/results/{output_filename}'
            total_time = time.time() - start_time
            logging.info(f"Total request time for {filename}: {total_time:.2f} seconds (rembg: {rembg_time:.2f}s, refine: {refine_time:.2f}s).")
            # Include timings if useful for frontend
            return jsonify({
                'url': result_url,
                'processing_time_rembg': round(rembg_time, 2),
                'processing_time_refine': round(refine_time, 2),
                'processing_time_total': round(total_time, 2)
            })

        except ImportError as imp_err:
             logging.error(f"Import error: {imp_err}", exc_info=True)
             return jsonify({'error': f'Error de configuración del servidor: Falta una librería necesaria ({imp_err}).'}), 500
        except ValueError as val_err:
             logging.error(f"Value error during processing for {filename}: {val_err}", exc_info=True)
             return jsonify({'error': f'Error al procesar los datos de la imagen: {val_err}.'}), 500
        except Exception as e:
            logging.error(f"Unhandled error processing file {filename}: {e}", exc_info=True)
            # Clean up potentially corrupted files
            # if os.path.exists(input_path): os.remove(input_path) # If temp file was saved
            if os.path.exists(output_path):
                 try:
                     os.remove(output_path)
                     logging.info(f"Cleaned up potentially corrupt output file: {output_path}")
                 except OSError as rm_err:
                     logging.warning(f"Could not remove output file during error cleanup {output_path}: {rm_err}")
            return jsonify({'error': 'Ocurrió un error interno inesperado al procesar la imagen.'}), 500
        # No finally block needed for input file cleanup as we read directly from bytes

    else:
        logging.warning(f"Upload attempt failed: Invalid file type for filename '{file.filename}'.")
        return jsonify({'error': 'Tipo de archivo no permitido. Usa PNG, JPG, JPEG o WEBP.'}), 400

@app.route('/results/<filename>')
def serve_result(filename):
    """Serves the processed image from the results folder."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

# --- Run Application ---
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
