import os
from flask import Flask, request, render_template_string, redirect, url_for, jsonify, send_from_directory
from google.cloud import vision
from PIL import Image, ImageDraw, ImageOps
import io
from dotenv import load_dotenv
import base64
from werkzeug.utils import secure_filename
import datetime
import fitz  # PyMuPDF
import google.generativeai as genai
import uuid
import multiprocessing

# Import the agents and the Phoenix setup function
from script_marker_agent.ocr_refinement_agent import OcrRefinementAgent, setup_phoenix_tracing

load_dotenv()
genai.configure()

app = Flask(__name__)

# --- Job Management Setup ---
# The manager and shared jobs dict will be created in the main block
# to ensure Windows compatibility with multiprocessing.


# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_document_text(image_path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Add language hints to improve OCR accuracy for letters like "I"
    image_context = vision.ImageContext(language_hints=["en"])
    response = client.document_text_detection(image=image, image_context=image_context)

    return response.full_text_annotation

def highlight_text_on_image_in_memory(image_path, text_annotation):
    """Highlights detected text on an image with transparency and returns it from memory."""
    # Open the base image and ensure it's in RGBA mode to handle transparency
    base_img = Image.open(image_path).convert("RGBA")

    # Create a transparent overlay to draw on
    overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Set opacity to 20% (255 * 0.20 = 51)
    # The RGBA fill is (Red, Green, Blue, Alpha)
    highlight_fill = (255, 0, 0, 51)

    for page in text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    vertices = [(v.x, v.y) for v in word.bounding_box.vertices]
                    draw.polygon(vertices, outline='red', fill=highlight_fill)

    # Combine the overlay with the base image
    combined_img = Image.alpha_composite(base_img, overlay)
    
    # Convert back to RGB for saving as JPEG (which doesn't support alpha)
    final_img = combined_img.convert("RGB")

    # Save the image to an in-memory buffer
    buffer = io.BytesIO()
    final_img.save(buffer, 'JPEG')
    buffer.seek(0)
    return buffer

# HTML Template for the page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini Edexia - Script Marker</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 2rem; }
        .header h1 { color: #1c1e21; }
        .upload-section { display: flex; flex-wrap: wrap; justify-content: center; gap: 2rem; max-width: 1200px; margin: 0 auto 2rem auto; }
        .card { flex: 1; min-width: 300px; max-width: 500px; padding: 2rem; background-color: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .card h2 { margin-top: 0; }
        .card button { width: 100%; padding: 12px; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; transition: background-color 0.2s; }
        .card button:hover { background-color: #0056b3; }
        #camera-view { display: none; margin-top: 1rem; }
        #camera-feed { width: 100%; border-radius: 4px; }
        .results-container { display: flex; max-width: 1200px; margin: 2rem auto; gap: 30px; }
        .container { flex: 1; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h2 { color: #1c1e21; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        pre { white-space: pre-wrap; word-wrap: break-word; background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; line-height: 1.6; }
        .error { color: #d9534f; text-align: center; margin-top: 1rem; }
        .spinner { margin: 20px auto; border: 5px solid rgba(0, 0, 0, 0.1); width: 40px; height: 40px; border-radius: 50%; border-left-color: #007bff; animation: spin 1s ease infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        #loading-indicator { text-align: center; color: #555; }
        .loading-placeholder { position: relative; display: flex; justify-content: center; align-items: center; height: 300px; color: #888; font-size: 20px; }
        .image-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255, 255, 255, 0.7); display: flex; justify-content: center; align-items: center; flex-direction: column; }
    </style>
</head>
<body>
    <div class="header"><h1>Mini Edexia Script Marker</h1></div>
    <div class="upload-section">
        <div class="card">
            <h2>Option 1: Upload a File</h2>
            <form id="upload-form" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".png, .jpg, .jpeg, .gif, .pdf">
                <button id="submit-button" type="submit">Process Script</button>
            </form>
        </div>
        <div class="card">
            <h2>Option 2: Use Camera</h2>
            <button id="start-camera">Start Camera</button>
            <div id="camera-view">
                <video id="camera-feed" playsinline></video>
                <button id="capture-photo" style="margin-top: 1rem;">Capture and Process</button>
            </div>
        </div>
    </div>

    <div id="loading-indicator" style="display: none;">
        <div class="spinner"></div>
        <p>Processing, please wait...</p>
    </div>

    {% if error %}
        <p class="error" style="text-align: center;">{{ error }}</p>
    {% endif %}

    {% if results %}
    <div class="results-container">
        <div class="container">
            <h2>Highlighted Image</h2>
            <img src="data:image/jpeg;base64,{{ results.img_base64 }}" alt="Highlighted OCR Image" />
        </div>
        <div class="container">
            <h2>Refined Text</h2>
            <pre>{{ results.refined_text }}</pre>
        </div>
    </div>
    {% endif %}

    <script>
        // --- Loading Indicator Logic ---
        function showLoading() {
            document.querySelector('.upload-section').style.display = 'none';
            document.getElementById('loading-indicator').style.display = 'block';
        }
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            const fileInput = this.querySelector('input[type="file"]');
            if (fileInput.files.length > 0) {
                showLoading();
            }
        });

        // --- Camera Logic ---
        const startCameraButton = document.getElementById('start-camera');
        const captureButton = document.getElementById('capture-photo');
        const cameraView = document.getElementById('camera-view');
        const video = document.getElementById('camera-feed');
        let stream;

        startCameraButton.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
                video.srcObject = stream;
                await video.play();
                startCameraButton.style.display = 'none';
                cameraView.style.display = 'block';
            } catch (err) {
                console.error("Error accessing camera: ", err);
                alert('Could not access the camera. Please ensure you have given permission and are using a secure connection (https).');
            }
        });

        captureButton.addEventListener('click', () => {
            showLoading();
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            // Convert canvas image to blob and send to server
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('file', blob, 'capture.jpg');
                
                fetch('/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.text())
                .then(html => {
                    document.open();
                    document.write(html);
                    document.close();
                })
                .catch(error => {
                    console.error('Error uploading captured image:', error);
                    window.location.reload(); // Reload on error
                });
            }, 'image/jpeg');
        });
    </script>
</body>
</html>
"""

def convert_pdf_to_images(pdf_path):
    """Converts each page of a PDF into a PNG image."""
    image_paths = []
    # Create a unique directory for this PDF's pages
    pdf_filename = os.path.basename(pdf_path)
    # Remove the extension to get a clean directory name
    pages_dir = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(pdf_filename)[0])
    os.makedirs(pages_dir, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
        print(f"Converting PDF '{pdf_filename}' with {len(doc)} pages.")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            output_path = os.path.join(pages_dir, f"page-{page_num + 1}.png")
            pix.save(output_path)
            image_paths.append(output_path)
        doc.close()
        print(f"Finished converting PDF. Images saved to '{pages_dir}'.")
        return image_paths, None
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return [], f"An error occurred while converting the PDF: {e}"


def process_image_pipeline(image_path):
    """The main processing pipeline for an image."""
    
    # 1. Run initial OCR
    full_annotation = detect_document_text(image_path)
    if not full_annotation or not full_annotation.text:
        return None, "Could not extract any text from the image."

    raw_text = full_annotation.text

    # 2. Run the OcrRefinementAgent
    refinement_agent = OcrRefinementAgent()
    refined_text = refinement_agent.run(
        raw_ocr_text=raw_text,
        original_image_path=image_path
    )

    # 3. Highlight the image
    image_buffer = highlight_text_on_image_in_memory(image_path, full_annotation)
    img_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    
    results = {
        "refined_text": refined_text,
        "img_base64": img_base64
    }
    return results, None

# --- Background Worker Function ---
def run_processing_job(job_id, image_paths, jobs_dict):
    """
    This function runs in a separate process to handle the heavy lifting.
    It processes each image and updates the shared jobs dictionary.
    """
    total_pages = len(image_paths)
    print(f"WORKER (Job {job_id}): Starting processing for {total_pages} pages.")

    for i, image_path in enumerate(image_paths):
        page_num = i + 1
        
        # Update the main job status to show which page is being worked on now.
        job_data = jobs_dict[job_id]
        job_data['currently_processing_page'] = page_num
        jobs_dict[job_id] = job_data

        print(f"WORKER (Job {job_id}): Processing page {page_num}/{total_pages}...")
        
        # Update status to processing
        jobs_dict[job_id]["pages"][page_num]["status"] = "processing"
        
        try:
            results, error = process_image_pipeline(image_path)
            if error:
                raise Exception(error)
            
            # --- START OF FIX ---
            # Get the current state of the entire job from the proxy
            job_data = jobs_dict[job_id]
            # Get the state of the pages dictionary proxy
            pages_data = job_data['pages']
            # Create a new, normal dictionary with the updated page results
            updated_page_data = {
                "status": "completed",
                "result": results,
                "error": None
            }
            # Assign the new dictionary to the pages proxy. This forces the update.
            pages_data[page_num] = updated_page_data
            # Re-assign the pages proxy back to the main job proxy to be certain.
            job_data['pages'] = pages_data
            jobs_dict[job_id] = job_data
            # --- END OF FIX ---

            print(f"WORKER (Job {job_id}): Page {page_num} completed successfully.")

        except Exception as e:
            print(f"WORKER (Job {job_id}): Error on page {page_num}: {e}")
            
            # --- Applying same fix for the error case ---
            job_data = jobs_dict[job_id]
            pages_data = job_data['pages']
            updated_page_data = {
                "status": "failed",
                "result": None,
                "error": str(e)
            }
            pages_data[page_num] = updated_page_data
            job_data['pages'] = pages_data
            jobs_dict[job_id] = job_data
    
    job_data = jobs_dict[job_id]
    job_data["status"] = "completed"
    jobs_dict[job_id] = job_data # Re-assign
    print(f"WORKER (Job {job_id}): All pages processed. Job complete.")


@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    jobs = app.config['JOBS'] # Get the shared jobs dict from app config
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, error="No file part in the request.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, error="No file selected.")

        if file and allowed_file(file.filename):
            if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or not os.getenv("GOOGLE_API_KEY"):
                return render_template_string(HTML_TEMPLATE, error="Server credential configuration error.")

            # --- Start Asynchronous Job ---
            job_id = str(uuid.uuid4())
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{timestamp}-{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image_paths_to_process = []
            if filename.lower().endswith('.pdf'):
                image_paths, error = convert_pdf_to_images(file_path)
                if error: return render_template_string(HTML_TEMPLATE, error=error)
                image_paths_to_process = image_paths
            else:
                image_paths_to_process.append(file_path)

            if not image_paths_to_process:
                return render_template_string(HTML_TEMPLATE, error="No images found to process.")
            
            # Initialize job status in the shared dictionary
            # We now also store the web-accessible path to the raw page image
            pages_data = {}
            for i, image_path in enumerate(image_paths_to_process):
                page_num = i + 1
                # Create a path relative to the 'uploads' folder for web access
                relative_path = os.path.relpath(image_path, app.config['UPLOAD_FOLDER']).replace('\\', '/')
                pages_data[page_num] = {
                    "status": "pending", 
                    "result": None, 
                    "error": None,
                    "raw_image_path": url_for('serve_upload', filename=relative_path)
                }

            jobs[job_id] = {
                "status": "processing",
                "total_pages": len(image_paths_to_process),
                "currently_processing_page": 1,
                "pages": pages_data
            }
            
            # Start the background process
            process = multiprocessing.Process(
                target=run_processing_job,
                args=(job_id, image_paths_to_process, jobs)
            )
            process.start()

            # Redirect the user to the results page for this job
            return redirect(url_for('show_results', job_id=job_id))

        else:
            return render_template_string(HTML_TEMPLATE, error="File type not allowed.")

    return render_template_string(HTML_TEMPLATE)

@app.route('/results/<job_id>')
def show_results(job_id):
    """
    This is the page the user sees while the job is processing.
    It now includes a paginator UI.
    """
    jobs = app.config['JOBS']
    if job_id not in jobs:
        return "Job not found.", 404
    
    total_pages = jobs[job_id].get('total_pages', 0)

    # This will be the new, interactive results page.
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Processing Results</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }
            .header { text-align: center; margin-bottom: 1rem; }
            .paginator { display: flex; justify-content: center; align-items: center; gap: 1rem; margin-bottom: 2rem; }
            .paginator button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            #page-indicator { font-size: 18px; font-weight: bold; }
            .results-container { display: flex; max-width: 1200px; margin: 0 auto; gap: 30px; }
            .container { flex: 1; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h2 { color: #1c1e21; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
            pre { white-space: pre-wrap; word-wrap: break-word; background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; line-height: 1.6; }
            .loading-placeholder { display: flex; justify-content: center; align-items: center; height: 300px; color: #888; font-size: 20px; }
            .spinner { margin: 20px auto; border: 5px solid rgba(0, 0, 0, 0.1); width: 40px; height: 40px; border-radius: 50%; border-left-color: #007bff; animation: spin 1s ease infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Document Results</h1>
            <p>Job ID: {{ job_id }}</p>
            <p><a href="/">Process another document</a></p>
        </div>

        <div class="paginator">
            <button id="prev-page" disabled>&laquo; Previous</button>
            <span id="page-indicator">Page 1 / {{ total_pages }}</span>
            <button id="next-page">Next &raquo;</button>
        </div>

        <div class="results-container">
            <div class="container" id="image-container">
                <h2>Highlighted Image</h2>
                <div class="loading-placeholder"><div class="spinner"></div></div>
            </div>
            <div class="container" id="text-container">
                <h2>Refined Text</h2>
                <div class="loading-placeholder" id="text-content">Processing page...</div>
            </div>
        </div>

        <script>
            const JOB_ID = "{{ job_id }}";
            const TOTAL_PAGES = parseInt("{{ total_pages }}", 10);
            let currentPage = 1;
            let pollingTimerId = null; // Keep track of the current polling timer

            const prevButton = document.getElementById('prev-page');
            const nextButton = document.getElementById('next-page');
            const pageIndicator = document.getElementById('page-indicator');
            const imageContainer = document.getElementById('image-container');
            const textContainer = document.getElementById('text-container');
            
            function updateNavButtons() {
                prevButton.disabled = currentPage === 1;
                nextButton.disabled = currentPage === TOTAL_PAGES;
            }

            function prefetchPage(pageNumber) {
                if (pageNumber < 1 || pageNumber > TOTAL_PAGES) {
                    return; // Page is out of bounds, do nothing.
                }
                // This silently fetches the data for a page and warms up the browser cache.
                // We don't need to do anything with the response.
                console.log(`Prefetching data for page ${pageNumber}`);
                fetch(`/api/job_status/${JOB_ID}`);
            }
            
            function fetchAndRenderPage(pageNumber) {
                // Clear any previously scheduled polling check to prevent race conditions
                if (pollingTimerId) {
                    clearTimeout(pollingTimerId);
                    pollingTimerId = null;
                }

                pageIndicator.textContent = `Page ${pageNumber} / ${TOTAL_PAGES}`;
                
                // Set initial loading state ONLY if the content isn't already there from a previous load
                if (!document.getElementById('image-content-wrapper')) {
                    imageContainer.innerHTML = `<h2>Highlighted Image</h2><div id="image-content-wrapper"><div class="loading-placeholder"><div class="spinner"></div></div></div>`;
                    textContainer.innerHTML = `<h2>Refined Text</h2><div class="loading-placeholder" id="text-content">Processing page...</div>`;
                }

                fetch(`/api/job_status/${JOB_ID}`)
                    .then(response => response.json())
                    .then(data => {
                        const pageData = data.pages[pageNumber];

                        if (pageData.status === 'completed') {
                            const result = pageData.result;
                            document.getElementById('image-container').innerHTML = `<h2>Highlighted Image</h2><img src="data:image/jpeg;base64,${result.img_base64}" alt="Highlighted OCR Image" />`;
                            document.getElementById('text-container').innerHTML = `<h2>Refined Text</h2><pre>${result.refined_text}</pre>`;

                            // Smart Pre-fetching
                            prefetchPage(pageNumber - 1);
                            prefetchPage(pageNumber + 1);
                        } else if (pageData.status === 'failed') {
                            document.getElementById('image-container').innerHTML = '<h2>Highlighted Image</h2><div class="loading-placeholder">Failed to process.</div>';
                            document.getElementById('text-container').innerHTML = `<h2>Refined Text</h2><div class="loading-placeholder">Error: ${pageData.error}</div>`;
                        } else {
                            // --- START OF FIX ---
                            // We are still loading. Only update the part of the DOM that needs to change.
                            const textContent = document.getElementById('text-content');
                            const imageContentWrapper = document.getElementById('image-content-wrapper');

                            // If the image placeholder isn't there yet, create it.
                            if (!imageContentWrapper || imageContentWrapper.className !== 'loading') {
                                document.getElementById('image-container').innerHTML = `<h2>Highlighted Image</h2>
                                    <div id="image-content-wrapper" class="loading" style="position: relative;">
                                        <img src="${pageData.raw_image_path}" alt="Page ${pageNumber} is processing">
                                        <div class="image-overlay">
                                            <div class="spinner"></div>
                                            Processing...
                                        </div>
                                    </div>`;
                            }
                            
                            // Only update the text content. This prevents the whole page from re-rendering.
                            const currentlyProcessing = data.currently_processing_page || '...';
                            if (textContent) {
                                textContent.textContent = `Processing page ${currentlyProcessing}...`;
                            }
                            // --- END OF FIX ---
                            
                            // Schedule a new check
                            pollingTimerId = setTimeout(() => fetchAndRenderPage(pageNumber), 2000);
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching page data:', error);
                        imageContainer.innerHTML = '<h2>Highlighted Image</h2><div class="loading-placeholder">Error fetching data.</div>';
                        textContainer.innerHTML = '<h2>Refined Text</h2><div class="loading-placeholder">Please refresh.</div>';
                    });
            }
            
            prevButton.addEventListener('click', () => {
                if (currentPage > 1) {
                    currentPage--;
                    // Reset the content before fetching the new page to ensure a clean state
                    imageContainer.innerHTML = '';
                    textContainer.innerHTML = '';
                    updateNavButtons();
                    fetchAndRenderPage(currentPage);
                }
            });

            nextButton.addEventListener('click', () => {
                if (currentPage < TOTAL_PAGES) {
                    currentPage++;
                    // Reset the content before fetching the new page
                    imageContainer.innerHTML = '';
                    textContainer.innerHTML = '';
                    updateNavButtons();
                    fetchAndRenderPage(currentPage);
                }
            });

            // Initial setup
            updateNavButtons();
            fetchAndRenderPage(currentPage); // Fetch the first page on load
        </script>
    </body>
    </html>
    """, job_id=job_id, total_pages=total_pages)


@app.route('/api/job_status/<job_id>')
def job_status(job_id):
    """Provides the current status of a job as JSON."""
    jobs = app.config['JOBS']
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job_proxy = jobs.get(job_id)
    # Perform a deep copy to convert all managed proxies (including nested ones)
    # into regular dicts so they can be properly serialized to JSON.
    job_data = {
        "status": job_proxy.get("status"),
        "total_pages": job_proxy.get("total_pages"),
        "currently_processing_page": job_proxy.get("currently_processing_page"), # Add new field
        "pages": {}
    }

    # Manually iterate and deep copy to ensure no proxies remain. This is the key fix.
    for page_num, page_proxy in job_proxy.get("pages", {}).items():
        page_dict = {
            "status": page_proxy.get("status"),
            "error": page_proxy.get("error"),
            "raw_image_path": page_proxy.get("raw_image_path"), # Pass the raw path
            "result": None
        }
        result_proxy = page_proxy.get("result")
        if result_proxy:
            # Explicitly convert the nested result object.
            page_dict["result"] = dict(result_proxy)
        
        job_data["pages"][page_num] = page_dict
        
    return jsonify(job_data)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serves files from the uploads directory."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Create the manager and shared dictionary here
    manager = multiprocessing.Manager()
    jobs = manager.dict()
    # Store the shared dictionary in the Flask app config
    app.config['JOBS'] = jobs

    # setup_phoenix_tracing() # Tracing can be noisy with multiprocessing, disabling for now.
    app.run(debug=True, use_reloader=False) 