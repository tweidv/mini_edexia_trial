import os
from flask import Flask, request, render_template_string, redirect, url_for, jsonify, send_from_directory, Blueprint
from google.cloud import vision
from PIL import Image, ImageDraw, ImageOps
import io
from dotenv import load_dotenv
import base64
from werkzeug.utils import secure_filename, safe_join
import datetime
import fitz  # PyMuPDF
import google.generativeai as genai
import uuid
import multiprocessing
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import markdown
from script_marker_agent.rubric_formatting_agent import RubricFormattingAgent

# Import the agents and the Phoenix setup function
from script_marker_agent.ocr_refinement_agent import OcrRefinementAgent, setup_phoenix_tracing
from script_marker_agent.agent import RubricCriterionExtractorAgent, MainMarkerAgent, MarkExtractionAgent, ReasoningExtractionAgent, EvidenceExtractionAgent, HighlightExtractionAgent
import threading
import time
import shelve
import tempfile
import shutil
import json

load_dotenv()
genai.configure()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

pure_ocr_bp = Blueprint('pure_ocr', __name__)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    submissions = db.relationship('WorkSubmission', backref='student', lazy=True)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    rubrics = db.relationship('Rubric', backref='task', lazy=True)
    submissions = db.relationship('WorkSubmission', backref='task', lazy=True)
    nudges = db.relationship('Nudge', backref='task', lazy=True)

class Rubric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'), nullable=False)

class WorkSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    filename = db.Column(db.String(256))
    filetype = db.Column(db.String(32))
    upload_time = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    raw_text = db.Column(db.Text)  # For pasted text or extracted text from docx/txt
    cache_dir = db.Column(db.String(256), nullable=True)

class Nudge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'), nullable=False)
    criterion = db.Column(db.String(128), nullable=False)
    nudge_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)


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

def extract_word_data_from_annotation(text_annotation):
    """Extracts text and bounding box vertices for each word."""
    word_data = []
    for page in text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    vertices = [{'x': v.x, 'y': v.y} for v in word.bounding_box.vertices]
                    word_data.append({
                        "text": word_text,
                        "vertices": vertices
                    })
    return word_data

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
                
                fetch('{{ url_for("pure_ocr.upload_and_process") }}', {
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
    
    # 1. Get original image data before any processing
    try:
        with Image.open(image_path) as img:
            original_dimensions = {"width": img.width, "height": img.height}
        with open(image_path, "rb") as image_file:
            raw_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return None, f"Could not read original image: {e}"

    # 2. Run initial OCR
    full_annotation = detect_document_text(image_path)
    if not full_annotation or not full_annotation.text:
        return None, "Could not extract any text from the image."

    raw_text = full_annotation.text

    # 3. Extract word coordinate data from the annotation
    word_data = extract_word_data_from_annotation(full_annotation)

    # 4. Run the OcrRefinementAgent
    refinement_agent = OcrRefinementAgent()
    refined_text = refinement_agent.run(
        raw_ocr_text=raw_text,
        original_image_path=image_path
    )
    
    results = {
        "refined_text": refined_text,
        "raw_image_base64": raw_image_base64,
        "original_dimensions": original_dimensions,
        "word_data": word_data
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


@pure_ocr_bp.route('/', methods=['GET', 'POST'])
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
                    "raw_image_path": url_for('pure_ocr.serve_upload', filename=relative_path)
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
            return redirect(url_for('pure_ocr.show_results', job_id=job_id))

        else:
            return render_template_string(HTML_TEMPLATE, error="File type not allowed.")

    return render_template_string(HTML_TEMPLATE)

@pure_ocr_bp.route('/results/<job_id>')
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
            .results-container { display: flex; max-width: 90vw; margin: 2rem auto; gap: 30px; }
            .container { flex: 1; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            #image-container { flex: 2; }
            h2 { color: #1c1e21; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
            pre { white-space: pre-wrap; word-wrap: break-word; background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; line-height: 1.6; }
            .loading-placeholder { display: flex; justify-content: center; align-items: center; height: 300px; color: #888; font-size: 20px; }
            .spinner { margin: 20px auto; border: 5px solid rgba(0, 0, 0, 0.1); width: 40px; height: 40px; border-radius: 50%; border-left-color: #007bff; animation: spin 1s ease infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .highlight {
                position: absolute;
                border: 1px solid;
                cursor: pointer;
                transition: background-color 0.2s, border-color 0.2s;
            }
            .highlight:hover {
                background-color: rgba(100, 100, 255, 0.4);
                border-color: blue;
            }
            /* New styles for search-based highlighting */
            .highlight.green { background-color: rgba(0, 255, 0, 0.3); border-color: green; }
            .highlight.yellow { background-color: rgba(255, 255, 0, 0.4); border-color: #cca300; }

            .search-section { text-align: center; margin-bottom: 2rem; }
            .search-section input { padding: 10px; font-size: 16px; width: 300px; border: 1px solid #ccc; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Document Results</h1>
            <p>Job ID: {{ job_id }}</p>
            <p><a href="{{ url_for('pure_ocr.upload_and_process') }}">Process another document</a></p>
        </div>

        <div class="paginator">
            <button id="prev-page" disabled>&laquo; Previous</button>
            <span id="page-indicator">Page 1 / {{ total_pages }}</span>
            <button id="next-page">Next &raquo;</button>
        </div>

        <div class="search-section">
            <input type="text" id="search-bar" placeholder="Type to highlight words...">
        </div>

        <div class="results-container">
            <div class="container" id="image-container">
                <h2>Script Image</h2>
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
            let pollingTimerId = null;
            let currentWordData = []; // Store the word data for the current page

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
                fetch(`/pureocr/api/job_status/${JOB_ID}`);
            }
            
            function updateHighlights() {
                const searchTerm = document.getElementById('search-bar').value;
                const imageWrapper = document.getElementById('image-content-wrapper');

                if (!imageWrapper || !currentWordData) return;

                // Remove existing highlights before redrawing
                imageWrapper.querySelectorAll('.highlight').forEach(h => h.remove());

                // If the search term is empty, do not draw any highlights
                if (!searchTerm) {
                    return;
                }

                let highlightsHTML = '';
                const originalWidth = imageWrapper.dataset.originalWidth;
                const originalHeight = imageWrapper.dataset.originalHeight;

                currentWordData.forEach((word, index) => {
                    let className = 'highlight';
                    let shouldHighlight = false;
                    
                    if (word.text.includes(searchTerm)) {
                        className += ' green';
                        shouldHighlight = true;
                    } else if (word.text.toLowerCase().includes(searchTerm.toLowerCase())) {
                        className += ' yellow';
                        shouldHighlight = true;
                    }

                    if (shouldHighlight) {
                        const vertices = word.vertices;
                        const x = vertices[0].x / originalWidth * 100;
                        const y = vertices[0].y / originalHeight * 100;
                        const width = (vertices[1].x - vertices[0].x) / originalWidth * 100;
                        const height = (vertices[2].y - vertices[1].y) / originalHeight * 100;

                        highlightsHTML += `<div class="${className}" data-word-index="${index}" style="left: ${x}%; top: ${y}%; width: ${width}%; height: ${height}%;"></div>`;
                    }
                });
                
                imageWrapper.insertAdjacentHTML('beforeend', highlightsHTML);
            }
            
            function fetchAndRenderPage(pageNumber) {
                // Clear any previously scheduled polling check to prevent race conditions
                if (pollingTimerId) {
                    clearTimeout(pollingTimerId);
                    pollingTimerId = null;
                }

                pageIndicator.textContent = `Page ${pageNumber} / ${TOTAL_PAGES}`;
                
                // Set initial loading state
                document.getElementById('image-container').innerHTML = `<h2>Script Image</h2><div id="image-content-wrapper"><div class="loading-placeholder"><div class="spinner"></div></div></div>`;
                document.getElementById('text-container').innerHTML = `<h2>Refined Text</h2><div class="loading-placeholder" id="text-content">Processing page...</div>`;

                fetch(`/pureocr/api/job_status/${JOB_ID}`)
                    .then(response => response.json())
                    .then(data => {
                        const pageData = data.pages[pageNumber];

                        if (pageData.status === 'completed') {
                            const result = pageData.result;
                            currentWordData = result.word_data; // Store the word data globally for this page
                            
                            const imageHTML = `
                                <h2>Script Image</h2>
                                <div id="image-content-wrapper" style="position: relative; line-height: 0;" data-original-width="${result.original_dimensions.width}" data-original-height="${result.original_dimensions.height}">
                                    <img src="data:image/jpeg;base64,${result.raw_image_base64}" alt="Original Script Image" style="width: 100%;" />
                                </div>`;
                            
                            document.getElementById('image-container').innerHTML = imageHTML;
                            document.getElementById('text-container').innerHTML = `<h2>Refined Text</h2><pre>${result.refined_text}</pre>`;

                            updateHighlights(); // Check if there's any text in the search bar on load

                            // Smart Pre-fetching
                            prefetchPage(pageNumber - 1);
                            prefetchPage(pageNumber + 1);
                        } else if (pageData.status === 'failed') {
                            document.getElementById('image-container').innerHTML = '<h2>Script Image</h2><div class="loading-placeholder">Failed to process.</div>';
                            document.getElementById('text-container').innerHTML = `<h2>Refined Text</h2><div class="loading-placeholder">Error: ${pageData.error}</div>`;
                        } else {
                            // Still loading, show the raw image (no highlights)
                            document.getElementById('image-container').innerHTML = `<h2>Script Image</h2>
                                <div id="image-content-wrapper" class="loading" style="position: relative;">
                                    <img src="${pageData.raw_image_path}" alt="Page ${pageNumber} is processing" style="width: 100%;">
                                    <div class="image-overlay">
                                        <div class="spinner"></div>
                                        Processing...
                                    </div>
                                </div>`;
                            
                            const currentlyProcessing = data.currently_processing_page || '...';
                            const textContent = document.getElementById('text-content');
                            if (textContent) {
                                textContent.textContent = `Processing page ${currentlyProcessing}...`;
                            }
                            
                            // Schedule a new check
                            pollingTimerId = setTimeout(() => fetchAndRenderPage(pageNumber), 2000);
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching page data:', error);
                        imageContainer.innerHTML = '<h2>Script Image</h2><div class="loading-placeholder">Error fetching data.</div>';
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

            // Add event listener for the search bar
            document.getElementById('search-bar').addEventListener('input', updateHighlights);

            // Initial setup
            updateNavButtons();
            fetchAndRenderPage(currentPage); // Fetch the first page on load
        </script>
    </body>
    </html>
    """, job_id=job_id, total_pages=total_pages)


@pure_ocr_bp.route('/api/job_status/<job_id>')
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

@pure_ocr_bp.route('/uploads/<path:filename>')
def serve_upload(filename):
    # Serve files from any subdirectory under uploads/
    uploads_dir = app.config['UPLOAD_FOLDER']
    # Use safe_join to prevent directory traversal
    file_path = safe_join(uploads_dir, filename)
    return send_from_directory(uploads_dir, filename)

@app.template_filter('markdown')
def markdown_filter(s):
    return markdown.markdown(s, extensions=['tables'])

@app.route('/')
def index():
    """Shows the main task dashboard."""
    # Pagination for tasks: show 5 per page, newest first
    page = int(request.args.get('page', 1))
    per_page = 5
    tasks_query = Task.query.order_by(Task.created_at.desc())
    tasks_paginated = tasks_query.paginate(page=page, per_page=per_page, error_out=False)
    tasks = tasks_paginated.items
    students = Student.query.order_by(Student.name).all()
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Mini Edexia Dashboard</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; margin: 40px; }
            .container { max-width: 90vw; width: 90vw; margin: auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1, h2 { color: #1c1e21; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .task-list { list-style: none; padding: 0; }
            .task-item { border: 1px solid #ddd; border-radius: 4px; margin-bottom: 1rem; }
            .task-header { background: #f7f7f7; padding: 1rem; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
            .rubric-content { padding: 1rem; border-top: 1px solid #ddd; display: none; }
            .rubric-content table { width: 100%; border-collapse: collapse; }
            .rubric-content th, .rubric-content td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            .form-section { margin-top: 2rem; border-top: 1px solid #eee; padding-top: 2rem; }
            .tabs { display: flex; border-bottom: 1px solid #ccc; margin-bottom: 1rem; }
            .tab-link { padding: 10px 15px; cursor: pointer; border: 1px solid transparent; border-bottom: none; }
            .tab-link.active { border-color: #ccc; border-bottom-color: white; background: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .form-group { margin-bottom: 1rem; }
            .form-group label { display: block; margin-bottom: .5rem; font-weight: bold; }
            .form-group input, .form-group textarea, .form-group input[type="file"] { width: 100%; padding: .5rem; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; }
            .form-group textarea { min-height: 150px; font-family: monospace; }
            .button { padding: 12px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
            .edit-btn { background: #f0ad4e; margin-right: 10px; }
            .delete-btn { background: #d9534f; }
            .edit-rubric-btn { background: #f0ad4e; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Mini Edexia Dashboard</h1>
            <p><a href="{{ url_for('pure_ocr.upload_and_process') }}">Go to Pure OCR Tool &raquo;</a></p>
            <h2>Existing Tasks</h2>
            {% if tasks %}
                <div class="task-list">
                    {% for task in tasks %}
                        <div class="task-item">
                            <div class="task-header">
                                <span>{{ task.name }}</span>
                                <div>
                                    <span style="color: #888; font-size: 0.95em;">Created: {{ task.created_at.strftime('%Y-%m-%d %H:%M') if task.created_at else 'N/A' }}</span>
                                </div>
                            </div>
                            <div class="rubric-content" id="rubric-content-{{ task.id }}" style="display:none;">
                                <!-- Associated Work List -->
                                <div style="margin-bottom: 1rem;">
                                    <h4 style="margin:0 0 0.5rem 0;">Student Work</h4>
                                    <ul style="list-style:none; padding:0; margin:0;">
                                        {% for work in task.submissions %}
                                            <li style="margin-bottom: 0.5rem; display: flex; align-items: center; gap: 1rem;">
                                                <span style="font-weight:500;">{{ work.student.name }}</span>
                                                <span style="color:#888; font-size:0.95em;">({{ work.upload_time.strftime('%Y-%m-%d %H:%M') }})</span>
                                                <a href="{{ url_for('review_work', submission_id=work.id) }}" class="button" style="background:#eee;color:#333;padding:4px 10px;font-size:14px;">Review</a>
                                                <form method="post" action="{{ url_for('delete_work_submission', submission_id=work.id) }}" style="display:inline; margin:0;" onsubmit="return confirm('Delete this work submission?');">
                                                    <button type="submit" class="button delete-btn" style="padding:4px 10px;font-size:14px;margin-left:6px;background:#d9534f;">Delete</button>
                                                </form>
                                            </li>
                                        {% else %}
                                            <li style="color:#888;">No work uploaded yet.</li>
                                        {% endfor %}
                                    </ul>
                                    <button type="button" class="button" style="margin-top:0.5rem; padding:6px 16px; font-size:15px;" onclick="showAddWorkForm({{ task.id }})" id="add-work-btn-{{ task.id }}">Add Student Work</button>
                                </div>
                                <!-- Add Work Form (hidden by default) -->
                                <div id="add-work-form-{{ task.id }}" style="display:none; margin-bottom:1.5rem;">
                                    <form method="post" action="{{ url_for('add_work_submission', task_id=task.id) }}" enctype="multipart/form-data">
                                        <div class="tabs">
                                            <div class="tab-link active" onclick="openTab(event, 'upload-doc-{{ task.id }}')">Upload Document</div>
                                            <div class="tab-link" onclick="openTab(event, 'paste-text-{{ task.id }}')">Paste Text</div>
                                            <div class="tab-link" onclick="openTab(event, 'camera-{{ task.id }}')">Use Camera</div>
                                        </div>
                                        <div id="upload-doc-{{ task.id }}" class="tab-content active">
                                            <div class="form-group">
                                                <label for="work_file_{{ task.id }}">File (PDF, DOCX, TXT, Image)</label>
                                                <input type="file" id="work_file_{{ task.id }}" name="work_file" accept=".pdf,.docx,.txt,.png,.jpg,.jpeg">
                                            </div>
                                        </div>
                                        <div id="paste-text-{{ task.id }}" class="tab-content">
                                            <div class="form-group">
                                                <label for="work_text_{{ task.id }}">Paste Student Work</label>
                                                <textarea id="work_text_{{ task.id }}" name="work_text"></textarea>
                                            </div>
                                        </div>
                                        <div id="camera-{{ task.id }}" class="tab-content">
                                            <div class="form-group">
                                                <label>Capture Image</label><br>
                                                <button type="button" class="button" id="start-camera-btn-{{ task.id }}">Start Camera</button>
                                                <div id="camera-view-{{ task.id }}" style="display:none; margin-top:1rem;">
                                                    <video id="camera-feed-{{ task.id }}" playsinline style="width:100%;max-width:350px;border-radius:4px;"></video>
                                                    <div id="camera-capture-controls-{{ task.id }}" style="margin-top:1rem;">
                                                        <button type="button" class="button" id="capture-photo-btn-{{ task.id }}">Capture</button>
                                                    </div>
                                                    <div id="camera-preview-{{ task.id }}" style="display:none; margin-top:1rem;">
                                                        <img id="captured-image-{{ task.id }}" style="width:100%;max-width:350px;border-radius:4px;" />
                                                        <div style="margin-top:1rem; display:flex; gap:10px;">
                                                            <button type="button" class="button" id="retake-photo-btn-{{ task.id }}">Retake</button>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="form-group">
                                            <label for="student_{{ task.id }}">Student</label>
                                            <select id="student_{{ task.id }}" name="student_id" required onchange="toggleNewStudentInput({{ task.id }})">
                                                <option value="">Select student...</option>
                                                {% for student in students %}
                                                    <option value="{{ student.id }}">{{ student.name }}</option>
                                                {% endfor %}
                                                <option value="new">New Student</option>
                                            </select>
                                            <input type="text" name="new_student_name" id="new-student-input-{{ task.id }}" placeholder="Enter new student name" style="margin-top:6px;width:100%;display:none;">
                                        </div>
                                        <button type="submit" class="button">Upload Work</button>
                                        <button type="button" class="button" style="background:#eee;color:#333;margin-left:10px;" onclick="hideAddWorkForm({{ task.id }})">Cancel</button>
                                    </form>
                                </div>
                                <div id="rubric-markdown-{{ task.id }}" style="display:block;">
                                    {% if task.rubrics %}
                                        {{ task.rubrics[0].content | markdown | safe }}
                                    {% endif %}
                                </div>
                                <div style="margin-top: 16px; display: flex; flex-direction: row; gap: 10px; align-items: center;">
                                    <button type="button" class="button edit-rubric-btn" style="margin:0;" onclick="showEditForm({{ task.id }})" id="edit-task-btn-{{ task.id }}">Edit Task</button>
                                    <button type="button" class="button" style="background:#aaa;color:#fff;margin:0 0 0 10px;" onclick="showRawEditForm({{ task.id }})" id="raw-edit-btn-{{ task.id }}">Force Edit (Raw)</button>
                                    <form method="post" action="{{ url_for('delete_task', task_id=task.id) }}" style="margin:0;display:inline;" onsubmit="return confirm('Delete this task?');">
                                        <button type="submit" class="button delete-btn" style="margin:0;">Delete</button>
                                    </form>
                                </div>
                                <form method="post" action="{{ url_for('edit_task', task_id=task.id) }}" enctype="multipart/form-data" id="edit-form-{{ task.id }}" style="display:none; margin-top: 20px;">
                                    <button type="button" class="button" style="background:#eee;color:#333;margin-bottom:1rem;" onclick="stopEditingTask({{ task.id }})" id="stop-edit-btn-{{ task.id }}">Stop Editing Task</button>
                                    <div class="form-group">
                                        <label for="task_name_{{ task.id }}">Task Name</label>
                                        <input type="text" id="task_name_{{ task.id }}" name="task_name" value="{{ task.name }}" required>
                                    </div>
                                    <div class="tabs">
                                        <div class="tab-link active" onclick="openTab(event, 'text-rubric-{{ task.id }}')">Paste Rubric Text</div>
                                        <div class="tab-link" onclick="openTab(event, 'file-rubric-{{ task.id }}')">Upload Rubric File</div>
                                    </div>
                                    <div id="text-rubric-{{ task.id }}" class="tab-content active">
                                        <div class="form-group">
                                            <label for="rubric_content_{{ task.id }}">Rubric (paste here)</label>
                                            <textarea id="rubric_content_{{ task.id }}" name="rubric_content">{{ task.rubrics[0].content if task.rubrics else '' }}</textarea>
                                        </div>
                                    </div>
                                    <div id="file-rubric-{{ task.id }}" class="tab-content">
                                        <div class="form-group">
                                            <label for="rubric_file_{{ task.id }}">Rubric File (PDF, PNG, JPG)</label>
                                            <input type="file" id="rubric_file_{{ task.id }}" name="rubric_file" accept=".pdf,.png,.jpg,.jpeg">
                                        </div>
                                    </div>
                                    <button type="submit" class="button" id="save-changes-btn-{{ task.id }}">Save Changes</button>
                                </form>
                                <form method="post" action="{{ url_for('raw_edit_task', task_id=task.id) }}" id="raw-edit-form-{{ task.id }}" style="display:none; margin-top: 20px;">
                                    <button type="button" class="button" style="background:#eee;color:#333;margin-bottom:1rem;" onclick="stopRawEditingTask({{ task.id }})" id="stop-raw-edit-btn-{{ task.id }}">Stop Raw Edit</button>
                                    <div class="form-group">
                                        <label for="raw_rubric_content_{{ task.id }}">Rubric (raw Markdown, no LLM formatting)</label>
                                        <textarea id="raw_rubric_content_{{ task.id }}" name="raw_rubric_content">{{ task.rubrics[0].content if task.rubrics else '' }}</textarea>
                                    </div>
                                    <button type="submit" class="button" id="save-raw-changes-btn-{{ task.id }}">Save Raw Changes</button>
                                </form>
                            </div>
                        </div>
                    {% endfor %}
                </div>
                <div style="display: flex; justify-content: center; margin: 1.5rem 0; gap: 1rem;">
                    {% if tasks_paginated.has_prev %}
                        <a href="?page={{ tasks_paginated.prev_num }}" class="button" style="background:#eee;color:#333;">&laquo; Previous</a>
                    {% endif %}
                    <span style="align-self: center; color: #888;">Page {{ tasks_paginated.page }} of {{ tasks_paginated.pages }}</span>
                    {% if tasks_paginated.has_next %}
                        <a href="?page={{ tasks_paginated.next_num }}" class="button" style="background:#eee;color:#333;">Next &raquo;</a>
                    {% endif %}
                </div>
            {% else %}
                <p>No tasks created yet. Use the form below to add one.</p>
            {% endif %}
            <div class="form-section">
                <h2>Create New Task</h2>
                <form method="post" action="{{ url_for('create_task') }}" enctype="multipart/form-data">
                    <div class="form-group">
                        <label for="task_name">Task Name</label>
                        <input type="text" id="task_name" name="task_name" required>
                    </div>
                    <div class="tabs">
                        <div class="tab-link active" onclick="openTab(event, 'text-rubric')">Paste Rubric Text</div>
                        <div class="tab-link" onclick="openTab(event, 'file-rubric')">Upload Rubric File</div>
                    </div>
                    <div id="text-rubric" class="tab-content active">
                        <div class="form-group">
                            <label for="rubric_content">Rubric (paste here)</label>
                            <textarea id="rubric_content" name="rubric_content"></textarea>
                        </div>
                    </div>
                    <div id="file-rubric" class="tab-content">
                        <div class="form-group">
                            <label for="rubric_file">Rubric File (PDF, PNG, JPG)</label>
                            <input type="file" id="rubric_file" name="rubric_file" accept=".pdf,.png,.jpg,.jpeg">
                        </div>
                    </div>
                    <button type="submit" class="button" id="create-task-btn">Create Task</button>
                </form>
            </div>
        </div>
        <div id="no-work-modal" style="display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;align-items:center;justify-content:center;background:rgba(0,0,0,0.4);z-index:10000;">
            <div style="background:white;padding:2rem 2.5rem;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.2);text-align:center;max-width:350px;">
                <h3 style="margin-top:0;">No work was provided</h3>
                <p>Are you sure you want to continue? This will save a placeholder for this submission.</p>
                <div style="margin-top:1.5rem;">
                    <button type="button" class="button" id="no-work-continue-btn">Continue</button>
                    <button type="button" class="button" style="background:#eee;color:#333;margin-left:10px;" id="no-work-cancel-btn">Cancel</button>
                </div>
            </div>
        </div>
        <script>
            // Accordion for task list
            document.querySelectorAll('.task-header').forEach(header => {
                header.addEventListener('click', () => {
                    const content = header.nextElementSibling;
                    content.style.display = content.style.display === 'block' ? 'none' : 'block';
                });
            });
            // Tabs for form (global and per-task)
            function openTab(evt, tabName) {
                let i, tabcontent, tablinks, parent;
                if(tabName.includes('-')) {
                    // Per-task tab
                    parent = evt.currentTarget.closest('.rubric-content');
                    tabcontent = parent.getElementsByClassName('tab-content');
                    tablinks = parent.getElementsByClassName('tab-link');
                } else {
                    tabcontent = document.getElementsByClassName('tab-content');
                    tablinks = document.getElementsByClassName('tab-link');
                }
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            // Show the edit form for a rubric (from inside the dropdown)
            function showEditForm(taskId) {
                // Hide all edit forms and show all markdowns
                document.querySelectorAll('[id^="edit-form-"]').forEach(f => f.style.display = 'none');
                document.querySelectorAll('[id^="raw-edit-form-"]').forEach(f => f.style.display = 'none');
                document.querySelectorAll('[id^="rubric-markdown-"]').forEach(m => m.style.display = 'block');
                // Show the edit form and hide the markdown for this task
                document.getElementById('edit-form-' + taskId).style.display = 'block';
                document.getElementById('rubric-markdown-' + taskId).style.display = 'none';
                // Change the Edit Task button to Stop Editing Task
                var editBtn = document.getElementById('edit-task-btn-' + taskId);
                if (editBtn) {
                    editBtn.style.display = 'none';
                }
                var rawEditBtn = document.getElementById('raw-edit-btn-' + taskId);
                if (rawEditBtn) {
                    rawEditBtn.style.display = '';
                }
            }
            // Stop editing and return to view mode
            function stopEditingTask(taskId) {
                document.getElementById('edit-form-' + taskId).style.display = 'none';
                document.getElementById('rubric-markdown-' + taskId).style.display = 'block';
                var editBtn = document.getElementById('edit-task-btn-' + taskId);
                if (editBtn) {
                    editBtn.style.display = '';
                }
            }
            // Show the raw edit form for a rubric (bypasses LLM)
            function showRawEditForm(taskId) {
                // Hide all edit forms and show all markdowns
                document.querySelectorAll('[id^="edit-form-"]').forEach(f => f.style.display = 'none');
                document.querySelectorAll('[id^="raw-edit-form-"]').forEach(f => f.style.display = 'none');
                document.querySelectorAll('[id^="rubric-markdown-"]').forEach(m => m.style.display = 'block');
                // Show the raw edit form and hide the markdown for this task
                document.getElementById('raw-edit-form-' + taskId).style.display = 'block';
                document.getElementById('rubric-markdown-' + taskId).style.display = 'none';
                // Hide the Edit Task button while raw editing
                var editBtn = document.getElementById('edit-task-btn-' + taskId);
                if (editBtn) {
                    editBtn.style.display = '';
                }
                var rawEditBtn = document.getElementById('raw-edit-btn-' + taskId);
                if (rawEditBtn) {
                    rawEditBtn.style.display = 'none';
                }
            }
            // Stop raw editing and return to view mode
            function stopRawEditingTask(taskId) {
                document.getElementById('raw-edit-form-' + taskId).style.display = 'none';
                document.getElementById('rubric-markdown-' + taskId).style.display = 'block';
                var rawEditBtn = document.getElementById('raw-edit-btn-' + taskId);
                if (rawEditBtn) {
                    rawEditBtn.style.display = '';
                }
            }
            // Depress and update text for Create Task and Save Changes buttons
            document.addEventListener('DOMContentLoaded', function() {
                var createBtn = document.getElementById('create-task-btn');
                if (createBtn) {
                    createBtn.addEventListener('click', function() {
                        createBtn.disabled = true;
                        createBtn.textContent = 'Creating Task...';
                        createBtn.form.submit();
                    });
                }
                document.querySelectorAll('[id^="save-changes-btn-"]').forEach(function(saveBtn) {
                    saveBtn.addEventListener('click', function(e) {
                        saveBtn.disabled = true;
                        saveBtn.textContent = 'Saving Changes...';
                        saveBtn.form.submit();
                    });
                });
            });
            // Tabs for form (global and per-task)
            function showAddWorkForm(taskId) {
                document.getElementById('add-work-form-' + taskId).style.display = 'block';
                document.getElementById('add-work-btn-' + taskId).style.display = 'none';
            }
            function hideAddWorkForm(taskId) {
                document.getElementById('add-work-form-' + taskId).style.display = 'none';
                document.getElementById('add-work-btn-' + taskId).style.display = '';
            }
            // Camera logic for each Add Work form
            function setupCameraForTask(taskId) {
                const startCameraButton = document.getElementById('start-camera-btn-' + taskId);
                const captureButton = document.getElementById('capture-photo-btn-' + taskId);
                const cameraView = document.getElementById('camera-view-' + taskId);
                const video = document.getElementById('camera-feed-' + taskId);
                const previewDiv = document.getElementById('camera-preview-' + taskId);
                const capturedImg = document.getElementById('captured-image-' + taskId);
                const retakeBtn = document.getElementById('retake-photo-btn-' + taskId);
                let stream;
                let capturedBlob = null;
                if (!startCameraButton || !captureButton || !cameraView || !video || !previewDiv || !capturedImg || !retakeBtn) return;
                startCameraButton.addEventListener('click', async () => {
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
                        video.srcObject = stream;
                        await video.play();
                        startCameraButton.style.display = 'none';
                        cameraView.style.display = 'block';
                    } catch (err) {
                        alert('Could not access the camera. Please ensure you have given permission and are using a secure connection (https).');
                    }
                });
                captureButton.addEventListener('click', () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    canvas.toBlob(blob => {
                        capturedBlob = blob;
                        const url = URL.createObjectURL(blob);
                        capturedImg.src = url;
                        previewDiv.style.display = 'block';
                        video.style.display = 'none';
                        captureButton.style.display = 'none';
                    }, 'image/jpeg');
                });
                retakeBtn.addEventListener('click', () => {
                    previewDiv.style.display = 'none';
                    video.style.display = 'block';
                    captureButton.style.display = '';
                    capturedImg.src = '';
                    capturedBlob = null;
                });
                // Remove upload button logic; form submission will handle the upload if capturedBlob is present
                // On form submit, if a capturedBlob exists, append it as work_camera
                const form = document.getElementById('add-work-form-' + taskId).querySelector('form');
                form.addEventListener('submit', function(e) {
                    if (capturedBlob) {
                        // Remove any previous file/text
                        const formData = new FormData(form);
                        formData.delete('work_file');
                        formData.delete('work_text');
                        formData.append('work_camera', capturedBlob, 'capture.jpg');
                        e.preventDefault();
                        fetch(form.action, {
                            method: 'POST',
                            body: formData
                        }).then(response => {
                            if (response.redirected) {
                                window.location.href = response.url;
                            } else {
                                window.location.reload();
                            }
                        }).catch(() => window.location.reload());
                    }
                });
            }
            // Show/hide new student input
            function toggleNewStudentInput(taskId) {
                var select = document.getElementById('student_' + taskId);
                var input = document.getElementById('new-student-input-' + taskId);
                if (select.value === 'new') {
                    input.style.display = '';
                } else {
                    input.style.display = 'none';
                }
            }
            // Setup camera for all tasks on DOMContentLoaded
            document.addEventListener('DOMContentLoaded', function() {
                {% for task in tasks %}
                    setupCameraForTask({{ task.id }});
                {% endfor %}
            });
            // --- No Work Provided Modal Logic ---
            document.querySelectorAll('[id^="add-work-form-"]').forEach(function(formDiv) {
                var form = formDiv.querySelector('form');
                if (!form) return;
                form.addEventListener('submit', function(e) {
                    // Only check for work if not already forcing save
                    if (form.querySelector('[name="force_save"]') && form.querySelector('[name="force_save"]').value === '1') return;
                    var fileInput = form.querySelector('input[name="work_file"]');
                    var textInput = form.querySelector('textarea[name="work_text"]');
                    var cameraInput = form.querySelector('input[name="work_camera"]');
                    var hasFile = fileInput && fileInput.files && fileInput.files.length > 0;
                    var hasText = textInput && textInput.value.trim().length > 0;
                    var hasCamera = cameraInput && cameraInput.files && cameraInput.files.length > 0;
                    if (!hasFile && !hasText && !hasCamera) {
                        e.preventDefault();
                        window._noWorkForm = form;
                        document.getElementById('no-work-modal').style.display = 'flex';
                    }
                });
            });
            var continueBtn = document.getElementById('no-work-continue-btn');
            var cancelBtn = document.getElementById('no-work-cancel-btn');
            if (continueBtn && cancelBtn) {
                continueBtn.onclick = function() {
                    if (window._noWorkForm) {
                        var hidden = document.createElement('input');
                        hidden.type = 'hidden';
                        hidden.name = 'force_save';
                        hidden.value = '1';
                        window._noWorkForm.appendChild(hidden);
                        window._noWorkForm.submit();
                    }
                    document.getElementById('no-work-modal').style.display = 'none';
                };
                cancelBtn.onclick = function() {
                    document.getElementById('no-work-modal').style.display = 'none';
                };
            }
        </script>
    </body>
    </html>
    """, tasks=tasks, tasks_paginated=tasks_paginated, students=students)

@app.route('/create_task', methods=['POST'])
def create_task():
    """Handles the creation of a new task and its rubric."""
    task_name = request.form.get('task_name')
    rubric_content = request.form.get('rubric_content')
    rubric_file = request.files.get('rubric_file')
    raw_rubric_text = ""

    if not task_name:
        # Handle error: task name is required
        return "Task name is required.", 400

    if rubric_content:
        raw_rubric_text = rubric_content
    elif rubric_file and allowed_file(rubric_file.filename):
        # Save the uploaded file temporarily to process it
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{secure_filename(rubric_file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        rubric_file.save(file_path)
        
        # Extract text using OCR
        if filename.lower().endswith('.pdf'):
            # For PDFs, convert the first page to an image first
            images, err = convert_pdf_to_images(file_path)
            if err or not images:
                return "Failed to convert PDF rubric.", 500
            ocr_target_path = images[0]
        else:
            ocr_target_path = file_path
        
        annotation = detect_document_text(ocr_target_path)
        if annotation:
            raw_rubric_text = annotation.text
    
    if not raw_rubric_text.strip():
        # Handle error: rubric content is required
        return "Rubric content is required, either via text or a valid file.", 400

    # Use the agent to format the rubric
    formatter = RubricFormattingAgent()
    formatted_rubric = formatter.run(raw_rubric_text)

    # Save to database
    new_task = Task(name=task_name)
    db.session.add(new_task)
    db.session.commit()

    new_rubric = Rubric(content=formatted_rubric, task_id=new_task.id)
    db.session.add(new_rubric)
    db.session.commit()

    return redirect(url_for('index'))

@app.route('/edit_task/<int:task_id>', methods=['POST'])
def edit_task(task_id):
    """Handles editing an existing task's name and rubric."""
    task = Task.query.get_or_404(task_id)
    rubric_content = request.form.get('rubric_content')
    rubric_file = request.files.get('rubric_file')
    raw_rubric_text = ""
    if 'task_name' in request.form:
        task.name = request.form['task_name']
    if rubric_content:
        raw_rubric_text = rubric_content
    elif rubric_file and allowed_file(rubric_file.filename):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{secure_filename(rubric_file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        rubric_file.save(file_path)
        if filename.lower().endswith('.pdf'):
            images, err = convert_pdf_to_images(file_path)
            if err or not images:
                return "Failed to convert PDF rubric.", 500
            ocr_target_path = images[0]
        else:
            ocr_target_path = file_path
        annotation = detect_document_text(ocr_target_path)
        if annotation:
            raw_rubric_text = annotation.text
    if raw_rubric_text.strip():
        formatter = RubricFormattingAgent()
        formatted_rubric = formatter.run(raw_rubric_text)
        # Update or create rubric
        if task.rubrics:
            task.rubrics[0].content = formatted_rubric
        else:
            new_rubric = Rubric(content=formatted_rubric, task_id=task.id)
            db.session.add(new_rubric)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/delete_task/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    """Handles deleting a task and its rubrics."""
    task = Task.query.get_or_404(task_id)
    for rubric in task.rubrics:
        db.session.delete(rubric)
    db.session.delete(task)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/raw_edit_task/<int:task_id>', methods=['POST'])
def raw_edit_task(task_id):
    """Handles direct editing of a rubric's Markdown, bypassing the LLM."""
    task = Task.query.get_or_404(task_id)
    raw_rubric_content = request.form.get('raw_rubric_content')
    if raw_rubric_content is not None:
        if task.rubrics:
            task.rubrics[0].content = raw_rubric_content
        else:
            new_rubric = Rubric(content=raw_rubric_content, task_id=task.id)
            db.session.add(new_rubric)
        db.session.commit()
    return redirect(url_for('index'))

@app.route('/add_work_submission/<int:task_id>', methods=['POST'])
def add_work_submission(task_id):
    import time
    task = Task.query.get_or_404(task_id)
    student_id = request.form.get('student_id')
    new_student_name = request.form.get('new_student_name')
    work_file = request.files.get('work_file')
    work_text = request.form.get('work_text')
    work_camera = request.files.get('work_camera')
    student = None
    if new_student_name and new_student_name.strip():
        student = Student(name=new_student_name.strip())
        db.session.add(student)
        db.session.commit()
    elif student_id:
        student = Student.query.get(student_id)
    if not student:
        return "Student is required.", 400
    filename = None
    filetype = None
    raw_text = None
    cache_dir = None
    if work_file and work_file.filename:
        filename = secure_filename(work_file.filename)
        filetype = filename.split('.')[-1].lower()
        # Generate unique cache dir: timestamp + name (no extension)
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        base_name = os.path.splitext(filename)[0]
        cache_dir = f'{timestamp}-{base_name}'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        work_file.save(file_path)
    elif work_camera and work_camera.filename:
        filename = secure_filename(work_camera.filename)
        filetype = filename.split('.')[-1].lower()
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        base_name = os.path.splitext(filename)[0]
        cache_dir = f'{timestamp}-{base_name}'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        work_camera.save(file_path)
    elif work_text and work_text.strip():
        raw_text = work_text.strip()
        filetype = 'txt'
    elif request.form.get('force_save') == '1':
        filetype = 'none'
        raw_text = '(no work provided)'
        filename = None
    else:
        return redirect(url_for('index'))
    submission = WorkSubmission(
        task_id=task.id,
        student_id=student.id,
        filename=filename,
        filetype=filetype,
        raw_text=raw_text,
        cache_dir=cache_dir
    )
    db.session.add(submission)
    db.session.commit()
    # --- One-time processing and caching for PDF/DOCX ---
    if filename and filetype in ['pdf', 'docx']:
        from app import process_image_pipeline, convert_pdf_to_images
        import tempfile, shutil
        import pythoncom
        submission_id = submission.id
        cache_path = os.path.join(app.config['UPLOAD_FOLDER'], cache_dir)
        os.makedirs(cache_path, exist_ok=True)
        pdf_path = None
        temp_dir = None
        if filetype == 'docx':
            from docx2pdf import convert
            temp_dir = tempfile.mkdtemp()
            docx_path = file_path
            pdf_path = os.path.join(temp_dir, os.path.splitext(filename)[0] + '.pdf')
            pythoncom.CoInitialize()
            try:
                convert(docx_path, pdf_path)
            finally:
                pythoncom.CoUninitialize()
        else:
            pdf_path = file_path
        image_paths, error = convert_pdf_to_images(pdf_path)
        manifest = []
        for i, img_path in enumerate(image_paths):
            img_filename = f'page-{i+1}.png'
            cached_img_path = os.path.join(cache_path, img_filename)
            shutil.copy(img_path, cached_img_path)
            results, err = process_image_pipeline(img_path)
            word_data = results.get('word_data', []) if results else []
            dims = results.get('original_dimensions', {}) if results else {}
            manifest.append({
                'img': img_filename,
                'word_data': word_data,
                'width': dims.get('width'),
                'height': dims.get('height')
            })
        with open(os.path.join(cache_path, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f)
        if temp_dir:
            shutil.rmtree(temp_dir)
    return redirect(url_for('index'))

@app.route('/delete_work_submission/<int:submission_id>', methods=['POST'])
def delete_work_submission(submission_id):
    submission = WorkSubmission.query.get_or_404(submission_id)
    db.session.delete(submission)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/review_work/<int:submission_id>')
def review_work(submission_id):
    from app import process_image_pipeline, convert_pdf_to_images  # Always import at the top of the function
    import tempfile
    import shutil
    import json
    submission = WorkSubmission.query.get_or_404(submission_id)
    task = submission.task
    rubric = task.rubrics[0] if task.rubrics else None
    criteria = []
    analysis = None
    status = 'Initializing...'
    refined_text = None
    filetype = submission.filetype or ''
    file_url = None
    text_content = None
    page_images = []  # List of dicts: {url, word_data, width, height}
    if rubric:
        extractor = RubricCriterionExtractorAgent()
        criteria = extractor.run(rubric.content)
    # Try to load manifest cache for any filetype with a filename or cache_dir
    manifest_loaded = False
    cache_dir = getattr(submission, 'cache_dir', None)
    if cache_dir:
        cache_path = os.path.join(app.config['UPLOAD_FOLDER'], cache_dir)
        manifest_path = os.path.join(cache_path, 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            for i, page in enumerate(manifest):
                img_url = url_for('pure_ocr.serve_upload', filename=f'{cache_dir}/' + page['img'])
                page_images.append({
                    'url': img_url,
                    'word_data': page['word_data'],
                    'width': page['width'],
                    'height': page['height']
                })
            status = 'Document loaded from cache.'
            manifest_loaded = True
    # If not loaded from manifest, fall back to previous logic
    if not manifest_loaded:
        print('[WARNING] No cache_dir or manifest found for this submission. Falling back to old logic.')
        if submission.raw_text and submission.filetype == 'txt':
            refined_text = submission.raw_text
            status = 'Marking text...'
            analysis = None
            status = 'Complete'
        elif submission.filename:
            if filetype in ['png', 'jpg', 'jpeg', 'gif']:
                file_url = url_for('pure_ocr.serve_upload', filename=submission.filename)
                results, err = process_image_pipeline(os.path.join(app.config['UPLOAD_FOLDER'], submission.filename))
                if err:
                    word_data = []
                    width = height = None
                else:
                    word_data = results.get('word_data', [])
                    dims = results.get('original_dimensions', {})
                    width = dims.get('width')
                    height = dims.get('height')
                page_images.append({
                    'url': file_url,
                    'word_data': word_data,
                    'width': width,
                    'height': height
                })
            else:
                file_url = url_for('pure_ocr.serve_upload', filename=submission.filename)
        else:
            status = 'No work uploaded.'
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Review Student Work</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; }
            .review-layout { display: flex; height: 100vh; }
            .review-left { flex: 2.5; overflow-y: auto; background: #fff; padding: 2rem; border-right: 1px solid #eee; }
            .review-right { flex: 1; overflow-y: auto; background: #fafbfc; padding: 2rem; position: sticky; top: 0; height: 100vh; }
            .rubric-criterion { background: #e0e0e0; border-radius: 8px; margin-bottom: 1.5rem; padding: 1.2rem 1.5rem; color: #333; box-shadow: 0 2px 6px rgba(0,0,0,0.04); transition: background 0.2s; }
            .rubric-criterion .mark { font-weight: bold; font-size: 1.2em; }
            .rubric-criterion .reasoning, .rubric-criterion .evidence { margin-top: 0.5em; font-size: 0.98em; }
            .rubric-criterion .criterion-title { font-weight: bold; font-size: 1.1em; margin-bottom: 0.3em; }
            .status-box { position: fixed; top: 2rem; right: 2rem; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 1rem 1.5rem; z-index: 1000; font-size: 1.1em; color: #333; display: flex; align-items: center; gap: 0.7em; }
            .spinner { width: 22px; height: 22px; border: 3px solid #eee; border-top: 3px solid #007bff; border-radius: 50%; animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .ocr-page-img-wrapper { position: relative; margin-bottom: 2.5em; }
            .ocr-page-img { width: 100%; max-width: none; display: block; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
            .highlight { position: absolute; border: 2px solid #00b894; background: rgba(0,184,148,0.18); pointer-events: none; border-radius: 2px; }
            .highlight.green { background: rgba(0,255,0,0.18); border-color: #00b894; }
            .highlight.yellow { background: rgba(255,255,0,0.18); border-color: #cca300; }
            .search-section { text-align: left; margin-bottom: 2rem; }
            .search-section input { padding: 10px; font-size: 16px; width: 300px; border: 1px solid #ccc; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="status-box" id="status-box">
            <span id="status-text">{{ status }}</span>
            <span id="status-spinner" class="spinner" {% if status == 'Complete' %}style="display:none;"{% endif %}></span>
            <button id="force-remark-btn" style="margin-left:1em;padding:0.4em 1em;font-size:1em;background:#f5f5f5;border:1px solid #bbb;border-radius:5px;cursor:pointer;">Force Remark</button>
        </div>
        <div class="review-layout">
            <div class="review-left">
                <a href="{{ url_for('index') }}" style="display:inline-block;margin-bottom:1em;color:#007bff;text-decoration:none;font-weight:500;font-size:1.05em;">&larr; Back</a>
                <h2>{{ submission.task.name }}</h2>
                <div style="color:#666;font-size:1.1em;margin-bottom:1.5em;">
                    {{ submission.student.name }} - {{ submission.upload_time.strftime('%d-%m-%Y %H:%M') }}
                </div>
                <div class="search-section">
                    <input type="text" id="search-bar" placeholder="Type to highlight words...">
                </div>
                {% if page_images and page_images|length > 0 %}
                    {% for page in page_images %}
                        <div class="ocr-page-img-wrapper" style="position:relative;">
                            <img src="{{ page.url }}" class="ocr-page-img" id="ocr-img-{{ loop.index0 }}" data-page-index="{{ loop.index0 }}" data-width="{{ page.width }}" data-height="{{ page.height }}" />
                            <div class="highlight-overlay" id="highlight-overlay-{{ loop.index0 }}" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;"></div>
                            <script type="application/json" id="word-data-{{ loop.index0 }}">{{ page.word_data|tojson }}</script>
                        </div>
                    {% endfor %}
                {% elif file_url %}
                    <img src="{{ file_url }}" style="width:100%;max-width:none;border-radius:8px;" />
                {% elif refined_text %}
                    <pre style="background:#f7f7f7;padding:1.5rem;border-radius:8px;">{{ refined_text }}</pre>
                {% else %}
                    <p style="color:#888;">No work uploaded.</p>
                {% endif %}
            </div>
            <div class="review-right" id="rubric-right">
                <h2>Rubric</h2>
                {% for criterion in criteria %}
                    <div class="rubric-criterion" style="background: hsl({{ loop.index0 * 60 }}, 40%, 85%);">
                        <div class="criterion-title">{{ criterion }}</div>
                        <div class="mark">Suggested Mark: <b id="mark-{{ loop.index0 }}">{{ analysis[criterion]['suggested_mark'] if analysis and criterion in analysis and analysis[criterion].get('suggested_mark') != None else '—' }}</b></div>
                        <div class="reasoning"><b>Reasoning:</b> <span id="reasoning-{{ loop.index0 }}" style="color:#444;">{{ analysis[criterion]['reasoning'] if analysis and criterion in analysis and analysis[criterion].get('reasoning') else '(not yet analyzed)' }}</span></div>
                        <div class="evidence"><b>Evidence + Justification:</b> <span id="evidence-{{ loop.index0 }}" style="color:#444;">{{ analysis[criterion]['evidence_justification'] if analysis and criterion in analysis and analysis[criterion].get('evidence_justification') else '(not yet analyzed)' }}</span></div>
                    </div>
                {% else %}
                    <p style="color:#888;">No rubric found.</p>
                {% endfor %}
            </div>
        </div>
        <script>
        // --- Highlighting Logic for Evidence Words ---
        let currentCriterionIdx = null;
        let analysis = null;
        let criteria = {{ criteria|tojson }};
        let rubricColors = criteria.map((_, idx) => `hsl(${idx * 60}, 60%, 70%)`);

        function clearAllRubricOutlines() {
            document.querySelectorAll('.rubric-criterion').forEach(el => {
                el.style.outline = '';
                el.style.zIndex = '';
            });
        }

        function clearAllHighlights() {
            document.querySelectorAll('.highlight-overlay').forEach(overlay => {
                overlay.innerHTML = '';
            });
        }

        function showHighlightsForCriterion(idx) {
            clearAllHighlights();
            if (!analysis) return;
            let highlights = analysis[criteria[idx]] && analysis[criteria[idx]].highlights ? analysis[criteria[idx]].highlights : [];
            let color = rubricColors[idx];
            // For each page
            for (let pageIdx = 0; pageIdx < {{ page_images|length }}; pageIdx++) {
                const wordData = JSON.parse(document.getElementById('word-data-' + pageIdx).textContent);
                const img = document.getElementById('ocr-img-' + pageIdx);
                const overlay = document.getElementById('highlight-overlay-' + pageIdx);
                overlay.innerHTML = '';
                const imgWidth = img.naturalWidth;
                const imgHeight = img.naturalHeight;
                wordData.forEach((word, widx) => {
                    if (highlights.includes(word.text) && word.vertices && word.vertices.length === 4) {
                        const x = word.vertices[0].x / imgWidth * 100;
                        const y = word.vertices[0].y / imgHeight * 100;
                        const w = (word.vertices[1].x - word.vertices[0].x) / imgWidth * 100;
                        const h = (word.vertices[2].y - word.vertices[1].y) / imgHeight * 100;
                        const div = document.createElement('div');
                        div.className = 'highlight';
                        div.style.position = 'absolute';
                        div.style.left = x + '%';
                        div.style.top = y + '%';
                        div.style.width = w + '%';
                        div.style.height = h + '%';
                        div.style.background = color.replace(')', ', 0.15)').replace('hsl', 'hsla');
                        div.style.borderColor = color;
                        overlay.appendChild(div);
                    }
                });
            }
        }

        function selectCriterion(idx) {
            clearAllRubricOutlines();
            clearAllHighlights();
            // Outline the selected rubric
            let rubricDivs = document.querySelectorAll('.rubric-criterion');
            rubricDivs[idx].style.outline = `3px solid ${rubricColors[idx]}`;
            rubricDivs[idx].style.zIndex = 2;
            showHighlightsForCriterion(idx);
            currentCriterionIdx = idx;
        }

        // Add click listeners to rubric containers
        document.addEventListener('DOMContentLoaded', function() {
            let rubricDivs = document.querySelectorAll('.rubric-criterion');
            rubricDivs.forEach((div, idx) => {
                div.addEventListener('click', function() {
                    selectCriterion(idx);
                });
            });
        });

        // --- Marking/Analysis Update Logic ---
        let statusText = document.getElementById('status-text');
        let statusSpinner = document.getElementById('status-spinner');
        let rubricRight = document.getElementById('rubric-right');
        let submissionId = {{ submission.id }};
        function updateRubricContainers(newAnalysis) {
            analysis = newAnalysis;
            criteria.forEach(function(criterion, idx) {
                let mark = analysis[criterion] && analysis[criterion]['suggested_mark'] !== undefined && analysis[criterion]['suggested_mark'] !== null ? analysis[criterion]['suggested_mark'] : '…';
                let reasoning = analysis[criterion] && analysis[criterion]['reasoning'] ? analysis[criterion]['reasoning'] : '(not yet analyzed)';
                let evidence = analysis[criterion] && analysis[criterion]['evidence_justification'] ? analysis[criterion]['evidence_justification'] : '(not yet analyzed)';
                document.getElementById('mark-' + idx).textContent = mark;
                document.getElementById('reasoning-' + idx).textContent = reasoning;
                document.getElementById('evidence-' + idx).textContent = evidence;
            });
            // If a criterion is already selected, refresh its highlights
            if (currentCriterionIdx !== null) {
                showHighlightsForCriterion(currentCriterionIdx);
            }
        }
        function runMarkingOnce(force=false) {
            statusText.textContent = 'Processing...';
            statusSpinner.style.display = '';
            let url = '/api/mark_work/' + submissionId;
            if (force) url += '?force=1';
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'complete') {
                        if (data.refined_text) {
                            let left = document.querySelector('.review-left');
                            if (!left.querySelector('pre')) {
                                let pre = document.createElement('pre');
                                pre.style.background = '#f7f7f7';
                                pre.style.padding = '1.5rem';
                                pre.style.borderRadius = '8px';
                                pre.textContent = data.refined_text;
                                left.appendChild(pre);
                            }
                        }
                        if (data.analysis) {
                            updateRubricContainers(data.analysis);
                        }
                        statusText.textContent = 'Complete';
                        statusSpinner.style.display = 'none';
                    } else if (data.status === 'error') {
                        statusText.textContent = 'Error: ' + (data.error || 'Unknown error');
                        statusSpinner.style.display = 'none';
                        let rubricRight = document.getElementById('rubric-right');
                        rubricRight.innerHTML = '<div style="color:#d9534f;font-weight:bold;">' + (data.error || 'Unknown error') + '</div>' + (data.raw_response ? '<pre style="margin-top:1em;max-width:100%;overflow-x:auto;">' + data.raw_response + '</pre>' : '');
                    } else {
                        statusText.textContent = data.status || 'Processing...';
                    }
                })
                .catch((err) => {
                    statusText.textContent = 'Error during marking.';
                    statusSpinner.style.display = 'none';
                    let rubricRight = document.getElementById('rubric-right');
                    rubricRight.innerHTML = '<div style="color:#d9534f;font-weight:bold;">Error during marking.</div>';
                });
        }
        // Run marking once on page load
        runMarkingOnce();
        document.getElementById('force-remark-btn').addEventListener('click', function() {
            runMarkingOnce(true);
        });
        </script>
    </body>
    </html>
    """, file_url=file_url, filetype=filetype, refined_text=refined_text, criteria=criteria, analysis=analysis, status=status, submission=submission, page_images=page_images)

@app.route('/api/mark_work/<int:submission_id>', methods=['GET'])
def api_mark_work(submission_id):
    from app import process_image_pipeline, convert_pdf_to_images
    import tempfile
    import shutil
    import traceback
    import json
    submission = WorkSubmission.query.get_or_404(submission_id)
    task = submission.task
    rubric = task.rubrics[0] if task.rubrics else None
    if not rubric:
        return jsonify({'error': 'No rubric found for this task.'}), 400
    # Step 1: Get text (OCR if needed)
    refined_text = None
    filetype = submission.filetype or ''
    try:
        if submission.raw_text and submission.filetype == 'txt':
            print('[DEBUG] Using raw text for marking.')
            refined_text = submission.raw_text
        elif submission.filename:
            job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, submission.filename))
            ocr_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}.txt')
            if os.path.exists(ocr_result_path):
                print(f'[DEBUG] Using cached OCR result: {ocr_result_path}')
                with open(ocr_result_path, 'r', encoding='utf-8') as f:
                    refined_text = f.read()
            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], submission.filename)
                cache_dir = getattr(submission, 'cache_dir', None)
                if cache_dir:
                    cache_path = os.path.join(app.config['UPLOAD_FOLDER'], cache_dir)
                    manifest_path = os.path.join(cache_path, 'manifest.json')
                    if os.path.exists(manifest_path):
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                        all_refined_texts = []
                        for page in manifest:
                            page_text = ' '.join([w['text'] for w in page['word_data']])
                            all_refined_texts.append(page_text)
                        refined_text = '\n\n'.join(all_refined_texts)
                        print('[DEBUG] Loaded refined text from manifest cache.')
                    else:
                        print('[DEBUG] No manifest cache found, fallback to processing.')
                        return jsonify({'status': 'error', 'error': 'No manifest cache found for this submission.'}), 500
                else:
                    print(f'[DEBUG] Running OCR pipeline on image: {file_path}')
                    results, error = process_image_pipeline(file_path)
                    if error:
                        print(f'[ERROR] OCR pipeline failed: {error}')
                        return jsonify({'status': 'error', 'error': error}), 500
                    refined_text = results['refined_text']
                with open(ocr_result_path, 'w', encoding='utf-8') as f:
                    f.write(refined_text)
                    print(f'[DEBUG] Wrote OCR result to cache: {ocr_result_path}')
        else:
            print('[ERROR] No work uploaded.')
            return jsonify({'status': 'error', 'error': 'No work uploaded.'}), 400
        # Extract criteria
        extractor = RubricCriterionExtractorAgent()
        criteria = extractor.run(rubric.content)
        print(f'[DEBUG] Extracted criteria: {criteria}')
        # Synchronous marking: run if not already done
        key = get_marking_result_key(submission_id)
        with shelve.open(MARKING_RESULTS_DB) as db:
            if not request.args.get('force', '0') == '1' and key in db and all(c in db[key] and db[key][c].get("status") == "complete" for c in criteria):
                print('[DEBUG] Using cached marking result.')
                result = db[key]
            else:
                print('[DEBUG] Running marking agents... (force_remark=' + str(request.args.get('force', '0') == '1') + ')')
                result = {c: {"suggested_mark": None, "reasoning": None, "evidence_justification": None, "status": "processing", "highlights": []} for c in criteria}
                db[key] = result
                main_agent = MainMarkerAgent()
                main_output = main_agent.run(rubric.content, refined_text)
                print(f'[DEBUG] Main marker output:\n{main_output}\n')
                highlight_agent = HighlightExtractionAgent()
                for c in criteria:
                    mark_agent = MarkExtractionAgent()
                    reasoning_agent = ReasoningExtractionAgent()
                    evidence_agent = EvidenceExtractionAgent()
                    mark = mark_agent.run(main_output, c)
                    print(f"[DEBUG] Mark for '{c}': {mark}")
                    reasoning = reasoning_agent.run(main_output, c)
                    print(f"[DEBUG] Reasoning for '{c}': {reasoning}")
                    evidence = evidence_agent.run(main_output, c)
                    print(f"[DEBUG] Evidence for '{c}': {evidence}")
                    highlights = highlight_agent.run(refined_text, evidence)
                    print(f"[DEBUG] Highlights for '{c}': {highlights}")
                    result[c] = {
                        "suggested_mark": mark,
                        "reasoning": reasoning,
                        "evidence_justification": evidence,
                        "highlights": highlights,
                        "status": "complete"
                    }
                db[key] = result
        print('[DEBUG] Marking complete. Returning result.')
        return jsonify({'status': 'complete', 'refined_text': refined_text, 'analysis': result})
    except Exception as e:
        print('[EXCEPTION] Exception during marking:', str(e))
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500

# In-memory or persistent store for progressive marking results (for demo, use shelve)
MARKING_RESULTS_DB = 'marking_results_db'

def get_marking_result_key(submission_id):
    return f"submission_{submission_id}"

def run_progressive_marking(submission_id, rubric_content, student_text, criteria):
    key = get_marking_result_key(submission_id)
    with shelve.open(MARKING_RESULTS_DB) as db:
        db[key] = {c: {"suggested_mark": None, "reasoning": None, "evidence_justification": None, "status": "processing"} for c in criteria}
    # Run main marker agent
    main_agent = MainMarkerAgent()
    main_output = main_agent.run(rubric_content, student_text)
    print(f"[DEBUG] Main marker output:\n{main_output}\n")
    # For each criterion, run subagents and update the store
    for c in criteria:
        mark_agent = MarkExtractionAgent()
        reasoning_agent = ReasoningExtractionAgent()
        evidence_agent = EvidenceExtractionAgent()
        mark = mark_agent.run(main_output, c)
        print(f"[DEBUG] Mark for '{c}': {mark}")
        with shelve.open(MARKING_RESULTS_DB) as db:
            db[key][c]["suggested_mark"] = mark
            db[key][c]["status"] = "mark_done"
        time.sleep(0.5)  # Simulate progressive fill
        reasoning = reasoning_agent.run(main_output, c)
        print(f"[DEBUG] Reasoning for '{c}': {reasoning}")
        with shelve.open(MARKING_RESULTS_DB) as db:
            db[key][c]["reasoning"] = reasoning
            db[key][c]["status"] = "reasoning_done"
        time.sleep(0.5)
        evidence = evidence_agent.run(main_output, c)
        print(f"[DEBUG] Evidence for '{c}': {evidence}")
        with shelve.open(MARKING_RESULTS_DB) as db:
            db[key][c]["evidence_justification"] = evidence
            db[key][c]["status"] = "complete"
        time.sleep(0.5)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Create the manager and shared dictionary here
    manager = multiprocessing.Manager()
    jobs = manager.dict()
    # Store the shared dictionary in the Flask app config
    app.config['JOBS'] = jobs

    # setup_phoenix_tracing() # Tracing can be noisy with multiprocessing, disabling for now.
    app.register_blueprint(pure_ocr_bp, url_prefix='/pureocr')
    app.run(debug=True, use_reloader=False) 