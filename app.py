import os
from flask import Flask, send_file
from google.cloud import vision
from PIL import Image, ImageDraw
import io
from dotenv import load_dotenv
import base64

# Import the agent and the Phoenix setup function
from script_marker_agent.ocr_refinement_agent import OcrRefinementAgent, setup_phoenix_tracing

load_dotenv()  # Load environment variables from .env file

# Enable Phoenix Tracing for the web application
setup_phoenix_tracing()

app = Flask(__name__)

# NOTE: The GOOGLE_APPLICATION_CREDENTIALS is now loaded from the .env file
# You no longer need to set it in the terminal manually.

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

@app.route('/process-image')
def process_image():
    """
    Processes a test image, runs the refinement agent, and returns an HTML
    page with the highlighted version and the refined text.
    """
    # Hardcoded for now, as requested
    test_image_path = "C:/Users/tweid/Projects/mini_edexia_trial/handwriting.jpg" 

    if not os.path.exists(test_image_path):
        return "Test image not found.", 404

    # Check for both required credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return "GOOGLE_APPLICATION_CREDENTIALS not found in .env file.", 500
    if not os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY not found in .env file.", 500

    # 1. Run initial OCR to get raw text and annotation for highlighting
    full_annotation = detect_document_text(test_image_path)
    
    if full_annotation:
        raw_text = full_annotation.text

        # 2. Run the OcrRefinementAgent to get the clean text
        print("--- Initializing and running OcrRefinementAgent from web app---")
        refinement_agent = OcrRefinementAgent()
        refined_text = refinement_agent.run(
            raw_ocr_text=raw_text,
            original_image_path=test_image_path
        )
        print("--- Refined text received ---")

        # 3. Highlight the image using the original annotation data
        image_buffer = highlight_text_on_image_in_memory(test_image_path, full_annotation)
        
        # Encode image to base64 to embed in HTML
        img_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        # Return an HTML page with the image and the REFINED text
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OCR Result</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; display: flex; margin: 20px; gap: 30px; background-color: #f0f2f5; }}
                .container {{ flex: 1; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }}
                h2 {{ color: #1c1e21; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; line-height: 1.6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Highlighted Image</h2>
                <img src="data:image/jpeg;base64,{img_base64}" alt="Highlighted OCR Image" />
            </div>
            <div class="container">
                <h2>Refined Text</h2>
                <pre>{refined_text}</pre>
            </div>
        </body>
        </html>
        """
        return html_content
    else:
        return "Could not process the image with Google Cloud Vision.", 500

if __name__ == '__main__':
    # The .env file is loaded automatically at the start of the script.
    app.run(debug=True) 