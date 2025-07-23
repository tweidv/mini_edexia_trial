import os
from google.cloud import vision
from PIL import Image, ImageDraw

# Set the environment variable for your service account key
# You will need to run this command in your terminal before running the script,
# or ensure your IDE runs it.
# $env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\tweid\Projects\mini_edexia_trial\credentials\gcloud_vision_key.json"

def detect_document_text(image_path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    full_text = response.full_text_annotation.text
    print(f"Full Text:\n{full_text}\n")

    # Iterate through paragraphs and words to get bounding box info
    print("Detected Blocks, Paragraphs, and Words:")
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print(f'\nBlock confidence: {block.confidence:.2f}')
            for paragraph in block.paragraphs:
                print(f'\tParagraph confidence: {paragraph.confidence:.2f}')
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    print(f'\t\tWord text: {word_text} (confidence: {word.confidence:.2f})')
                    print(f'\t\t\tBounding Box: {word.bounding_box.vertices}')

    return response.full_text_annotation

def highlight_text_on_image(image_path, text_annotation, output_path="highlighted_output.jpg"):
    """Highlights detected text on an image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Iterate through paragraphs and words to draw rectangles
    for page in text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    vertices = [(v.x, v.y) for v in word.bounding_box.vertices]
                    # Draw a semi-transparent rectangle
                    draw.polygon(vertices, outline='red', fill=(255, 0, 0, 50)) # RGBA: Red, semi-transparent

    img.save(output_path)
    print(f"\nHighlighted image saved to {output_path}")


if __name__ == "__main__":
    # Create a dummy image for testing, or use a real one
    # For a real test, put an image file (e.g., test_handwriting.png)
    # in your mini_edexia_trial directory
    test_image_path = "C:/Users/tweid/Projects/mini_edexia_trial/handwriting.jpg" # Replace with your actual image path

    # Create a dummy image for testing if it doesn't exist
    if not os.path.exists(test_image_path):
        print(f"'{test_image_path}' not found. Creating a dummy image for basic testing.")
        dummy_img = Image.new('RGB', (400, 200), color = (255, 255, 255))
        d = ImageDraw.Draw(dummy_img)
        d.text((50,50), "Hello, handwritten World!", fill=(0,0,0))
        d.text((50,100), "This is printed text.", fill=(0,0,0))
        dummy_img.save(test_image_path)
        print("Dummy image created. Please replace it with a real handwritten scan for better testing.")


    # Ensure you have your GOOGLE_APPLICATION_CREDENTIALS environment variable set
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        print("Please run: $env:GOOGLE_APPLICATION_CREDENTIALS=\"C:\\path\\to\\your\\mini_edexia_trial\\gcloud_vision_key.json\"")
        print("in your terminal before running this script.")
    else:
        print("Attempting OCR and highlighting...")
        full_annotation = detect_document_text(test_image_path)
        if full_annotation:
            highlight_text_on_image(test_image_path, full_annotation, "highlighted_test_image.jpg")
