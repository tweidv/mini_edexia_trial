import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import os
import io
import base64
import json
import sys

# --- Google Cloud Vision AI Imports ---
from google.cloud import vision_v1p3beta1 as vision

# --- Google Generative AI (Gemini) Imports ---
import google.generativeai as genai

# --- API Key Configuration (IMPORTANT: Replace with your actual keys or environment variables) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- Step 1: Initial Image Processing & OCR for Labels ---
def initial_image_processing_ocr(image_path):
    """
    Uses Google Cloud Vision AI to perform OCR and detect labels and their bounding boxes.
    """
    print(f"--- Step 1: Processing image {image_path} and performing OCR using Google Cloud Vision AI ---")
    sys.stdout.flush()
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        sys.stdout.flush()
        return None, None, None

    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    try:
        response = client.text_detection(image=image)
        texts = response.text_annotations

        ocr_capital_labels = []
        ocr_lowercase_greek_labels = []

        if texts:
            # The first text_annotation is usually the full detected text, skip it for individual labels
            for text in texts[1:]: # Iterate from the second element to get individual words/characters
                label = text.description
                vertices = text.bounding_poly.vertices
                # Convert vertices to [x_min, y_min, x_max, y_max] format
                x_coords = [v.x for v in vertices]
                y_coords = [v.y for v in vertices]
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

                # Simple heuristic to classify labels (can be refined)
                if len(label) == 1 and label.isupper():
                    ocr_capital_labels.append({'label': label, 'bbox': bbox})
                elif len(label) == 1 and label.islower():
                    ocr_lowercase_greek_labels.append({'label': label, 'bbox': bbox})
                # For now, any other detected text is also considered for lowercase/greek
                else:
                    ocr_lowercase_greek_labels.append({'label': label, 'bbox': bbox})

        print(f"OCR detected capital labels: {ocr_capital_labels}")
        print(f"OCR detected lowercase/Greek labels: {ocr_lowercase_greek_labels}")
        sys.stdout.flush()

    except Exception as e:
        print(f"Error during Google Cloud Vision AI OCR: {e}")
        sys.stdout.flush()
        return img, [], [] # Return empty lists on error

    print("--- Step 1 Complete ---")
    sys.stdout.flush()
    return img, ocr_capital_labels, ocr_lowercase_greek_labels

# --- Helper function for infinite line intersection ---
def line_line_intersection(line1, line2):
    """
    Calculates the intersection point of two infinite lines.
    Lines are given as [x1, y1, x2, y2].
    Returns (x, y) if they intersect, None otherwise (parallel or collinear).
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Lines are parallel or collinear

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den

    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)
    return (intersect_x, intersect_y)


# --- Step 2: Identify Diagram Vertices/Points & Shapes from Hand-drawn Diagram ---
def identify_diagram_features(img, ocr_capital_labels, easiest_anchor_points=[], expected_line_segments=[]):
    """
    Applies Canny, Hough Transform for lines and circles, and calculates intersections.
    Associates OCR-detected capital letters with nearby geometric points (intersections or circle centers).
    If no suitable geometric point is found, uses the darkest average intensity pixel in the vicinity as a fallback.
    Prioritizes 'easiest_anchor_points' for robustness.
    """
    print("--- Step 2: Identifying diagram vertices/points and shapes ---")
    sys.stdout.flush()
    
    # Pre-processing: Convert to grayscale and potentially invert colors
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if the image is mostly dark (drawing is light on dark background)
    average_intensity = np.mean(gray)
    if average_intensity < 128: # If average is less than half, assume dark background
        gray = cv2.bitwise_not(gray) # Invert colors
        print("Image colors inverted for better feature detection (drawing is now dark on light background).")
        sys.stdout.flush()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 40, 120, apertureSize=3)
    print("Canny edge detection applied.")
    sys.stdout.flush()

    # Hough Transform for Lines (tuned for stricter detection)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                            threshold=100,      # Higher threshold: requires more votes, fewer lines
                            minLineLength=50,   # Longer lines only
                            maxLineGap=20)      # Smaller gaps allowed in lines
    
    detected_raw_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_raw_lines.append(((x1, y1), (x2, y2)))
        print(f"Detected {len(detected_raw_lines)} raw lines.")
        sys.stdout.flush()
    else:
        print("No raw lines detected.")
        sys.stdout.flush()

    # Hough Circle Transform (keeping for completeness, but user indicated no circles expected)
    detected_circles = []
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30, # Increased minDist
                               param1=50, param2=70, minRadius=15, maxRadius=150) # Loosened param2, adjusted radius range
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            detected_circles.append((i[0], i[1], i[2])) # x, y, radius
        print(f"Detected {len(detected_circles)} circles.")
        sys.stdout.flush()
    else:
        print("No circles detected.")
        sys.stdout.flush()

    # Gather all geometric candidate points (line intersections + circle centers)
    all_geometric_candidates = []

    # Add line intersections directly from detected_raw_lines
    if detected_raw_lines: 
        for i in range(len(detected_raw_lines)):
            for j in range(i + 1, len(detected_raw_lines)): # Only check distinct lines/groups
                p1, p2 = detected_raw_lines[i]
                p3, p4 = detected_raw_lines[j]
                intersection = line_line_intersection([p1[0], p1[1], p2[0], p2[1]], [p3[0], p3[1], p4[0], p4[1]])
                if intersection:
                    all_geometric_candidates.append(np.array(intersection))
        # Remove duplicate or very close candidate vertices
        unique_line_intersections = []
        if all_geometric_candidates:
            all_geometric_candidates = np.array(all_geometric_candidates)
            for cand_v in all_geometric_candidates:
                is_unique = True
                for unique_v in unique_line_intersections:
                    if np.linalg.norm(cand_v - unique_v) < 15: # Increased tolerance for uniqueness
                        is_unique = False
                        break
                if is_unique:
                    unique_line_intersections.append(cand_v)
        all_geometric_candidates = unique_line_intersections
        print(f"Found {len(all_geometric_candidates)} candidate vertices from line intersections.")
        sys.stdout.flush()
    else:
        print("No candidate vertices from line intersections.")
        sys.stdout.flush()

    # Add circle centers to geometric candidates
    for cx, cy, r in detected_circles:
        all_geometric_candidates.append(np.array([cx, cy]))
    print(f"Total geometric candidates (intersections + circle centers): {len(all_geometric_candidates)}")
    sys.stdout.flush()


    # Refine points by associating OCR labels with the closest geometric candidate within a threshold
    refined_points = {}
    img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2) # For a more dynamic search radius

    # Create a list of OCR labels, prioritizing easy anchor points
    sorted_ocr_labels = sorted(ocr_capital_labels, key=lambda x: x['label'] not in easiest_anchor_points) # Puts easy anchors first

    if ocr_capital_labels:
        for ocr_label_info in sorted_ocr_labels:
            label = ocr_label_info['label']
            print(f"DEBUG: Processing label: {label} (Is easy anchor: {label in easiest_anchor_points})")
            sys.stdout.flush()
            bbox = ocr_label_info['bbox']
            bbox_center = np.array(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
            bbox_height = bbox[3] - bbox[1] # Height of the capital letter's bounding box

            # Define search radius: max of (3 times bbox height, a fixed minimum, and a percentage of image diagonal)
            # Make search radius larger for easiest anchor points
            base_search_radius = max(3 * bbox_height, 50, img_diagonal * 0.03)
            search_radius = base_search_radius * (1.5 if label in easiest_anchor_points else 1.0) # 50% larger for easy anchors


            min_dist = float('inf')
            closest_geometric_point = None

            # Attempt to find closest geometric point
            current_geometric_candidates = list(all_geometric_candidates) # Use a copy to iterate
            for geom_point in current_geometric_candidates:
                dist = np.linalg.norm(bbox_center - geom_point)
                if dist < min_dist and dist <= search_radius:
                    min_dist = dist
                    closest_geometric_point = geom_point
            
            if closest_geometric_point is not None:
                refined_points[label] = closest_geometric_point
                # Remove the associated geometric point to avoid re-assigning it
                all_geometric_candidates = [p for p in all_geometric_candidates if np.linalg.norm(p - closest_geometric_point) > 1.0] # Remove if very close
                print(f"Associated OCR label '{label}' with geometric point {closest_geometric_point.tolist()} (distance: {min_dist:.2f}, radius: {search_radius:.2f}).")
                sys.stdout.flush()
            else:
                print(f"No suitable geometric point found within radius {search_radius:.2f} for OCR label '{label}' at {bbox_center.tolist()}. Initiating fallback.")
                sys.stdout.flush()
                
                best_pixel_coords_fallback = bbox_center # Initialize fallback with OCR center (last resort)
                lowest_avg_intensity = float('inf')
                
                # Define ROI boundaries for searching the darkest average area
                # Make fallback search radius larger if it's an easy anchor
                fallback_search_radius = search_radius * (1.2 if label in easiest_anchor_points else 1.0)
                x_min_roi = max(0, int(bbox_center[0] - fallback_search_radius))
                y_min_roi = max(0, int(bbox_center[1] - fallback_search_radius))
                x_max_roi = min(gray.shape[1], int(bbox_center[0] + fallback_search_radius))
                y_max_roi = min(gray.shape[0], int(bbox_center[1] + fallback_search_radius))

                # Calculate expanded bbox for exclusion (20% more than original bbox)
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                
                expansion_factor = 0.20 # 20% more
                
                # Calculate expansion amounts for width and height
                expand_x = bbox_width * expansion_factor / 2
                expand_y = bbox_height * expansion_factor / 2
                
                # Define expanded bounding box
                expanded_bbox = [
                    max(0, int(bbox[0] - expand_x)),
                    max(0, int(bbox[1] - expand_y)),
                    min(gray.shape[1], int(bbox[2] + expand_x)),
                    min(gray.shape[0], int(bbox[3] + expand_y))
                ]

                # Perform "darkest average intensity" search
                if x_min_roi < x_max_roi and y_min_roi < y_max_roi:
                    # Pad the image to safely sample neighborhoods near edges
                    pad_amount = 4 # for 9x9 neighborhood (kernel size // 2)
                    padded_gray = np.pad(gray, pad_amount, mode='edge') 
                    
                    for y_coord in range(y_min_roi, y_max_roi):
                        for x_coord in range(x_min_roi, x_max_roi):
                            # Exclude pixels within the OCR label's expanded bounding box
                            if not (expanded_bbox[0] <= x_coord <= expanded_bbox[2] and expanded_bbox[1] <= y_coord <= expanded_bbox[3]):
                                # Adjust coordinates for padded image
                                padded_y = y_coord + pad_amount
                                padded_x = x_coord + pad_amount
                                neighborhood = padded_gray[padded_y - pad_amount : padded_y + pad_amount + 1, 
                                                           padded_x - pad_amount : padded_x + pad_amount + 1]
                                
                                if neighborhood.size == (2 * pad_amount + 1)**2: # Ensure full neighborhood
                                    current_avg_intensity = np.mean(neighborhood)
                                    if current_avg_intensity < lowest_avg_intensity:
                                        lowest_avg_intensity = current_avg_intensity
                                        best_pixel_coords_fallback = np.array([x_coord, y_coord])
                
                # Assign the best found fallback (either darkest average or original bbox_center)
                refined_points[label] = best_pixel_coords_fallback 
                print(f"Using fallback point {refined_points[label].tolist()} for '{label}'.")
                sys.stdout.flush()
            
            print(f"DEBUG: refined_points after processing '{label}': {refined_points.get(label).tolist()}") # Debug print for refined_points
            sys.stdout.flush()


    print(f"Final refined points based on OCR labels and geometric features: {refined_points}")
    sys.stdout.flush()


    print("--- Step 2 Complete ---")
    sys.stdout.flush()
    # Return detected_raw_lines directly instead of filtered_student_lines
    return refined_points, detected_raw_lines, detected_circles

# --- Step 3: Understand Diagram Structure from Question (LLMs) ---
def understand_diagram_structure_llm(question_text):
    """
    Uses the Gemini API to process the math olympiad problem question text.
    It determines drawing instructions, identifies "easiest anchor points",
    and explicitly lists expected line segments.
    """
    print(f"--- Step 3: Understanding diagram structure from question using Gemini LLM: '{question_text}' ---")
    sys.stdout.flush()
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""
    Analyze the following math olympiad problem question and provide structured drawing instructions.
    Determine:
    1.  A list of step-by-step drawing instructions.
    2.  The number of independent points needed to fully determine the diagram.
    3.  A list of the "easiest anchor points" (the first X independent points mentioned in the instructions, where X is the number of independent points).
    4.  A list of expected line segments, where each segment is represented as a list of two point labels (e.g., ["A", "B"]).

    Return the response as a JSON object with the following keys:
    "drawing_instructions": list of strings
    "num_determining_points": integer
    "easiest_anchor_points": list of strings (e.g., ["A", "B", "C"])
    "expected_line_segments": list of lists of strings (e.g., [["A", "B"], ["B", "C"]])

    Question: "{question_text}"
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "drawing_instructions": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "num_determining_points": {"type": "INTEGER"},
                        "easiest_anchor_points": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "expected_line_segments": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "STRING"}}}
                    },
                    "required": ["drawing_instructions", "num_determining_points", "easiest_anchor_points", "expected_line_segments"]
                }
            }
        )
        llm_output = json.loads(response.text)

        drawing_instructions = llm_output.get("drawing_instructions", [])
        num_determining_points = llm_output.get("num_determining_points", 0)
        easiest_anchor_points = llm_output.get("easiest_anchor_points", [])
        expected_line_segments = llm_output.get("expected_line_segments", [])

        print(f"Gemini LLM drawing instructions: {drawing_instructions}")
        print(f"Gemini LLM number of determining points: {num_determining_points}")
        print(f"Gemini LLM easiest anchor points: {easiest_anchor_points}")
        print(f"Gemini LLM expected line segments: {expected_line_segments}")
        sys.stdout.flush()


    except Exception as e:
        print(f"Error during Gemini LLM call for diagram structure: {e}")
        sys.stdout.flush()
        # Fallback to mock data on error
        drawing_instructions = [
            "Draw point A.", "Draw point C.", "Draw point B such that triangle ABC is isosceles with AB = BC.",
            "Draw line segment AC.", "Draw line segment AB.", "Draw line segment BC.",
            "Find the reflection D of point B over line AC.", "Draw line segment AD.", "Draw line segment CD."
        ]
        num_determining_points = 3
        easiest_anchor_points = ['A', 'C', 'B']
        expected_line_segments = [['A', 'B'], ['B', 'C'], ['A', 'C'], ['A', 'D'], ['C', 'D']]
        print("Using fallback mock data for LLM output due to error.")
        sys.stdout.flush()

    print("--- Step 3 Complete ---")
    sys.stdout.flush()
    return drawing_instructions, num_determining_points, easiest_anchor_points, expected_line_segments

# --- Helper function for reflection of a point over a line ---
def reflect_point_over_line(point, line_p1, line_p2):
    """
    Reflects a point (x0, y0) over a line defined by two points (x1, y1) and (x2, y2).
    Returns the reflected point (x', y').
    """
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    # Line equation: Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = -A * x1 - B * y1

    # Check for vertical or horizontal lines to avoid division by zero or simplify
    if A == 0: # Horizontal line y = -C/B
        x_prime = x0
        y_prime = 2 * (-C / B) - y0
    elif B == 0: # Vertical line x = -C/A
        x_prime = 2 * (-C / A) - x0
        y_prime = y0
    else:
        # General formula for reflection
        denominator = A**2 + B**2
        if denominator == 0: # Points are the same, line is undefined or a point
            return point # Or raise an error, depending on desired behavior

        temp = -2 * (A * x0 + B * y0 + C) / denominator
        x_prime = x0 + temp * A
        y_prime = y0 + temp * B

    return np.array([x_prime, y_prime])


# --- Step 4: Construct the Ideal Diagram Image (Programmatic Drawing) ---
def construct_ideal_diagram(student_refined_points, drawing_instructions, img_shape):
    """
    Programmatically constructs an ideal, complete diagram image using Pillow.
    This "ideal" diagram is a precise completion of the student's drawing,
    using the student's anchor points (A, B, C) to derive the position of D.
    The output image will be transparent with red lines/shapes and labels.
    """
    print("--- Step 4: Constructing the ideal diagram using Pillow (based on student's points) ---")
    sys.stdout.flush()

    # Get the student's detected pixel coordinates for A, B, C
    # These are the "anchor points" for this ideal diagram
    A_student_pixel = student_refined_points.get('A')
    B_student_pixel = student_refined_points.get('B')
    C_student_pixel = student_refined_points.get('C')

    # Ensure all crucial anchor points are present. D will be calculated.
    if A_student_pixel is None or B_student_pixel is None or C_student_pixel is None:
        missing_points = [p for p, val in {'A': A_student_pixel, 'B': B_student_pixel, 'C': C_student_pixel}.items() if val is None]
        print(f"Error: Missing crucial anchor points {missing_points} from student's diagram. Cannot construct ideal diagram.")
        sys.stdout.flush()
        # Return empty data and a specific error path
        return {}, "ideal_diagram_error.png"

    # Calculate the ideal position of D (reflection of B over AC) using student's pixel coordinates
    # This is the core of "finishing off their diagram for them"
    D_ideal_pixel = reflect_point_over_line(B_student_pixel, A_student_pixel, C_student_pixel)

    # The "ideal" coordinates for comparison are now these pixel coordinates
    ideal_coords_for_comparison = {
        'A': A_student_pixel,
        'B': B_student_pixel,
        'C': C_student_pixel,
        'D': D_ideal_pixel
    }

    print(f"Student's anchor points (pixel): {A_student_pixel}, {B_student_pixel}, {C_student_pixel}")
    print(f"Calculated ideal D point (pixel): {D_ideal_pixel}")
    sys.stdout.flush()

    img_width, img_height = img_shape[1], img_shape[0] # OpenCV returns (height, width, channels)

    # Create a new transparent image using Pillow
    ideal_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0)) # Transparent background
    draw = ImageDraw.Draw(ideal_img)

    # Define colors and line width
    line_color_solid = (255, 0, 0, 255) # Red, fully opaque
    point_color = (255, 0, 0, 255) # Red, fully opaque
    text_color = (255, 0, 0, 255) # Red, fully opaque
    line_width = 4 # To cover "within 2 pixels distance" (2 pixels on each side of the center line)
    point_radius = 8 # Increased point radius for better visibility
    text_offset_x = 15 # Increased offset for label placement
    text_offset_y = 15

    # Try to load a default font, or use Pillow's default if not found
    font = None
    font_paths = [
        "C:/Users/tweid/Projects/mini_edexia_trial/.venv/Lib/site-packages/Pillow/Libs/arial.ttf", # Common Windows path
        "C:/Windows/Fonts/arial.ttf", # Common Windows path
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", # Common Linux path
        "/Library/Fonts/Arial.ttf", # Common macOS path
        "arial.ttf" # Generic lookup
    ]
    
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 40) # Adjusted font size for better visibility
            print(f"DEBUG: Loaded font: {font.font.family}, size {font.size} from {path}")
            sys.stdout.flush()
            break
        except IOError:
            continue # Try next path
        except Exception as e:
            print(f"ERROR: Unexpected error loading font from {path}: {e}")
            sys.stdout.flush()
            continue
    
    if font is None:
        font = ImageFont.load_default()
        print(f"Warning: No specific Arial font found, using default Pillow font: {font.font.family}, size {font.size}.")
        sys.stdout.flush()


    # Draw triangle ABC (solid lines) using student's points
    draw.line([tuple(A_student_pixel), tuple(B_student_pixel)], fill=line_color_solid, width=line_width) # AB
    draw.line([tuple(B_student_pixel), tuple(C_student_pixel)], fill=line_color_solid, width=line_width) # BC
    draw.line([tuple(C_student_pixel), tuple(A_student_pixel)], fill=line_color_solid, width=line_width) # AC

    # Draw triangle ADC (using the calculated ideal D)
    draw.line([tuple(D_ideal_pixel), tuple(ideal_coords_for_comparison['C'])], fill=line_color_solid, width=line_width) # DC
    draw.line([tuple(A_student_pixel), tuple(D_ideal_pixel)], fill=line_color_solid, width=line_width) # AD


    # Draw points (small circles) and add labels for all points in the ideal diagram
    print("DEBUG: Starting drawing loop for labels on ideal diagram.")
    sys.stdout.flush()
    for label, coords_pixel in ideal_coords_for_comparison.items():
        x, y = coords_pixel
        print(f"DEBUG: Drawing point and label for '{label}' at ({x:.2f}, {y:.2f}).")
        sys.stdout.flush()
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color)
        # Add label slightly offset from the point
        draw.text((x + text_offset_x, y + text_offset_y), label, font=font, fill=text_color)
        sys.stdout.flush() # Flush after each label drawing
    print("DEBUG: Finished drawing loop for labels on ideal diagram.")
    sys.stdout.flush()


    ideal_diagram_img_path = "ideal_diagram.png"
    try:
        ideal_img.save(ideal_diagram_img_path)
        print(f"Ideal diagram (transparent PNG) saved to {ideal_diagram_img_path}")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR in construct_ideal_diagram: Failed to save {ideal_diagram_img_path}: {e}")
        sys.stdout.flush()
        return {}, "ideal_diagram_error_save_failed.png"
    print("--- Step 4 Complete ---")
    sys.stdout.flush()
    # Return the pixel coordinates of all points in this "completed" ideal diagram
    return ideal_coords_for_comparison, ideal_diagram_img_path

# --- Step 5: Identify and Add Labels (Angles/Lengths) to Student's Diagram (Geometric Analysis) ---
def identify_and_add_labels_geometric_analysis(refined_points, detected_raw_lines, ocr_lowercase_greek_labels):
    """
    Identifies angle and length labels on the student's diagram based on proximity.
    """
    print("--- Step 5: Identifying and adding labels (angles/lengths) ---")
    sys.stdout.flush()
    labeled_features = [] # To store associations like {'label': 'alpha', 'type': 'angle', 'associated_points': ['B']}

    # Calculate midpoints of detected lines (using raw lines now)
    line_midpoints = []
    for line_seg in detected_raw_lines:
        # Ensure line_seg contains valid coordinate tuples
        if len(line_seg) == 2 and isinstance(line_seg[0], tuple) and isinstance(line_seg[1], tuple):
            mid_x = (line_seg[0][0] + line_seg[1][0]) / 2
            mid_y = (line_seg[0][1] + line_seg[1][1]) / 2
            line_midpoints.append((mid_x, mid_y, line_seg))
        else:
            print(f"Warning: Skipping malformed line segment: {line_seg}")
            sys.stdout.flush()


    for label_info in ocr_lowercase_greek_labels:
        label_text = label_info['label']
        bbox = label_info['bbox']
        bbox_center = np.array(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))

        is_greek = False
        # Simple check for Greek letters (can be expanded with a more comprehensive list)
        if label_text.lower() in ['alpha', 'beta', 'gamma', 'delta', 'theta', 'phi', 'pi', 'sigma', 'omega']:
            is_greek = True

        min_dist_point = float('inf')
        closest_point_label = None
        for point_label, coords in refined_points.items():
            if coords is not None and not np.isnan(coords).any(): # Ensure coords are valid numpy array
                dist = np.linalg.norm(bbox_center - np.array(coords))
                if dist < min_dist_point:
                    min_dist_point = dist
                    closest_point_label = point_label

        min_dist_midpoint = float('inf')
        closest_line_segment = None
        for mid_x, mid_y, line_seg in line_midpoints:
            dist = np.linalg.norm(bbox_center - np.array((mid_x, mid_y)))
            if dist < min_dist_midpoint:
                min_dist_midpoint = dist
                closest_line_segment = line_seg

        if is_greek or (closest_point_label and min_dist_point < min_dist_midpoint and min_dist_point < 100): # Added a distance threshold
            # If it's a Greek letter OR closer to a point than a midpoint, assume angle label
            # And the point is reasonably close
            labeled_features.append({
                'label': label_text,
                'type': 'angle',
                'associated_feature': closest_point_label # Associated with a vertex
            })
            print(f"Label '{label_text}' identified as an angle at/near point '{closest_point_label}'.")
            sys.stdout.flush()
        elif closest_line_segment and min_dist_midpoint < 100: # Added a distance threshold
            # If closer to a midpoint, assume length label, and the line is reasonably close
            labeled_features.append({
                'label': label_text,
                'type': 'length',
                'associated_feature': closest_line_segment # Associated with a line segment
            })
            print(f"Label '{label_text}' identified as a length for line segment '{closest_line_segment}'.")
            sys.stdout.flush()
        else:
            print(f"Could not confidently associate label '{label_text}' with a feature.")
            sys.stdout.flush()

    print(f"Labeled features: {labeled_features}")
    sys.stdout.flush()
    print("--- Step 5 Complete ---")
    sys.stdout.flush()
    return labeled_features

# --- Helper Function: Visualize Student's Detected Points ---
def visualize_student_detected_points(original_img, student_refined_points, output_path="student_diagram_detected_points.png"):
    """
    Creates an image showing the original student's diagram with detected points
    overlaid in red. Does NOT draw text labels.
    """
    print(f"--- Visualizing student's detected points to {output_path} ---")
    sys.stdout.flush()
    print("DEBUG: Entering visualize_student_detected_points.")
    sys.stdout.flush()

    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    point_color = (255, 0, 0, 255) # Red, fully opaque
    point_radius = 10 # Larger radius for visibility

    print("DEBUG: Starting drawing loop for student detected points.")
    sys.stdout.flush()
    for label, coords in student_refined_points.items():
        print(f"DEBUG: Attempting to draw point for label '{label}' at {coords}.")
        sys.stdout.flush()
        if coords is None or np.isnan(coords).any():
            print(f"ERROR: Skipping drawing for label '{label}' due to invalid coordinates: {coords}")
            sys.stdout.flush()
            continue
        try:
            x, y = int(round(coords[0])), int(round(coords[1])) # Round and cast to int
            print(f"DEBUG: Drawing ellipse for '{label}' at ({x}, {y}).")
            sys.stdout.flush()
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color, outline=point_color)
            print(f"DEBUG: Successfully drew point for '{label}'.")
            sys.stdout.flush()
        except Exception as e:
            print(f"ERROR in visualize_student_detected_points drawing loop for label '{label}' at {coords}: {e}")
            sys.stdout.flush()
            continue
    print("DEBUG: Finished drawing loop in visualize_student_detected_points.")
    sys.stdout.flush()

    try:
        print(f"DEBUG: Attempting to save {output_path} in visualize_student_detected_points.")
        sys.stdout.flush()
        img_pil.save(output_path)
        print(f"Student's detected points visualization saved to {output_path}")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR in visualize_student_detected_points: Failed to save {output_path}: {e}")
        sys.stdout.flush()
    print("--- Visualization Complete ---")
    sys.stdout.flush()

# --- Helper Function: Visualize Detected Lines ---
def visualize_detected_lines(original_img, detected_lines, output_path):
    """
    Creates an image showing the original student's diagram with detected lines overlaid in blue.
    Does NOT draw text labels.
    """
    print(f"--- Visualizing detected lines to {output_path} ---")
    sys.stdout.flush()
    img_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    line_color = (0, 0, 255, 200) # Blue, semi-transparent
    line_width = 3

    for line_seg in detected_lines:
        try:
            # Ensure coordinates are integers and in the correct format for Pillow
            p1_x, p1_y = int(round(line_seg[0][0])), int(round(line_seg[0][1]))
            p2_x, p2_y = int(round(line_seg[1][0])), int(round(line_seg[1][1]))
            print(f"DEBUG: Drawing raw line from ({p1_x}, {p1_y}) to ({p2_x}, {p2_y}).") # Added debug print
            draw.line([(p1_x, p1_y), (p2_x, p2_y)], fill=line_color, width=line_width)
        except Exception as e:
            print(f"ERROR in visualize_detected_lines: Failed to draw line segment {line_seg}: {e}")
            sys.stdout.flush()
            continue

    sys.stdout.flush() # Flush after loop
    try:
        img_pil.save(output_path)
        print(f"Detected lines visualization saved to {output_path}")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR in visualize_detected_lines: Failed to save {output_path}: {e}")
        sys.stdout.flush()
    print("--- Line Visualization Complete ---")
    sys.stdout.flush()

# --- Removed: Visualize Processed (Joined) Lines function as per user request ---
# def visualize_processed_lines(original_img, processed_lines, output_path="student_diagram_processed_lines.png"):
#     ...

# --- Step 6: Judge the Student's Diagram (Comparison & Assessment) ---
def judge_student_diagram(student_refined_points, ideal_coords_for_comparison, hand_drawn_img_path, ideal_diagram_img_path):
    """
    Compares the student's diagram with the ideal diagram (completed based on student's anchors)
    both quantitatively and qualitatively using Gemini multi-modal LLM.
    """
    print("--- Step 6: Judging the student's diagram using Gemini Multi-modal LLM ---")
    sys.stdout.flush()

    # Quantitative Comparison: Compare point distances
    aggregate_distance = 0
    num_compared_points = 0
    point_comparison_details = {}

    # ideal_coords_for_comparison are already in the pixel space of the student's diagram,
    # so no further scaling/offset is needed for direct comparison.
    for label, ideal_point_pixel in ideal_coords_for_comparison.items():
        if label in student_refined_points:
            student_point = student_refined_points[label]
            # Ensure points are valid before comparison
            if student_point is not None and not np.isnan(student_point).any() and \
               ideal_point_pixel is not None and not np.isnan(ideal_point_pixel).any():
                distance = np.linalg.norm(np.array(student_point) - np.array(ideal_point_pixel))
                aggregate_distance += distance
                num_compared_points += 1
                point_comparison_details[label] = {
                    'student_coords': student_point.tolist(),
                    'ideal_coords_completed': ideal_point_pixel.tolist(), # Renamed for clarity
                    'distance': distance
                }
            else:
                print(f"Warning: Skipping quantitative comparison for label '{label}' due to invalid coordinates (student: {student_point}, ideal: {ideal_point_pixel}).")
                sys.stdout.flush()
        else:
            # This case should ideally not happen for A, B, C as they are input anchors.
            # It might happen for D if OCR failed to detect 'D' but it was calculated.
            print(f"Warning: Label '{label}' from ideal diagram (completed) not found in student's detected points for quantitative comparison.")
            sys.stdout.flush()

    if num_compared_points > 0:
        average_distance = aggregate_distance / num_compared_points
        print(f"Quantitative Comparison: Average point distance = {average_distance:.2f} pixels.")
        print("Point-by-point comparison details:")
        sys.stdout.flush()
        for label, details in point_comparison_details.items():
            print(f"  {label}: Student({details['student_coords'][0]:.0f}, {details['student_coords'][1]:.0f}), "
                  f"Ideal_Completed({details['ideal_coords_completed'][0]:.0f}, {details['ideal_coords_completed'][1]:.0f}), "
                  f"Distance={details['distance']:.2f}")
            sys.stdout.flush()
    else:
        print("No common points to compare quantitatively or all points had invalid coordinates.")
        sys.stdout.flush()

    # Qualitative Comparison (Multi-modal LLM)
    model_vision = genai.GenerativeModel('gemini-2.0-flash')

    # Function to load image and convert to base64
    def get_image_data(image_path):
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}. Skipping LLM vision call.")
            sys.stdout.flush()
            return None
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        # Determine mime type based on file extension
        mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        return {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode('utf-8')
        }

    try:
        hand_drawn_image_data = get_image_data(hand_drawn_img_path)
        ideal_diagram_image_data = get_image_data(ideal_diagram_img_path)

        if hand_drawn_image_data is None or ideal_diagram_image_data is None:
            raise FileNotFoundError("One or both diagram images for LLM comparison were not found.")

        prompt_parts = [
            {"text": "Compare these two diagrams. The first image is a hand-drawn diagram by a student, and the second image is the ideal, programmatically generated diagram based on the student's initial points. Provide a structured assessment as a JSON object with the following keys:\n'overall_assessment': string\n'similarities': list of strings\n'differences': list of strings\n'suggestions_for_improvement': list of strings"},
            hand_drawn_image_data,
            ideal_diagram_image_data
        ]

        response = model_vision.generate_content(
            prompt_parts,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {
                        "overall_assessment": {"type": "STRING"},
                        "similarities": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "differences": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "suggestions_for_improvement": {"type": "ARRAY", "items": {"type": "STRING"}}
                    },
                    "required": ["overall_assessment", "similarities", "differences", "suggestions_for_improvement"]
                }
            }
        )
        llm_response = json.loads(response.text)

        print("\nQualitative Comparison (Gemini Multi-modal LLM):")
        print(f"LLM Overall Assessment: {llm_response.get('overall_assessment', 'N/A')}")
        print("Similarities:")
        for s in llm_response.get('similarities', []):
            print(f"- {s}")
        print("Differences:")
        for d in llm_response.get('differences', []):
            print(f"- {d}")
        print("Suggestions for Improvement:")
        for s in llm_response.get('suggestions_for_improvement', []): # Corrected key from 'suggestions' to 'suggestions_for_improvement'
            print(f"- {s}")
        sys.stdout.flush()

    except Exception as e:
        print(f"Error during Gemini multi-modal LLM call: {e}")
        sys.stdout.flush()
        llm_response = {
            "overall_assessment": "Could not perform qualitative assessment due to API error or missing image files.",
            "similarities": [],
            "differences": [],
            "suggestions_for_improvement": ["Check API key and network connection.", "Ensure image files exist at specified paths."]
        }
        print("Using fallback mock data for multi-modal LLM output due to error.")
        sys.stdout.flush()

    print("--- Step 6 Complete ---")
    sys.stdout.flush()
    return llm_response, point_comparison_details

# --- Step 7: Visual Overlay on Website (Conceptual) ---
def visual_overlay_conceptual(hand_drawn_img_path, ideal_diagram_img_path):
    """
    Describes how the visual overlay would be achieved on a website.
    This function doesn't produce an actual overlay, but explains the concept.
    """
    print("\n--- Step 7: Visual Overlay on Website (Conceptual) ---")
    sys.stdout.flush()
    print("To achieve a visual overlay on a website, you would typically use HTML and CSS.")
    print("1. Display the original hand-drawn image as the base layer:")
    print(f"   <img src='{hand_drawn_img_path}' style='position: absolute; top: 0; left: 0; width: 100%; height: auto;'>")
    print("2. Display the programmatically constructed ideal diagram image as an overlay:")
    print(f"   <img src='{ideal_diagram_img_path}' style='position: absolute; top: 0; left: 0; width: 100%; height: auto; opacity: 0.5;'>")
    print("   (Adjust 'top', 'left', 'width', 'height' and 'opacity' to align and blend the images.)")
    print("Alternatively, you could use a canvas element and draw both images onto it,")
    print("allowing for more precise control over blending and transformations.")
    print("--- Step 7 Complete ---")
    sys.stdout.flush()

# --- Main Workflow Execution ---
def main_workflow(image_path, question_text):
    """
    Orchestrates the entire diagram processing workflow.
    """
    print("--- Starting Diagram Processing Workflow ---")
    sys.stdout.flush()

    # Step 1: Initial Image Processing & OCR for Labels
    hand_drawn_img, ocr_capital_labels, ocr_lowercase_greek_labels = initial_image_processing_ocr(image_path)
    if hand_drawn_img is None:
        print("Workflow aborted due to image loading error.")
        sys.stdout.flush()
        return
    print("DEBUG: After initial_image_processing_ocr.")
    sys.stdout.flush()

    # Step 3 (called early to get anchor points for Step 2)
    drawing_instructions, num_determining_points, easiest_anchor_points, expected_line_segments = understand_diagram_structure_llm(question_text)
    print("DEBUG: After understand_diagram_structure_llm (early call).")
    sys.stdout.flush()


    # Step 2: Identify Diagram Vertices/Points & Shapes from Hand-drawn Diagram
    student_refined_points, student_detected_raw_lines, student_detected_circles = \
        identify_diagram_features(hand_drawn_img, ocr_capital_labels, easiest_anchor_points, expected_line_segments)
    print("DEBUG: After identify_diagram_features.")
    sys.stdout.flush()

    # Step 4: Construct the Ideal Diagram Image (Programmatic Drawing)
    ideal_coords_for_comparison, ideal_diagram_img_path = construct_ideal_diagram(student_refined_points, drawing_instructions, hand_drawn_img.shape)
    print("DEBUG: After construct_ideal_diagram.")
    sys.stdout.flush()
    if ideal_diagram_img_path == "ideal_diagram_error.png" or ideal_diagram_img_path == "ideal_diagram_error_save_failed.png":
        print("Workflow aborted due to ideal diagram construction error.")
        sys.stdout.flush()
        return

    # Step 5: Identify and Add Labels (Angles/Lengths) to Student's Diagram (Geometric Analysis)
    # Pass student_detected_raw_lines instead of student_detected_lines_processed
    labeled_features = identify_and_add_labels_geometric_analysis(student_refined_points, student_detected_raw_lines, ocr_lowercase_greek_labels)
    print("DEBUG: After identify_and_add_labels_geometric_analysis.")
    sys.stdout.flush()

    # Step 6: Judge the Student's Diagram (Comparison & Assessment)
    print("DEBUG: Calling judge_student_diagram.")
    sys.stdout.flush()
    llm_assessment, point_comparison_details = judge_student_diagram(student_refined_points, ideal_coords_for_comparison, image_path, ideal_diagram_img_path)
    print("DEBUG: After judge_student_diagram.")
    sys.stdout.flush()

    # --- NEW: Visualizations for debugging ---
    visualize_student_detected_points(hand_drawn_img, student_refined_points)
    visualize_detected_lines(hand_drawn_img, student_detected_raw_lines, output_path="student_diagram_detected_raw_lines.png")
    print("DEBUG: After all visualizations.")
    sys.stdout.flush()

    # Step 7: Visual Overlay on Website (Conceptual)
    visual_overlay_conceptual(image_path, ideal_diagram_img_path)
    print("DEBUG: After visual_overlay_conceptual.")
    sys.stdout.flush()

    print("\n--- Diagram Processing Workflow Complete ---")
    sys.stdout.flush()
    print("\nSummary of Results:")
    sys.stdout.flush()
    print(f"  Student's Detected Points: {student_refined_points}")
    print(f"  Ideal Diagram (Completed based on student's anchors): {ideal_coords_for_comparison}")
    print(f"  Quantitative Comparison Details: {point_comparison_details}")
    print(f"  Qualitative Assessment: {llm_assessment.get('overall_assessment', 'N/A')}")
    print(f"  Ideal Diagram Image saved at: {ideal_diagram_img_path}")
    sys.stdout.flush()


# --- Execute the workflow ---
if __name__ == "__main__":
    input_image_path = "image_e14ecd.jpg"
    math_olympiad_question = "Draw an isosceles triangle where AB = BC then construct ABDC where D is the reflection of B over AC"

    main_workflow(input_image_path, math_olympiad_question)
