import cv2  
import numpy as np  
import easyocr 
import pyperclip        # For copying text to clipboard
from concurrent.futures import ThreadPoolExecutor   # Helps to run multiple tasks at once (multithreading)

# load the OCR model for English and german
reader = easyocr.Reader(['en', 'de'])

# function: load image and resize it to minimize runtime
def load_img(file, scale_factor=0.5):
    img = cv2.imread(file)
    height, width = img.shape[:2]
    resized_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))  # shrink image
    return resized_img 


# Function for text detection (OCR) on image
def detect_text(img):
    # check if input is a NumPy array (which is how images are stored in memory)
    if isinstance(img, np.ndarray):
        # ThreadPoolExecutor for parallel processing (multithreading)
        with ThreadPoolExecutor() as executor:
            future = executor.submit(reader.readtext, img)      # Submit OCR task to the thread pool
            return future.result()
    else:
        raise ValueError("Invalid image format")


# Function to draw text boxes
def draw_text_box(img, grouped_texts):
    for box, text in grouped_texts:
        top_left = (int(box[0][0]), int(box[0][1]))
        bottom_right = (int(box[2][0]), int(box[2][1]))

        # Draw a pink rectangle around the detected text
        cv2.rectangle(img, top_left, bottom_right, (100, 0, 200), 2)
    
    cv2.imshow("Detected Text", img)  # Display image


# Function to group nearby text boxes together if they are close enough
def group_text_boxes(results, x_threshold = 50, y_threshold = 90, font_threshold = 10):
    grouped_texts = []  # Store grouped text boxes here
    
    # Sort text boxes by top-left corner for easier grouping and sort y-coordinate first, then x
    results.sort(key = lambda x: (x[0][0][1], x[0][0][0]))

    for result in results:
        box, text = result[:2]      # Get first 2 elements from result (coordinates, text)

        # If the current box is close to the last one, merge them
        if grouped_texts and is_nearby(grouped_texts[-1][0], box, x_threshold, y_threshold, font_threshold):
            grouped_texts[-1][1] += " " + text      # Add new text to previous group
            grouped_texts[-1][0] = merge_boxes(grouped_texts[-1][0], box)
        else:
            grouped_texts.append([box, text])   # Otherwise, treat this as a new group

    return grouped_texts  # Return grouped text boxes [top left, top right, bottom right, bottom left]


# check if two text boxes are close to each other
def is_nearby(box1, box2, x_threshold, y_threshold, font_threshold):
    x_dist = abs(box1[2][0] - box2[0][0])   # Distance between boxes on the x-axis
    y_dist = abs(box1[0][1] - box2[0][1])   # Distance on the y-axis
    
    # Calculate the height of the text in each box
    height1 = box1[2][1] - box1[0][1]
    height2 = box2[2][1] - box2[0][1]
    
    # Check if the font sizes are similar enough
    font_size_similar = abs(height1 - height2) < font_threshold
    
    # Return True if distances and font size are within thresholds
    return x_dist < x_threshold and y_dist < y_threshold and font_size_similar


# merge two boxes into one (larger box that covers both)
def merge_boxes(box1, box2):
    top_left = (min(box1[0][0], box2[0][0]), min(box1[0][1], box2[0][1])) 
    bottom_right = (max(box1[2][0], box2[2][0]), max(box1[2][1], box2[2][1]))
    return [top_left, (bottom_right[0], top_left[1]), bottom_right, (top_left[0], bottom_right[1])]


# merge boxes that are close to eachother in y-axis
def check_paragraph(grouped_texts, parag_y_threshold = 5, parag_x_threshold = 47):
    boxes = [grouped_texts[0]]
    delete = []

    for box1, text1 in grouped_texts[1:]:  # Start from the second box
        merged = False      # Flag to check if box1 was merged with an existing box

        for i, (box2, text2) in enumerate(boxes):
            # If the current box is near an existing paragraph box, merge them
            if is_nearby_parag(box2, box1, parag_y_threshold, parag_x_threshold):
                boxes[i][1] += " " + text1      # Combine the text
                boxes[i][0] = merge_boxes(box2, box1)
                delete.extend([[box1, text1]])
                merged = True       # Mark as merged
                break

        # If no merging occurred, add this box as a new paragraph
        if not merged:
            boxes.append([box1, text1])

    return boxes


# helper function to caculate distances
def is_nearby_parag(box1, box2, parag_y_threshold, parag_x_threshold):
    # Calculate the vertical and horizontal distances between the two boxes
    parag_dist_y = abs(box1[3][1] - box2[0][1]) 
    parag_dist_x = abs(box1[0][0] - box2[0][0])     # Horizontal distance (ensure lines are somewhat aligned)

    return parag_dist_y < parag_y_threshold and parag_dist_x < parag_x_threshold


# Function that lets us copy text when we click on a box
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:      # Check for left mouse button click
        results, img = param                # Get the detected text results and the image
        for box, text in results:
            top_left = (int(box[0][0]), int(box[0][1]))
            bottom_right = (int(box[2][0]), int(box[2][1])) 
            if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:     # Check if the click was inside the box
                pyperclip.copy(text.lower())                                                    # Copy the text (in lowercase)
                print(f"Copied text: {text.lower()}")



def main():
    # Load image from file
    img = "brain/yogurt.png"
    img = load_img(img)
    
    # Create window to display the image
    cv2.namedWindow("Detected Text", cv2.WINDOW_NORMAL)
    
    # Run text detection on image, group texts and check paragraphs
    results = detect_text(img)
    grouped_texts = group_text_boxes(results)
    boxes = check_paragraph(grouped_texts)
    
    draw_text_box(img, boxes)
    
    # Set up mouse click callback to copy by clicking
    cv2.setMouseCallback("Detected Text", mouse_callback, (boxes, img))
    
    # Resize window to fit image
    screen_height, screen_width = img.shape[:2]     # Get image size
    cv2.resizeWindow("Detected Text", screen_width, screen_height)  # Make window the right size
    
    # show window until 'q' is pressed
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cv2.destroyAllWindows()  # Close all OpenCV windows when done

if __name__ == "__main__":
    main()