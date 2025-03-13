import pytesseract
from PIL import Image
import re
from Levenshtein import distance
import os

# Set the path to the Tesseract executable
# For Windows: Update this path to your Tesseract installation
# For Linux/Mac: Ensure Tesseract is installed and accessible in the PATH
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path, lang='eng'):
    """
    Extracts text from an image using Tesseract OCR.

    Args:
        image_path (str): Path to the image file.
        lang (str): Language code for OCR (default is 'eng' for English).

    Returns:
        str: Extracted text.
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(image, lang=lang)
        
        # Clean the extracted text
        text = clean_text(text)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def clean_text(text):
    """
    Cleans the extracted text by removing unnecessary characters and spaces.

    Args:
        text (str): Extracted text from the image.

    Returns:
        str: Cleaned text.
    """
    try:
        # Replace multiple newlines with a single space
        text = re.sub(r'\n+', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text.strip()
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text  # Return original text if cleaning fails

def calculate_wer(reference, hypothesis):
    """
    Calculates Word Error Rate (WER) between reference and hypothesis texts.

    Args:
        reference (str): Ground truth text.
        hypothesis (str): Extracted text from OCR.

    Returns:
        float: Word Error Rate (WER).
    """
    # Split the texts into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Calculate the number of substitutions, insertions, and deletions
    wer = distance(ref_words, hyp_words) / len(ref_words)
    return wer

def main():
    # Path to the image
    image_path = r"D:\UNSTOP\Hackathon\newspaper_summarization_app\images\newspaper.jpg"  # Replace with your image path
    
    # Path to the ground truth text file
    ground_truth_path = "ground_truth.txt"  # Replace with your ground truth file path

    # Step 1: Extract text from the image
    print("Extracting text from the image...")
    extracted_text = extract_text_from_image(image_path, lang='eng')
    print("Extracted Text:")
    print(extracted_text)

    # Step 2: Read ground truth text
    with open(ground_truth_path, "r") as f:
        reference_text = f.read()
    print("\nGround Truth Text:")
    print(reference_text)

    # Step 3: Calculate Word Error Rate (WER)
    print("\nCalculating Word Error Rate (WER)...")
    wer = calculate_wer(reference_text, extracted_text)
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")

if __name__ == "__main__":
    main()
