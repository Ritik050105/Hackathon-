import pytesseract
from PIL import Image

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
        image = Image.open(r"D:\UNSTOP\Hackathon\newspaper_summarization_app\images\newspaper.jpg")
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(image, lang=lang)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""
