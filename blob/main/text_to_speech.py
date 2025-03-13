from gtts import gTTS
import os

def text_to_speech(text, lang='en', output_file="output_audio.mp3"):
    """
    Converts text to speech using gTTS.
    
    Args:
        text (str): Text to convert to speech.
        lang (str): Language code for speech (default is 'en' for English).
        output_file (str): Path to save the output audio file.
    
    Returns:
        str: Path to the generated audio file.
    """
    try:
        # Convert text to speech
        tts = gTTS(text=text, lang=lang)
        tts.save(output_file)
        return output_file
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return None
