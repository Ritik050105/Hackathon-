from gtts import gTTS
import os
import soundfile as sf
import numpy as np
from pesq import pesq

def text_to_speech(text, lang='en', output_file="output_audio.wav", slow=False):
    """
    Converts text to speech using gTTS and saves it as a .wav file.

    Args:
        text (str): Text to convert to speech.
        lang (str): Language code for speech (default is 'en' for English).
        output_file (str): Path to save the output audio file.
        slow (bool): Whether to speak slowly (default is False).

    Returns:
        str: Path to the generated audio file.
    """
    try:
        # Convert text to speech using gTTS
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Save as a temporary MP3 file
        temp_mp3 = "temp_audio.mp3"
        tts.save(temp_mp3)
        
        # Convert MP3 to WAV for better analysis
        data, samplerate = sf.read(temp_mp3)
        sf.write(output_file, data, samplerate)
        
        # Remove the temporary MP3 file
        os.remove(temp_mp3)
        
        print(f"Audio saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return None

def calculate_noise_ratio(audio_file):
    """
    Calculates the noise ratio of an audio file.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        float: Noise ratio (percentage of noise in the audio).
    """
    try:
        # Load the audio file
        data, samplerate = sf.read(audio_file)
        
        # Calculate the noise ratio
        noise_ratio = np.mean(np.abs(data)) * 100
        return noise_ratio
    except Exception as e:
        print(f"Error calculating noise ratio: {e}")
        return None

def calculate_pesq(reference_file, degraded_file):
    """
    Calculates the PESQ score between a reference and degraded audio file.

    Args:
        reference_file (str): Path to the reference audio file.
        degraded_file (str): Path to the degraded audio file.

    Returns:
        float: PESQ score.
    """
    try:
        # Calculate PESQ score
        score = pesq(reference_file, degraded_file, 'wb')
        return score
    except Exception as e:
        print(f"Error calculating PESQ score: {e}")
        return None

def main():
    # Example text to convert to speech
    text = "This is a sample text for testing the text-to-speech functionality."
    
    # Step 1: Convert text to speech
    print("Converting text to speech...")
    audio_file = text_to_speech(text, lang='en', output_file="output_audio.wav", slow=False)
    
    if audio_file:
        # Step 2: Calculate noise ratio
        print("\nCalculating noise ratio...")
        noise_ratio = calculate_noise_ratio(audio_file)
        print(f"Noise Ratio: {noise_ratio:.2f}%")

        # Step 3: Calculate PESQ score (humaneness)
        print("\nCalculating PESQ score...")
        # Provide a reference audio file for comparison
        reference_file = "reference_audio.wav"  # Replace with your reference file
        if os.path.exists(reference_file):
            pesq_score = calculate_pesq(reference_file, audio_file)
            print(f"PESQ Score: {pesq_score:.2f}")
        else:
            print("Reference audio file not found. Skipping PESQ calculation.")

if __name__ == "__main__":
    main()
