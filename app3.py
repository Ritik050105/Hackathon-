import pytesseract
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator
import os
import streamlit as st

# Function to extract text from image using OCR with the selected language
def extract_text_from_image(image_path, lang='eng'):
    """
    Extracts text from an image using pytesseract OCR.
    Supports multiple languages (e.g., 'eng' for English, 'fra' for French, etc.).
    """
    # Open the image file
    img = Image.open(image_path)
    
    # Use pytesseract to extract text from the image in the specified language
    extracted_text = pytesseract.image_to_string(img, lang=lang)
    
    return extracted_text

# Function to translate text to the desired language
def translate_text(text, dest_language='en'):
    """
    Translates the input text to the desired language using deep-translator.
    """
    translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
    return translated

# Improved summarization: Taking the first few paragraphs as summary
def long_summarize(text, num_paragraphs=5):
    """
    Summarizes the text by taking the first few paragraphs.
    """
    paragraphs = text.split('\n\n')  # Split the text into paragraphs
    summary = '\n\n'.join(paragraphs[:num_paragraphs])  # Join the first 'num_paragraphs' paragraphs
    return summary

# Function to convert text to speech in the specified language
def text_to_speech(text, language='en'):
    """
    Converts the input text to speech in the specified language.
    """
    tts = gTTS(text=text, lang=language, slow=False)
    audio_file = "output.mp3"
    tts.save(audio_file)
    return audio_file

# Streamlit App
def main():
    st.title("OCR, Translate, Summarize, and Text-to-Speech")
    st.write("Upload an image, extract text using OCR, translate it, summarize it, and optionally convert the summary to speech.")

    # Define supported languages for OCR (Tesseract) and translation (Google Translate)
    ocr_languages = {
        "English (eng)": "eng",
        "French (fra)": "fra",
        "Spanish (spa)": "spa",
        "German (deu)": "deu",
        "Chinese (Simplified) (chi_sim)": "chi_sim",
        "Chinese (Traditional) (chi_tra)": "chi_tra",
        "Japanese (jpn)": "jpn",
        "Korean (kor)": "kor",
        "Arabic (ara)": "ara",
        "Russian (rus)": "rus",
        "Italian (ita)": "ita",
        "Portuguese (por)": "por",
        "Dutch (nld)": "nld",
        "Hindi (hin)": "hin",
        "Turkish (tur)": "tur",
        "Vietnamese (vie)": "vie",
        "Thai (tha)": "tha",
        "Greek (ell)": "ell",
        "Hebrew (heb)": "heb",
        "Polish (pol)": "pol",
        "Swedish (swe)": "swe",
        "Finnish (fin)": "fin",
        "Danish (dan)": "dan",
        "Norwegian (nor)": "nor",
        "Czech (ces)": "ces",
        "Hungarian (hun)": "hun",
        "Romanian (ron)": "ron",
        "Ukrainian (ukr)": "ukr",
        "Bulgarian (bul)": "bul",
        "Croatian (hrv)": "hrv",
        "Slovak (slk)": "slk",
        "Slovenian (slv)": "slv",
        "Estonian (est)": "est",
        "Latvian (lav)": "lav",
        "Lithuanian (lit)": "lit",
        "Malay (msa)": "msa",
        "Indonesian (ind)": "ind",
        "Filipino (fil)": "fil",
        "Urdu (urd)": "urd",
        "Persian (fas)": "fas",
        "Bengali (ben)": "ben",
        "Tamil (tam)": "tam",
        "Telugu (tel)": "tel",
        "Kannada (kan)": "kan",
        "Malayalam (mal)": "mal",
        "Sinhala (sin)": "sin",
        "Burmese (mya)": "mya",
        "Khmer (khm)": "khm",
        "Lao (lao)": "lao",
        "Tibetan (bod)": "bod",
        "Mongolian (mon)": "mon",
        "Nepali (nep)": "nep",
        "Pashto (pus)": "pus",
        "Kurdish (kur)": "kur",
        "Uighur (uig)": "uig",
        "Yiddish (yid)": "yid",
        "Esperanto (epo)": "epo",
        "Latin (lat)": "lat",
    }

    output_languages = {
        "English (en)": "en",
        "French (fr)": "fr",
        "Spanish (es)": "es",
        "German (de)": "de",
        "Chinese (Simplified) (zh-CN)": "zh-CN",
        "Chinese (Traditional) (zh-TW)": "zh-TW",  # Corrected to zh-TW
        "Japanese (ja)": "ja",
        "Korean (ko)": "ko",
        "Arabic (ar)": "ar",
        "Russian (ru)": "ru",
        "Italian (it)": "it",
        "Portuguese (pt)": "pt",
        "Dutch (nl)": "nl",
        "Hindi (hi)": "hi",
        "Turkish (tr)": "tr",
        "Vietnamese (vi)": "vi",
        "Thai (th)": "th",
        "Greek (el)": "el",
        "Hebrew (he)": "he",
        "Polish (pl)": "pl",
        "Swedish (sv)": "sv",
        "Finnish (fi)": "fi",
        "Danish (da)": "da",
        "Norwegian (no)": "no",
        "Czech (cs)": "cs",
        "Hungarian (hu)": "hu",
        "Romanian (ro)": "ro",
        "Ukrainian (uk)": "uk",
        "Bulgarian (bg)": "bg",
        "Croatian (hr)": "hr",
        "Slovak (sk)": "sk",
        "Slovenian (sl)": "sl",
        "Estonian (et)": "et",
        "Latvian (lv)": "lv",
        "Lithuanian (lt)": "lt",
        "Malay (ms)": "ms",
        "Indonesian (id)": "id",
        "Filipino (tl)": "tl",
        "Urdu (ur)": "ur",
        "Persian (fa)": "fa",
        "Bengali (bn)": "bn",
        "Tamil (ta)": "ta",
        "Telugu (te)": "te",
        "Kannada (kn)": "kn",
        "Malayalam (ml)": "ml",
        "Sinhala (si)": "si",
        "Burmese (my)": "my",
        "Khmer (km)": "km",
        "Lao (lo)": "lo",
        "Tibetan (bo)": "bo",
        "Mongolian (mn)": "mn",
        "Nepali (ne)": "ne",
        "Pashto (ps)": "ps",
        "Kurdish (ku)": "ku",
        "Uighur (ug)": "ug",
        "Yiddish (yi)": "yi",
        "Esperanto (eo)": "eo",
        "Latin (la)": "la",
    }

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # OCR language selection
        ocr_language = st.selectbox(
            "Select OCR Language",
            options=list(ocr_languages.keys()),
            index=0
        )

        # Output language selection
        output_language = st.selectbox(
            "Select Output Language for Translation",
            options=list(output_languages.keys()),
            index=0
        )

        # Checkbox to play audio
        play_audio = st.checkbox("Play Audio Summary")

        # Process the image when the user clicks the button
        if st.button("Process Image"):
            # Extract text from the image
            extracted_text = extract_text_from_image(uploaded_file, lang=ocr_languages[ocr_language])
            st.subheader("Extracted Text")
            st.text_area("Full Extracted Text", extracted_text, height=300)

            # Translate the extracted text
            if ocr_languages[ocr_language] != output_languages[output_language]:
                translated_text = translate_text(extracted_text, dest_language=output_languages[output_language])
            else:
                translated_text = extracted_text
            st.subheader("Translated Text")
            st.text_area("Full Translated Text", translated_text, height=300)

            # Summarize the translated text
            summary = long_summarize(translated_text)
            st.subheader("Summary")
            st.write(summary)

            # Convert the summary to speech if requested
            if play_audio:
                audio_file = text_to_speech(summary, output_languages[output_language])
                st.audio(audio_file, format="audio/mp3")

                # Add a download button for the audio file
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                st.download_button(
                    label="Download Audio Summary",
                    data=audio_bytes,
                    file_name="summary_audio.mp3",
                    mime="audio/mp3"
                )

# Run the Streamlit app
if __name__ == "__main__":
    main()
