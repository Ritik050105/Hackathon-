import streamlit as st
from PIL import Image
import pytesseract
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from gtts import gTTS
from deep_translator import GoogleTranslator
import os
import base64

# Set the path to the Tesseract executable
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # Linux (Docker/Streamlit Sharing)
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Download NLTK data (if not already downloaded)
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')

# Function to extract text from an image
def extract_text_from_image(image, lang='eng'):
    """
    Extracts text from an image using pytesseract OCR.
    Supports multiple languages (e.g., 'eng' for English, 'fra' for French, etc.).
    """
    try:
        # Use pytesseract to extract text from the image in the specified language
        text = pytesseract.image_to_string(image, lang=lang)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

# Function to clean the extracted text
def clean_text(text):
    """
    Cleans the extracted text by removing unnecessary characters and spaces.
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
        st.error(f"Error cleaning text: {e}")
        return text  # Return original text if cleaning fails

# Function to translate text to the desired language
def translate_text(text, dest_language='en'):
    """
    Translates the input text to the desired language using deep-translator.
    """
    try:
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        return translated
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return text  # Return original text if translation fails

# Function to summarize the text
def summarize_text(text, sentences_count=3):
    """
    Summarizes the text using LSA summarization.
    """
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return ""

# Function to analyze sentiment
def analyze_sentiment(text):
    """
    Analyzes the sentiment of the text using NLTK's SentimentIntensityAnalyzer.
    """
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)
        return sentiment_score
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return {}

# Function to add military flair and sentiment analysis
def add_military_flair(text, sentiment_score):
    """
    Adds a military-style intro and outro to the text based on sentiment analysis.
    """
    # Determine sentiment description
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        sentiment_desc = "This news is generally positive."
    elif compound_score <= -0.05:
        sentiment_desc = "This news is generally negative."
    else:
        sentiment_desc = "This news is neutral."

    # Add military-style intro and outro
    intro = f"Incoming transmission! Stand by for news update!\n\nSentiment Analysis: {sentiment_desc}\n\n"
    outro = "\n\nMessage complete. Over and out."
    return f"{intro}{text}{outro}"

# Function to convert text to speech
def text_to_speech(text, lang='en', output_file="output_audio.mp3"):
    """
    Converts the input text to speech using gTTS.
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)
        return output_file
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

# Streamlit App
def main():
    st.title("Automated Newspaper Summarization and Analysis")
    st.write("Upload a newspaper image to extract, summarize, and analyze the text.")

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
        "Chinese (Traditional) (zh-TW)": "zh-TW",
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
            extracted_text = extract_text_from_image(image, lang=ocr_languages[ocr_language])
            st.subheader("Extracted Text")
            st.text_area("Full Extracted Text", extracted_text, height=300)

            # Clean the extracted text
            cleaned_text = clean_text(extracted_text)
            st.subheader("Cleaned Text")
            st.write(cleaned_text)

            # Translate the extracted text
            if ocr_languages[ocr_language] != output_languages[output_language]:
                translated_text = translate_text(cleaned_text, dest_language=output_languages[output_language])
            else:
                translated_text = cleaned_text
            st.subheader("Translated Text")
            st.text_area("Full Translated Text", translated_text, height=300)

            # Summarize the translated text
            summary = summarize_text(translated_text)
            st.subheader("Summary")
            st.write(summary)

            # Analyze sentiment
            sentiment_score = analyze_sentiment(summary)
            st.subheader("Sentiment Analysis")
            st.write(sentiment_score)

            # Add military flair and sentiment analysis
            final_text = add_military_flair(summary, sentiment_score)
            st.subheader("Final Output with Military Flair and Sentiment Analysis")
            st.write(final_text)

            # Convert the summary to speech if requested
            if play_audio:
                audio_file = text_to_speech(final_text, output_languages[output_language])
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