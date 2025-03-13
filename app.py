import streamlit as st
from PIL import Image
import pytesseract
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import pyttsx3
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
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

# Function to clean the extracted text
def clean_text(text):
    try:
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a single space
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters
        return text.strip()
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        return ""

# Function to summarize the text
def summarize_text(text, sentences_count=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return ""

# Function to extract keywords
def extract_keywords(text, top_n=5):
    try:
        vectorizer = CountVectorizer(stop_words='english')
        word_count_matrix = vectorizer.fit_transform([text])
        word_counts = word_count_matrix.sum(axis=0)
        word_frequencies = [(word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        sorted_word_frequencies = sorted(word_frequencies, key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_word_frequencies[:top_n]]
        return keywords
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")
        return []

# Function to analyze sentiment
def analyze_sentiment(text):
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)
        return sentiment_score
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return {}

# Function to add military flair and sentiment analysis
def add_military_flair(text, sentiment_score):
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
def text_to_speech(text, output_file="output_audio.mp3"):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        return output_file
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

# Streamlit app
def main():
    st.title("Automated Newspaper Summarization and Analysis")
    st.write("Upload a newspaper image to extract, summarize, and analyze the text.")

    # Upload image
    uploaded_file = st.file_uploader("Choose a newspaper image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Newspaper Image", use_column_width=True)

        # Extract text
        st.write("Extracting text from the image...")
        extracted_text = extract_text_from_image(image)
        if extracted_text:
            st.subheader("Extracted Text")
            st.write(extracted_text)

            # Clean text
            cleaned_text = clean_text(extracted_text)
            st.subheader("Cleaned Text")
            st.write(cleaned_text)

            # Summarize text
            st.write("Summarizing the text...")
            summary = summarize_text(cleaned_text)
            if summary:
                st.subheader("Summarized Text")
                st.write(summary)

                # Analyze sentiment
                st.write("Analyzing sentiment...")
                sentiment_score = analyze_sentiment(summary)
                st.subheader("Sentiment Analysis")
                st.write(sentiment_score)

                # Extract keywords
                st.write("Extracting keywords...")
                keywords = extract_keywords(summary)
                st.subheader("Top Keywords")
                st.write(keywords)

                # Add military flair and sentiment analysis
                final_text = add_military_flair(summary, sentiment_score)
                st.subheader("Final Output with Military Flair and Sentiment Analysis")
                st.write(final_text)

                # Convert to speech
                st.write("Converting text to speech...")
                audio_file = text_to_speech(final_text)
                if audio_file:
                    st.subheader("Audio Output")
                    st.audio(audio_file, format="audio/mp3", autoplay=True)  # Autoplay the audio

                    # Provide a one-click download button
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name="output_audio.mp3",
                        mime="audio/mp3"
                    )
        else:
            st.error("No text extracted from the image. Please try another image.")

if __name__ == "__main__":
    main()
