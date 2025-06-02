import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from pytube import YouTube
from moviepy import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from google.cloud import speech, translate_v2 as translate
from langdetect import detect
from functools import lru_cache

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai

# ------------------------ Load .env ------------------------
load_dotenv()

# ------------------------ LLM Setup ------------------------
class GeminiLLM:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-04-17"):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=model_name)

    def __call__(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[LLM Error] {str(e)}"

# ------------------------ Utility Functions ------------------------
def extract_video_id(url: str):
    from urllib.parse import urlparse, parse_qs
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query).get('v', [None])[0]
        if query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        if query.path.startswith('/v/'):
            return query.path.split('/')[2]
    return None

@lru_cache(maxsize=10)
def cached_transcription(video_url: str, audio_path: str) -> str:
    return transcribe_with_google(audio_path)

def transcribe_with_google(audio_path: str) -> str:
    speech_client = speech.SpeechClient()
    translate_client = translate.Client()

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="hi-IN",
        enable_automatic_punctuation=True
    )

    try:
        response = speech_client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        if detect(transcript) == "hi":
            translation = translate_client.translate(transcript, target_language="en")
            return translation["translatedText"]
        return transcript
    except Exception as e:
        return f"[Speech Recognition Error] {str(e)}"

def get_youtube_text(video_url: str, youtube_api_key: str) -> str:
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL or could not extract video ID.")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        try:
            from googleapiclient.discovery import build
            youtube = build('youtube', 'v3', developerKey=youtube_api_key)
            response = youtube.videos().list(part="snippet", id=video_id).execute()
            if response['items']:
                snippet = response['items'][0]['snippet']
                title = snippet.get('title', '')
                description = snippet.get('description', '')
                return f"Title: {title}\nDescription: {description}"
        except Exception:
            pass

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yt = YouTube(video_url)
                stream = yt.streams.filter(only_audio=False, file_extension="mp4").first()
                file_path = stream.download(output_path=tmpdir, filename="video.mp4")
                video = VideoFileClip(file_path)
                audio_path = os.path.join(tmpdir, "temp_audio.wav")
                video.audio.write_audiofile(audio_path, fps=16000)
                video.close()
                return cached_transcription(video_url, audio_path)
        except Exception as e:
            return f"[Download/Transcription Error] {str(e)}"

def create_retriever(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})

def generate_prompt(context: str, question: str) -> str:
    return f"""
You are a knowledgeable assistant. Answer the user's question based only on the transcript below.
If there's not enough information, say so honestly.

Transcript:
{context}

Question: {question}
"""

def iterative_rag(video_url: str, question: str, youtube_api_key: str, gemini_api_key: str,
                  max_iterations: int = 3) -> str:
    text = get_youtube_text(video_url, youtube_api_key)
    retriever = create_retriever(text)
    llm = GeminiLLM(api_key=gemini_api_key)

    current_question = question
    last_answer = ""

    for _ in range(max_iterations):
        docs = retriever.get_relevant_documents(current_question)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = generate_prompt(context, current_question)
        answer = llm(prompt)

        if answer.lower() == last_answer.lower():
            break
        last_answer = answer
        current_question = answer

    return last_answer

# ------------------------ Streamlit UI ------------------------
def main():
    st.set_page_config(page_title="YouTube Q&A with Gemini", layout="wide")
    st.title("🎥 YouTube Video Q&A using Gemini + LangChain")

    # Load from .env only
    youtube_api_key = os.getenv("YOU_TUBE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not youtube_api_key or not gemini_api_key or not google_credentials:
        st.error("Missing API keys or credentials. Please set them in your .env file.")
        return

    # Required for speech-to-text
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

    video_url = st.text_input("Enter YouTube video URL:")
    question = st.text_input("Ask a question about the video:")

    if st.button("Submit") and video_url and question:
        with st.spinner("Processing video and generating response..."):
            try:
                result = iterative_rag(video_url, question, youtube_api_key, gemini_api_key)
                st.success("✅ Answer Generated:")
                st.write(result)
            except Exception as e:
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
