import os
import tempfile
from typing import Optional

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
def extract_video_id(url: str) -> Optional[str]:
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

# ------------------------ Caching ------------------------
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

# ------------------------ Text Extraction ------------------------
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

# ------------------------ Retriever ------------------------
def create_retriever(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})

# ------------------------ Prompt Template ------------------------
def generate_prompt(context: str, question: str) -> str:
    return f"""
You are a knowledgeable assistant. Answer the user's question based only on the transcript below.
If there's not enough information, say so honestly.

Transcript:
{context}

Question: {question}
"""

# ------------------------ Iterative RAG ------------------------
def iterative_rag(video_url: str, question: str, youtube_api_key: str, gemini_api_key: str,
                  max_iterations: int = 3) -> str:

    print("[INFO] Extracting video text...")
    text = get_youtube_text(video_url, youtube_api_key)

    retriever = create_retriever(text)
    llm = GeminiLLM(api_key=gemini_api_key)

    current_question = question
    last_answer = ""

    for i in range(max_iterations):
        docs = retriever.get_relevant_documents(current_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = generate_prompt(context, current_question)
        answer = llm(prompt)

        print(f"\n[Iteration {i + 1}]")
        print("Decomposed Question:", current_question)
        print("Answer:", answer)

        if answer.lower() == last_answer.lower():
            break

        last_answer = answer
        current_question = answer  # feed previous answer as next iteration's input

    print("\n✅ Final Summary:\n", last_answer)
    return last_answer

# ------------------------ Main ------------------------
if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    YOUTUBE_API_KEY = os.getenv("YOU_TUBE_API_KEY")
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not YOUTUBE_API_KEY or not GEMINI_API_KEY or not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Please set YOU_TUBE_API_KEY, GOOGLE_API_KEY, and GOOGLE_APPLICATION_CREDENTIALS")

    url = "https://www.youtube.com/watch?v=N3Tdmt1SRTM"
    q = "What is this video about?"

    iterative_rag(url, q, YOUTUBE_API_KEY, GEMINI_API_KEY)
