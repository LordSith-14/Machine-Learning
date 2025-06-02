# youtube_qa_assistant_app.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from pytube import YouTube
from moviepy import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from google.cloud import speech, translate_v2 as translate
from langdetect import detect
from functools import lru_cache
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# ------------------------ Load .env ------------------------
load_dotenv()

# ------------------------ Gemini LLM Setup ------------------------
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
        if query.path.startswith('/embed/') or query.path.startswith('/v/'):
            return query.path.split('/')[2]
    return None

@lru_cache(maxsize=10)
def cached_transcription(video_url: str, language_code: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True, file_extension="mp4").first()
        file_path = stream.download(output_path=tmpdir, filename="audio.mp4")

        audio_wav_path = os.path.join(tmpdir, "audio.wav")
        video_clip = VideoFileClip(file_path)
        video_clip.audio.write_audiofile(audio_wav_path, fps=16000)
        video_clip.close()

        return transcribe_with_google(audio_wav_path, language_code)

def transcribe_with_google(audio_path: str, language_code: str) -> str:
    speech_client = speech.SpeechClient()
    translate_client = translate.Client()

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True
    )

    try:
        response = speech_client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        if language_code.startswith("hi") or detect(transcript) == "hi":
            translation = translate_client.translate(transcript, target_language="en")
            return translation["translatedText"]
        return transcript
    except Exception as e:
        return f"[Speech Recognition Error] {str(e)}"

def get_youtube_text(video_url: str, youtube_api_key: str, language_code: str) -> str:
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL or could not extract video ID.")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code.split('-')[0]])
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

        return cached_transcription(video_url, language_code)

def add_multiple_videos_to_vectorstore(video_urls, youtube_api_key, vectorstore_path="faiss_index", lang_choice="Auto-Detect"):
    lang_map = {"Auto-Detect": "en-US", "Hindi": "hi-IN", "English": "en-US"}
    language_code = lang_map.get(lang_choice, "en-US")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.load_local(vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True) if os.path.exists(vectorstore_path) else None

    for video_url in video_urls:
        text = get_youtube_text(video_url, youtube_api_key, language_code)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([text])
        for chunk in chunks:
            chunk.metadata["video_url"] = video_url

        if vector_store:
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(vectorstore_path)
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})

def generate_detailed_summary(llm, text_chunks, question):
    chunk_summaries = [llm(f"""You are an expert assistant. Based on the content below, give a detailed summary for the user's question.

Content:
{chunk.page_content}

Question: {question}
""") for chunk in text_chunks]

    combined = "\n\n".join(chunk_summaries)
    return llm(f"""You are an expert summarizer. Merge and elaborate on the following chunk-level summaries into a complete, coherent explanation:

{combined}

Question: {question}
""")

def iterative_rag_multiple_detailed(video_urls, question, youtube_api_key, gemini_api_key, lang_choice="Auto-Detect"):
    lang_map = {"Auto-Detect": "en-US", "Hindi": "hi-IN", "English": "en-US"}
    language_code = lang_map.get(lang_choice, "en-US")

    all_docs = []
    for video_url in video_urls:
        text = get_youtube_text(video_url, youtube_api_key, language_code)
        docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([text])
        all_docs.extend(docs)

    return generate_detailed_summary(GeminiLLM(gemini_api_key), all_docs, question)

# ------------------------ Streamlit UI ------------------------
st.set_page_config(page_title="YouTube Q&A Assistant", layout="wide")

st.title("🎥 YouTube Q&A Assistant with FAISS Viewer")
st.caption("Powered by LangChain, FAISS, and Gemini LLM")

# Load API keys
youtube_api_key = os.getenv("YOU_TUBE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not youtube_api_key or not gemini_api_key or not google_credentials:
    st.error("❌ Missing API keys or credentials. Please set them in your `.env` file.")
    st.stop()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

# Tabs
tab1, tab2 = st.tabs(["🔎 Ask Video Questions", "📂 Browse FAISS Store"])

# ------------------------ Tab 1 ------------------------
with tab1:
    st.subheader("🔍 Analyze YouTube Videos using AI")

    video_urls_input = st.text_area("Enter YouTube video URLs (one per line):")
    question_input = st.text_input("Enter your question:")
    language_choice = st.selectbox("Select video language:", ["Auto-Detect", "Hindi", "English"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Add Videos to FAISS Store"):
            urls = [u.strip() for u in video_urls_input.splitlines() if u.strip()]
            if not urls:
                st.warning("Please enter at least one YouTube URL.")
            else:
                with st.spinner("Processing..."):
                    retriever = add_multiple_videos_to_vectorstore(urls, youtube_api_key, lang_choice=language_choice)
                st.success("✅ Videos added to vector store.")

    with col2:
        if st.button("Get Detailed Answer"):
            urls = [u.strip() for u in video_urls_input.splitlines() if u.strip()]
            if not urls or not question_input:
                st.warning("Enter both video URLs and your question.")
            else:
                with st.spinner("Generating answer..."):
                    answer = iterative_rag_multiple_detailed(
                        urls, question_input, youtube_api_key, gemini_api_key, lang_choice=language_choice
                    )
                st.markdown("### Answer:")
                st.write(answer)

# ------------------------ Tab 2 ------------------------
with tab2:
    st.subheader("📂 Explore FAISS Store")
    if st.button("Load Store"):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )
        path = "faiss_index"
        if os.path.exists(path):
            vs = FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
            retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
            st.success("✅ Store loaded.")

            query = st.text_input("Enter a query to search:")
            if query:
                results = retriever.get_relevant_documents(query)
                for i, doc in enumerate(results, 1):
                    st.markdown(f"**Result {i}:**")
                    st.write(doc.page_content[:1000])
                    st.caption(f"_Source: {doc.metadata.get('video_url', 'N/A')}_")
        else:
            st.warning("No FAISS index found. Add videos first.")
