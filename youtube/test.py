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
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# ------------------------ Global clients ------------------------
# Create Google clients once and reuse for efficiency
speech_client = speech.SpeechClient()
translate_client = translate.Client()

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
def cached_transcription(video_url: str, language_code: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            yt = YouTube(video_url)
            stream = yt.streams.filter(only_audio=True, file_extension="mp4").first()
            file_path = stream.download(output_path=tmpdir, filename="audio.mp4")

            audio_wav_path = os.path.join(tmpdir, "audio.wav")
            video_clip = VideoFileClip(file_path)
            video_clip.audio.write_audiofile(audio_wav_path, fps=16000)
            video_clip.close()

            return transcribe_with_google(audio_wav_path, language_code)
        except Exception as e:
            return f"[Audio Extraction Error] {str(e)}"

def transcribe_with_google(audio_path: str, language_code: str) -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_code,
            enable_automatic_punctuation=True
        )

        response = speech_client.recognize(config=config, audio=audio)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])

        # Language detection fallback - you can extend lang_map here
        detected_lang = detect(transcript) if transcript else ""
        if language_code.startswith("hi") or detected_lang == "hi":
            translation = translate_client.translate(transcript, target_language="en")
            return translation.get("translatedText", transcript)
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
        # fallback to YouTube API snippet title & description
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

        # fallback to audio transcription
        return cached_transcription(video_url, language_code)

def add_multiple_videos_to_vectorstore(video_urls, youtube_api_key, vectorstore_path="faiss_index", lang_choice="Auto-Detect"):
    lang_map = {
        "Auto-Detect": "en-US",
        "Hindi": "hi-IN",
        "English": "en-US"
    }
    language_code = lang_map.get(lang_choice, "en-US")

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(vectorstore_path):
        vector_store = FAISS.load_local(
            vectorstore_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vector_store = None

    for video_url in video_urls:
        with st.spinner(f"Processing video: {video_url}"):
            text = get_youtube_text(video_url, youtube_api_key, language_code)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([text])

            docs_with_metadata = []
            for chunk in chunks:
                new_metadata = chunk.metadata.copy() if chunk.metadata else {}
                new_metadata["video_url"] = video_url
                docs_with_metadata.append(Document(page_content=chunk.page_content, metadata=new_metadata))

            if vector_store:
                vector_store.add_documents(docs_with_metadata)
            else:
                vector_store = FAISS.from_documents(docs_with_metadata, embeddings)

    vector_store.save_local(vectorstore_path)
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})

# (rest of your unchanged code below...)

# You can also consider adding multiprocessing or async calls to speed up multiple video processing if needed.

# Streamlit UI code continues here as before...


def generate_prompt(context: str, question: str) -> str:
    return f"""
You are a knowledgeable assistant. Answer the user's question based only on the transcript below.
If there's not enough information, say so honestly.

Transcript:
{context}

Question: {question}
"""

# --------- NEW detailed summarization function ---------
def generate_detailed_summary(llm, text_chunks, question):
    chunk_summaries = []
    for idx, chunk in enumerate(text_chunks):
        prompt = f"""
You are an expert assistant. Based on the following content, provide a detailed and elaborative summary explaining all key points:

Content:
{chunk.page_content}

Question: {question}

Please provide a detailed summary.
"""
        summary = llm(prompt)
        chunk_summaries.append(summary)

    combined_summary_text = "\n\n".join(chunk_summaries)

    final_prompt = f"""
You are an expert assistant. Based on the following detailed summaries from multiple parts of videos, write a comprehensive, cohesive, and thorough explanation covering all key points:

{combined_summary_text}

Question: {question}

Please provide a detailed and elaborative explanation.
"""
    final_summary = llm(final_prompt)
    return final_summary

# --------- UPDATED iterative_rag_multiple to use detailed summary ---------
def iterative_rag_multiple_detailed(video_urls, question, youtube_api_key, gemini_api_key, lang_choice="Auto-Detect"):
    lang_map = {
        "Auto-Detect": "en-US",
        "Hindi": "hi-IN",
        "English": "en-US"
    }
    language_code = lang_map.get(lang_choice, "en-US")

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load or create vector store
    vectorstore_path = "faiss_index"
    if os.path.exists(vectorstore_path):
        vector_store = FAISS.load_local(
            vectorstore_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vector_store = None

    # For all videos, get full transcripts (no vector store for this function)
    all_docs = []
    for video_url in video_urls:
        text = get_youtube_text(video_url, youtube_api_key, language_code)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([text])
        all_docs.extend(docs)

    llm = GeminiLLM(api_key=gemini_api_key)
    detailed_answer = generate_detailed_summary(llm, all_docs, question)

    return detailed_answer

# ------------------------ Streamlit UI ------------------------
st.set_page_config(page_title="YouTube Q&A Assistant", layout="wide")
st.markdown(
    """
    <style>
    .big-font {
        font-size:22px !important;
        font-weight:600;
    }
    .small-font {
        font-size:14px !important;
        color: gray;
    }
    .stTextInput > label {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎥 YouTube Q&A Assistant with FAISS Viewer")
st.caption("Interact with YouTube videos using LLM + Vector Search")
youtube_api_key = os.getenv("YOU_TUBE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
# Sidebar: Input URLs and language selection
with st.sidebar:
    st.header("🎬 Video Inputs")
    video_urls_input = st.text_area(
        "Enter YouTube video URLs (one per line):",
        placeholder="https://www.youtube.com/watch?v=xxxxxxx\nhttps://youtu.be/yyyyyyy"
    )
    language_choice = st.selectbox("Select the video language:", ["Auto-Detect", "Hindi", "English"])

    st.markdown("---")
    if st.button("Add Videos to FAISS Store & Build Embeddings"):
        if not video_urls_input.strip():
            st.warning("Please enter at least one YouTube video URL.")
        else:
            with st.spinner("Processing videos and updating FAISS vector store..."):
                video_urls = [url.strip() for url in video_urls_input.splitlines() if url.strip()]
                retriever = add_multiple_videos_to_vectorstore(video_urls, youtube_api_key, lang_choice=language_choice)
            st.success("✅ Videos processed and added to vector store!")

# Main center area: question input and answer display
st.subheader("🔍 Ask Questions About Videos")
question_input = st.text_input("Enter your question:")

if st.button("Get Detailed Answer for Question"):
    if not video_urls_input.strip() or not question_input.strip():
        st.warning("Please enter video URLs in the sidebar and a question here.")
    else:
        with st.spinner("Generating detailed answer, please wait..."):
            video_urls = [url.strip() for url in video_urls_input.splitlines() if url.strip()]
            answer = iterative_rag_multiple_detailed(
                video_urls,
                question_input,
                youtube_api_key,
                gemini_api_key,
                lang_choice=language_choice
            )
        st.markdown("### Detailed Answer:")
        st.write(answer)

# Tab 2 remains unchanged or can also be moved somewhere as needed
tab2 = st.expander("📂 Browse FAISS Vector Store")

with tab2:
    if st.button("Load FAISS Store"):
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore_path = "faiss_index"
        if os.path.exists(vectorstore_path):
            vector_store = FAISS.load_local(
                vectorstore_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":4, "fetch_k":10})
            st.success("✅ FAISS vector store loaded.")

            query = st.text_input("Enter query to search in vector store:")
            if query:
                results = retriever.get_relevant_documents(query)
                for idx, doc in enumerate(results):
                    st.markdown(f"**Result {idx+1}:**")
                    st.write(doc.page_content[:1000])
                    st.write(f"_Source URL:_ {doc.metadata.get('video_url', 'Unknown')}")
        else:
            st.warning("No FAISS vector store found. Please add videos first.")

