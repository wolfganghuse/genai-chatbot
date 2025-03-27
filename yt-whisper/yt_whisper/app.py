import os
import tempfile
from pathlib import Path
from tempfile import mkdtemp

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pytube import YouTube
from pytube.exceptions import RegexMatchError
from streamlit.logger import get_logger
from yt_whisper.vtt_utils import merge_webvtt_to_list
from sqlalchemy import create_engine, Column, String, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import json

load_dotenv(".env")
logger = get_logger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))

# Database Setup
db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class VideoEmbedding(Base):
    __tablename__ = "video_embeddings"

    id = Column(String, primary_key=True)
    embedding = Column(LargeBinary)
    initial_time = Column(Float)
    title = Column(String)
    thumbnail = Column(String)
    video_url = Column(String)
    text = Column(String)

Base.metadata.create_all(bind=engine)

def store_embedding(video_data):
    session = SessionLocal()
    for entry in video_data:
        embedding_vector = np.array(entry["embedding"]).tobytes()
        record = VideoEmbedding(
            id=entry["id"],
            embedding=embedding_vector,
            initial_time=entry["initial_time"],
            title=entry["title"],
            thumbnail=entry["thumbnail"],
            video_url=entry["video_url"],
            text=entry["text"],
        )
        session.add(record)
    session.commit()
    session.close()

@st.cache_data
def process_video(video_url: str) -> dict[str, str]:
    try:
        from pytube.innertube import _default_clients

        _default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID"]

        yt_handler = YouTube(video_url)
    except RegexMatchError:
        st.error("Please enter a valid YouTube URL", icon="🚨")
        return {}

    with st.spinner("Processing your video"):
        logger.info(f"Processing video: {yt_handler.watch_url}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_stream = yt_handler.streams.filter(only_audio=True).first()
            audio_file = audio_stream.download(tmp_dir)
            file_stats = os.stat(audio_file)
            
            if file_stats.st_size > 24 * 1024 * 1024:
                st.error("Please select a shorter video, OpenAI has a 25MB limit", icon="🚨")
                return {}
            
            with open(audio_file, "rb") as audio_file:
                whisper_transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, response_format="vtt"
                )
            
            seconds_to_merge = 8
            transcript = merge_webvtt_to_list(whisper_transcript, seconds_to_merge)
            stride = 3
            video_data = []
            
            for block in range(0, len(transcript), stride):
                initial_time = transcript[block]["initial_time_in_seconds"]
                text = " ".join([t["text"] for t in transcript[block:block + stride]]).replace("\n", " ")
                id = f"{yt_handler.video_id}-t{initial_time}"
                
                video_data.append(
                    {
                        "id": id,
                        "initial_time": initial_time,
                        "title": yt_handler.title,
                        "thumbnail": yt_handler.thumbnail_url,
                        "video_url": f"{video_url}&t={initial_time}s",
                        "text": text,
                        "embedding": client.embeddings.create(
                            input=text, model="text-embedding-3-small"
                        ).data[0].embedding,
                    }
                )
                
            store_embedding(video_data)
            
            output_transcript_path = Path(st.session_state.tempfolder) / f"{yt_handler.video_id}.txt"
            with open(output_transcript_path, "w") as transcript_file:
                transcript_text = "\n".join([t["text"] for t in transcript])
                transcript_file.write(transcript_text)
            
            return {
                "video_id": yt_handler.video_id,
                "title": yt_handler.title,
                "thumbnail": yt_handler.thumbnail_url,
            }

def main():
    logger.debug("Rendering app")
    if "tempfolder" not in st.session_state:
        st.session_state.tempfolder = Path(mkdtemp(prefix="yt_transcription_"))
    if "videos" not in st.session_state:
        st.session_state.videos = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    st.header("Chat with your YouTube videos")
    st.write(
        "This app uses OpenAI's Whisper model to generate a transcription of your videos and store embeddings in PostgreSQL using pgvector."
    )

    yt_uri = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=8CY2aq3tcXA")
    if st.button(
        "Submit",
        type="primary",
        on_click=lambda: setattr(st.session_state, "processing", True),
        disabled=st.session_state.processing,
    ):
        result = process_video(yt_uri)
        st.session_state.videos.append(result)
        st.session_state.processing = False
        st.rerun()

    st.header("Processed videos:")
    for video in st.session_state.videos:
        with st.container(border=True):
            st.title(video["title"])
            st.image(video["thumbnail"], width=320)
            if st.button("Download transcription"):
                with open(
                    st.session_state.tempfolder / f"{video['video_id']}.txt"
                ) as transcript_file:
                    st.download_button(
                        f"Download transcription {video['video_id']}",
                        transcript_file,
                        file_name=video["video_id"],
                    )

if __name__ == "__main__":
    main()
