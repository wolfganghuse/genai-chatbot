import os
import argparse

import whisper

import numpy as np
import ffmpeg
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, Column, String, LargeBinary, Float
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import tempfile
from yt_whisper.vtt_utils import merge_webvtt_to_list

load_dotenv(".env")

client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))

# Whisper Setup
model = whisper.load_model("base")

# Database Setup
db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class VideoEmbedding(Base):
    __tablename__ = "video_embeddings"

    id = Column(String, primary_key=True)
    embedding = Column(Vector(1536))
    initial_time = Column(Float)
    title = Column(String)
    thumbnail = Column(String)
    video_url = Column(String)
    text = Column(String)

Base.metadata.create_all(bind=engine)

def store_embedding(video_data):
    session = SessionLocal()
    for entry in video_data:
        embedding_vector=entry["embedding"]
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

def process_video(video_url: str) -> dict[str, str]:
    
    video_title = video_url.rsplit("/", 1)[0]
    print(f"Processing video file: {video_title}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_file = os.path.join(tmp_dir, "audio.mp3")
        ffmpeg.input(video_url).output(audio_file, format="mp3", acodec="libmp3lame").run()
        
        with open(audio_file, "rb") as audio_file:
            whisper_transcript = model.transcribe(audio_file)

            #whisper_transcript = client.audio.transcriptions.create(
            #    model="whisper-1", file=audio_file, response_format="vtt"
            #)
        
        seconds_to_merge = 8
        transcript = merge_webvtt_to_list(whisper_transcript, seconds_to_merge)
        stride = 3
        video_data = []
        
        for block in range(0, len(transcript), stride):
            initial_time = transcript[block]["initial_time_in_seconds"]
            text = " ".join([t["text"] for t in transcript[block:block + stride]]).replace("\n", " ")
            id = f"video-t{initial_time}" #need to change this to a unique id
            
            video_data.append(
                {
                    "id": id,
                    "initial_time": initial_time,
                    "title": video_title,
                    "thumbnail": "",  # No thumbnail for local files
                    "video_url": f"{video_url}#t={initial_time}", 
                    "text": text,
                    "embedding": client.embeddings.create(
                        input=text, model="text-embedding-3-small"
                    ).data[0].embedding,
                }
            )
        
        store_embedding(video_data)
        
        output_transcript_path = Path("transcription.txt")
        with open(output_transcript_path, "w") as transcript_file:
            transcript_text = "\n".join([t["text"] for t in transcript])
            transcript_file.write(transcript_text)
        
        return {"title": "Local Video"}

def main():
    parser = argparse.ArgumentParser(description="Process a local MP4 video file.")
    parser.add_argument("video_path", type=str, help="Path to the local MP4 file")
    args = parser.parse_args()
    
    
    result = process_video(args.video_path)
    print(f"Processing complete: {result}")

if __name__ == "__main__":
    main()
