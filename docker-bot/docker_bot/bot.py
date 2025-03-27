import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from sqlalchemy import create_engine, Column, String, Float, select
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from streamlit.logger import get_logger

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
    embedding = Column(Vector(1536))
    initial_time = Column(Float)
    title = Column(String)
    thumbnail = Column(String)
    video_url = Column(String)
    text = Column(String)

Base.metadata.create_all(bind=engine)
    
def query_pgvector(input_text):
    session = SessionLocal()
    try:
        # Generate embedding for the input text
        embedding_response = client.embeddings.create(
            input=[input_text], model="text-embedding-3-small"
        )
        input_embedding = embedding_response.data[0].embedding
        
        # Query the database for the most similar embeddings
        #id, title, text, video_url
        query = (
            select(VideoEmbedding.id, VideoEmbedding.title, VideoEmbedding.text, VideoEmbedding.video_url, VideoEmbedding.embedding.cosine_distance(input_embedding).label('distance'))
            .order_by('distance')
            .limit(5)
        )
        results = session.execute(query).fetchall()
        return results
    except Exception as e:
        logger.error(f"Error querying pgvector: {e}")
        return []
    finally:
        session.close()


def generate_response(input_text):
    results = query_pgvector(input_text)
    context = "The following are the top 5 video transcriptions that match your query: \n"
    references = ""
    
    for result in results:
        context += f"Title: {result.title}\n"
        context += f"Transcription: {result.text}\n"
        references += f"\n - {result.video_url}\n"

    primer = """You are Q&A bot. A highly intelligent system that answers 
    user questions based on the information provided by video transcriptions. You can use your inner knowledge,
    but consider more with emphasis the information provided. Put emphasis on the transcriptions provided. If you see titles repeated, you can assume it is the same video.
    Provide samples of the transcriptions that are important to your query.
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": context},
            {"role": "user", "content": input_text},
        ],
        model="gpt-4-turbo-preview",
    )
    response = chat_completion.choices[0].message.content
    
    response += "\n Click on the following for more information: " + references
    
    return response

logger = get_logger(__name__)

styl = """
<style>
    .element-container:has([aria-label="Select RAG mode"]) {{
      position: fixed;
      bottom: 33px;
      background: white;
      z-index: 101;
    }}
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}
    textarea[aria-label="Description"] {{
        height: 200px;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

def chat_input():
    user_input = st.chat_input("What do you want to know about your videos?")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.caption("Dockerbot")
            result = generate_response(user_input)
            st.session_state["user_input"].append(user_input)
            st.session_state["generated"].append(result)
            st.rerun()

def display_chat():
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = []
    
    if st.session_state["generated"]:
        size = len(st.session_state["generated"])
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state["user_input"][i])
            with st.chat_message("assistant"):
                st.caption("Dockerbot")
                st.write(st.session_state["generated"][i])
        with st.container():
            st.write("&nbsp;")

display_chat()
chat_input()
