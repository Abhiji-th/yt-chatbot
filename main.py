from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

video_id = "Gfr50f6ZBvo"
yt_api = YouTubeTranscriptApi()
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

try:
    transcript_list = yt_api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("Transcript not available for this video")

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
chunks = splitter.create_documents([transcript])

# vectorstore = FAISS.from_documents(chunks, embeddings)

# vectorstore.save_local("faiss_index")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print(vectorstore.index_to_docstore_id)
