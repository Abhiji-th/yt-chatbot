from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

video_id = "Gfr50f6ZBvo"

yt_api = YouTubeTranscriptApi()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

llm = GoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})

question = "What is valorant"

retrieved_docs = retriever.invoke(question)

context = "\n\n".join(doc.page_content for doc in retrieved_docs)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"]
)

final_prompt = prompt.invoke({"context": context, "question": question})

result = llm.invoke(final_prompt)

print(result)
