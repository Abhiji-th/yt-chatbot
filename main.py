from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

video_id = "Gfr50f6ZBvo"

yt_api = YouTubeTranscriptApi()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

llm = GoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.5
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

query = "What is Deepmind"

def format_docs(retrieved_docs):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {query}
    """,
    input_variables=["context", "query"]
)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "query": RunnablePassthrough()
})

chain = parallel_chain | prompt | llm

result = chain.invoke(query)

print(result)
