from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

video_id = "Gfr50f6ZBvo"
yt_api = YouTubeTranscriptApi()

try:
    transcript_list = yt_api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("Transcript not available for this video")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))
