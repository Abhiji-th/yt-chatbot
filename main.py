from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "Gfr50f6ZBvo"
yt_api = YouTubeTranscriptApi()

try:
    transcript_list = yt_api.fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("Transcript not available for this video")
