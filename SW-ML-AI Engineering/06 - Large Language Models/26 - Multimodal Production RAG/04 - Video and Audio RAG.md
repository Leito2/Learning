# 🎯 04 - Video and Audio RAG — Whisper, Frame Extraction, and Real-Time Video Understanding

> **The modalities beyond text and images. Voice messages, meeting recordings, training videos. Production pipelines for transcription, frame extraction, and multimodal retrieval.**

## 🎯 Learning Objectives
- Transcribe audio with Whisper (large-v3-turbo for production)
- Build audio RAG with vector search over transcripts
- Extract key frames from video and embed with CLIP
- Implement real-time video understanding with GPT-4o vision
- Build a video RAG for meetings, lectures, or surveillance
- Apply to portfolio projects (StayBot voice search, MARS video analysis)
- Use Gemini Live for real-time video question answering

## Introduction

Audio and video are the **fastest-growing enterprise data modalities** in 2026:

- **Voicemails** and **call recordings** in customer service
- **Meeting recordings** in sales and product management
- **Training videos** in education and onboarding
- **Surveillance video** in retail and security
- **Voice messages** in customer-facing applications

This note covers:

1. **Audio RAG** — Whisper transcription + text-based retrieval
2. **Video RAG** — frame extraction + multimodal embeddings
3. **Real-time video** — GPT-4o vision + Gemini Live for live streams

By the end, you can build a multimodal RAG that handles voice messages, video content, and real-time video streams.

![Audio + Video RAG architecture](https://example.com/audio-video-rag.png)

---

## 1. Audio RAG — Whisper Transcription

The canonical pipeline:

```
Audio file → Whisper transcription → Text chunks → Embeddings → Vector DB
```

### 1.1 Whisper transcription

```bash
pip install openai-whisper
```

```python
import whisper


def transcribe_audio(audio_path: str, language: str = "en") -> dict:
    """Transcribe audio with Whisper, including timestamps."""
    
    model = whisper.load_model("large-v3-turbo")  # or "medium", "small" for faster
    
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,  # word-level timestamps for chunking
        verbose=False,
    )
    
    # Extract segments with timestamps
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })
    
    return {
        "language": result["language"],
        "duration": result["duration"],
        "full_text": result["text"],
        "segments": segments,
    }
```

For production at scale, use **Whisper large-v3-turbo** (faster) or **faster-whisper** (CTranslate2-based, 4× faster).

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel


def transcribe_fast(audio_path: str) -> list[dict]:
    """Transcribe with faster-whisper (4× faster, lower memory)."""
    
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
    )
    
    chunks = []
    for seg in segments:
        chunks.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words],
        })
    
    return chunks
```

### 1.2 Chunking transcribed audio

For RAG, chunk the transcript by topic or by time:

```python
def chunk_transcript(segments: list[dict], chunk_duration_sec: int = 30) -> list[dict]:
    """Chunk transcript by time intervals."""
    
    chunks = []
    current_chunk = {"start": segments[0]["start"], "end": 0, "text": ""}
    
    for seg in segments:
        if seg["end"] - current_chunk["start"] > chunk_duration_sec:
            # Finalize current chunk
            chunks.append(current_chunk)
            current_chunk = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        else:
            current_chunk["end"] = seg["end"]
            current_chunk["text"] += " " + seg["text"]
    
    if current_chunk["text"]:
        chunks.append(current_chunk)
    
    return chunks
```

For **speaker-aware chunking** (using pyannote-audio for diarization):

```python
from pyannote.audio import Pipeline


def transcribe_with_speakers(audio_path: str) -> list[dict]:
    """Transcribe with speaker labels."""
    
    # Diarization (speaker identification)
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization = diarization_pipeline(audio_path)
    
    # Whisper transcription
    transcript = transcribe_fast(audio_path)
    
    # Combine: each word/segment gets a speaker label
    chunks = []
    for seg in transcript:
        speaker = get_speaker_for_time(diarization, seg["start"], seg["end"])
        chunks.append({
            **seg,
            "speaker": speaker,
        })
    
    return chunks
```

---

## 2. Audio RAG Pipeline

```python
class AudioRAGPipeline:
    """RAG over audio content using Whisper + vector search."""
    
    def __init__(self):
        self.transcriber = WhisperModel("large-v3-turbo")
        self.embedder = ...  # SentenceTransformer or OpenAI embed
        self.vector_db = QdrantClient(url="http://localhost:6333")
    
    def index(self, audio_path: str, audio_id: str):
        """Transcribe and index audio."""
        
        # 1. Transcribe
        segments = transcribe_fast(audio_path)
        
        # 2. Chunk
        chunks = chunk_transcript(segments, chunk_duration_sec=30)
        
        # 3. Embed and store
        points = []
        for chunk_id, chunk in enumerate(chunks):
            embedding = self.embedder.encode(chunk["text"])
            points.append(PointStruct(
                id=f"{audio_id}#{chunk_id}",
                vector=embedding.tolist(),
                payload={
                    "audio_id": audio_id,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                    "speaker": chunk.get("speaker", "unknown"),
                },
            ))
        
        self.vector_db.upsert(
            collection_name="audio_rag",
            points=points,
        )
    
    def query(self, question: str, top_k: int = 5) -> list[dict]:
        """Search across indexed audio."""
        
        query_embedding = self.embedder.encode(question)
        
        results = self.vector_db.search(
            collection_name="audio_rag",
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )
        
        return [{
            "text": r.payload["text"],
            "audio_id": r.payload["audio_id"],
            "start_time": r.payload["start"],
            "end_time": r.payload["end"],
            "speaker": r.payload["speaker"],
            "score": r.score,
        } for r in results]
```

### 2.1 Direct audio embeddings — CLAP

For audio that doesn't have text (music, ambient sounds), use **CLAP** (Contrastive Language-Audio Pretraining):

```python
from transformers import ClapModel, ClapProcessor
import librosa


def embed_audio(audio_path: str) -> torch.Tensor:
    """CLAP embedding for audio."""
    
    model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    
    audio, sr = librosa.load(audio_path, sr=48000)
    inputs = processor(audios=[audio], return_tensors="pt", sampling_rate=sr)
    
    with torch.no_grad():
        audio_embedding = model.get_audio_features(**inputs)
    
    return audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)


def embed_audio_query(query: str) -> torch.Tensor:
    """Text query for CLAP — enables 'find me sound of rain' searches."""
    model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    
    return text_embedding / text_embedding.norm(dim=-1, keepdim=True)
```

CLAP enables queries like:
- "Find the part of the meeting where we discussed the budget" → finds audio segment
- "Find sounds of applause" → finds applause moments
- "Find music in major key" → finds matching music

---

## 3. Video RAG — Frame Extraction

For video, the standard approach:

```
Video → Extract keyframes → CLIP embed each frame → Text-embed transcript → Multimodal search
```

### 3.1 Frame extraction

```python
import cv2
from PIL import Image
import scenedetect


def extract_keyframes(video_path: str, fps_sample: int = 1) -> list[Image.Image]:
    """Extract keyframes from a video at sampled FPS."""
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every fps_sample frames
        if frame_count % fps_sample == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        frame_count += 1
    
    cap.release()
    return frames


def extract_scene_changes(video_path: str) -> list[Image.Image]:
    """Extract frames at scene changes (PySceneDetect)."""
    
    from scenedetect import open_video, SceneManager, ContentDetector
    
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    
    scene_list = scene_manager.get_scene_list()
    
    frames = []
    for scene in scene_list:
        start, end = scene
        # Extract frame at the middle of each scene
        timestamp = (start.get_seconds() + end.get_seconds()) / 2
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
    
    return frames
```

### 3.2 Video index with CLIP embeddings

```python
def index_video(video_path: str, video_id: str):
    """Build a video index with frames + transcript."""
    
    # 1. Extract audio + transcribe
    audio_path = f"/tmp/{video_id}.wav"
    extract_audio(video_path, audio_path)
    transcript_segments = transcribe_fast(audio_path)
    
    # 2. Extract keyframes
    keyframes = extract_scene_changes(video_path)[:50]  # max 50 frames per video
    
    # 3. Embed and store
    client = QdrantClient(url="http://localhost:6333")
    points = []
    
    # Frames
    for i, frame in enumerate(keyframes):
        frame_embedding = embed_image(frame)
        points.append(PointStruct(
            id=f"{video_id}#frame{i}",
            vector=frame_embedding.tolist()[0],
            payload={
                "type": "frame",
                "video_id": video_id,
                "frame_index": i,
                "timestamp": i * 30,  # approximate
            },
        ))
    
    # Transcript chunks
    chunks = chunk_transcript(transcript_segments, chunk_duration_sec=30)
    for chunk_id, chunk in enumerate(chunks):
        chunk_embedding = embed_text(chunk["text"])
        points.append(PointStruct(
            id=f"{video_id}#chunk{chunk_id}",
            vector=chunk_embedding.tolist()[0],
            payload={
                "type": "transcript",
                "video_id": video_id,
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
            },
        ))
    
    client.upsert(
        collection_name="video_rag",
        points=points,
    )
```

### 3.3 Query video RAG

```python
def query_video_rag(question: str, top_k: int = 5) -> dict:
    """Query over video content."""
    
    # 1. Text-based search over transcript
    text_results = client.search(
        collection_name="video_rag",
        query_vector=embed_text(question).tolist()[0],
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="transcript"))]),
        limit=top_k,
    )
    
    # 2. For each relevant transcript chunk, get nearby frames
    response = {"text_matches": [], "video_segments": []}
    
    for text_match in text_results:
        video_id = text_match.payload["video_id"]
        chunk_text = text_match.payload["text"]
        start_time = text_match.payload["start"]
        end_time = text_match.payload["end"]
        
        response["text_matches"].append({
            "text": chunk_text,
            "video_id": video_id,
            "start": start_time,
            "end": end_time,
            "score": text_match.score,
        })
        
        # Get frames near this transcript
        frame_results = client.search(
            collection_name="video_rag",
            query_vector=embed_image_query(question).tolist()[0],
            query_filter=Filter(must=[
                FieldCondition(key="video_id", match=MatchValue(value=video_id)),
                FieldCondition(key="type", match=MatchValue(value="frame")),
            ]),
            limit=3,
        )
        
        response["video_segments"].append({
            "video_id": video_id,
            "start": start_time,
            "end": end_time,
            "frames": [{"timestamp": r.payload["timestamp"], "score": r.score} for r in frame_results],
        })
    
    return response
```

The user gets: relevant transcript chunks (text) + relevant frames (images, timestamps) — both can be played back in the original video.

---

## 4. Real-time Video Understanding with GPT-4o / Gemini Live

For **live video streams** (security cameras, online classes, livestreams), you need real-time multimodal inference.

### 4.1 GPT-4o vision for video frames

```python
from openai import OpenAI
import base64


def analyze_frame(frame_path: str, question: str) -> str:
    """Use GPT-4o vision to analyze a single video frame."""
    
    client = OpenAI()
    
    with open(frame_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    
    return response.choices[0].message.content


# Process a live video stream
def monitor_live_stream(stream_url: str, alert_callback):
    """Monitor a live video stream with GPT-4o vision."""
    
    cap = cv2.VideoCapture(stream_url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample 1 frame per second
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image.save("/tmp/frame.jpg")
        
        # Ask GPT-4o about the scene
        description = analyze_frame("/tmp/frame.jpg", "What's happening in this scene?")
        
        # Trigger alerts based on description
        if "fire" in description.lower() or "smoke" in description.lower():
            alert_callback(description)
        
        time.sleep(1)  # 1 FPS for cost control
```

### 4.2 Gemini Live for native video

Gemini 1.5 / 2.0 processes video natively (no frame extraction needed):

```python
import google.generativeai as genai


def analyze_video_native(video_path: str, question: str) -> str:
    """Native video understanding with Gemini."""
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    # Upload the video file
    video_file = genai.upload_file(video_path)
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = genai.get_file(video_file.name)
    
    # Generate response
    response = model.generate_content([
        video_file,
        question,
    ])
    
    return response.text


# Usage
result = analyze_video_native(
    "meeting.mp4",
    "When did the team discuss the budget?"
)
# "The budget was discussed between 12:34 and 15:22, where Alice said..."
```

Gemini is the only model with **native video understanding** — no frame extraction or preprocessing. Handles hours of video in a single context window.

---

## 5. Case Study — Customer Support Voice Bot

For customer service with voice messages:

```python
class CustomerSupportVoiceBot:
    """Process voice messages from customers."""
    
    def __init__(self):
        self.whisper = WhisperModel("large-v3-turbo")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_db = QdrantClient(url="http://localhost:6333")
    
    def process_voice_message(self, audio_path: str) -> dict:
        """Process a customer's voice message."""
        
        # 1. Transcribe
        segments = transcribe_fast(audio_path)
        
        # 2. Detect intent via transcript + vector search
        full_text = " ".join(s["text"] for s in segments)
        query_embedding = self.embedder.encode(full_text)
        
        # Find similar past issues in the FAQ knowledge base
        similar_issues = self.vector_db.search(
            collection_name="faq",
            query_vector=query_embedding.tolist(),
            limit=3,
        )
        
        # 3. Generate response via LLM
        response = llm.complete(
            f"Customer said (transcribed from voice): {full_text}\n\n"
            f"Similar past issues:\n{format_issues(similar_issues)}\n\n"
            f"Generate a helpful response:"
        )
        
        return {
            "transcription": full_text,
            "speaker_segments": segments,
            "similar_issues": similar_issues,
            "response": response,
        }
    
    def generate_voice_response(self, text: str) -> bytes:
        """Generate a voice response using TTS."""
        from openai import OpenAI
        client = OpenAI()
        
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice="alloy",
            input=text,
        )
        
        return response.read()
```

The flow:
1. Customer sends voice message
2. Whisper transcribes to text
3. Vector search finds similar past issues
4. LLM generates a text response
5. TTS converts back to voice
6. Customer hears the response

---

## 6. The StayBot Voice Search

For your **StayBot**:

```python
def staybot_voice_search(audio_path: str) -> list[dict]:
    """Voice-based Airbnb listing search."""
    
    # 1. Transcribe query
    segments = transcribe_fast(audio_path)
    query_text = " ".join(s["text"] for s in segments)
    
    # 2. Search listings
    text_results = text_search(query_text, top_k=5)
    image_results = image_search(query_text, top_k=5)
    
    # 3. Combine and return
    return hybrid_multimodal_search(text_results, image_results)
```

The user says "Show me a cozy apartment near the beach with Wi-Fi"; the bot returns matching listings (with images).

---

## 7. Antipatterns

### 7.1 Antipattern 1: Using Whisper base for production

```python
# ❌ Base model is too inaccurate
model = whisper.load_model("base")

# ✅ Use large-v3 or large-v3-turbo
model = whisper.load_model("large-v3-turbo")  # or faster-whisper
```

### 7.2 Antipattern 2: One frame per second for video analysis

```python
# ❌ Expensive; generates 30+ frames per minute
while running:
    frame = get_frame()
    gpt4_vision.analyze(frame)

# ✅ Sample 1 frame per 10 seconds OR use scene detection
key_frames = extract_scene_changes(video_path)
```

### 7.3 Antipattern 3: No speaker labels in transcript

```python
# ❌ Transcript without speaker info is hard to search
transcript = whisper.transcribe(audio)

# ✅ Add diarization (speaker ID)
transcript_with_speakers = transcribe_with_pyannote_diarization(audio)
# Now search "What did Bob say about the budget?"
```

### 7.4 Antipattern 4: Converting audio at runtime

```python
# ❌ Convert every request to MP3 inside the request handler
async def process(audio_path):
    converted = convert_to_wav(audio_path)  # slow
    transcript = whisper.transcribe(converted)

# ✅ Convert once at upload time; cache the result
async def process_upload(audio_path):
    converted = convert_to_wav(audio_path)
    transcript = whisper.transcribe(converted)
    cache.set(audio_path, transcript, ttl=3600)

async def process(audio_path):
    transcript = cache.get(audio_path) or transcribe(audio_path)
```

### 7.5 Antipattern 5: Storing raw audio in the vector DB

```python
# ❌ Storing audio bytes is wasteful
client.upsert(... payload={"audio_blob": raw_audio_bytes})

# ✅ Store transcript text + reference to audio file
client.upsert(... payload={
    "transcript": "...",
    "audio_path": "s3://bucket/audio/meeting.mp4",
    "start": 0,
    "end": 30,
})
```

---

## 🎯 Key Takeaways

- **Whisper large-v3-turbo** (or faster-whisper) transcribes audio in 4× faster than v3.
- **Audio RAG**: transcribe → chunk → embed → retrieve; speaker diarization adds search richness.
- **CLAP** enables audio search without transcripts (for music, ambient sounds).
- **Video RAG**: extract keyframes → CLIP embed + transcript chunks → multimodal retrieval.
- **GPT-4o vision** processes single frames for live monitoring.
- **Gemini 1.5** processes entire videos natively (no frame extraction).
- For portfolio: StayBot voice search, MARS video analysis, LLM Gateway audio logs.
- Avoid Whisper base in production, 1 FPS video analysis, no speaker labels, runtime conversion, storing raw audio.

## References

- Whisper — [github.com/openai/whisper](https://github.com/openai/whisper)
- faster-whisper — [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- CLAP — [github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
- pyannote-audio (diarization) — [github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- PySceneDetect — [github.com/Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- Gemini Video — [ai.google.dev/gemini-api/docs/video-understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- OpenAI TTS — [platform.openai.com/docs/guides/text-to-speech](https://platform.openai.com/docs/guides/text-to-speech)
- [[06 - Large Language Models/16 - HuggingFace Transformers Deep Dive/05 - Vision, Audio, and Multimodal Transformers|06/16/05 Vision, Audio, and Multimodal Transformers]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/01 - PDF Parsing and Extraction|Note 01 — PDF Parsing]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/03 - Multimodal Embedding Models|Note 03 — Multimodal Embeddings]]
- [[06 - Large Language Models/26 - Multimodal Production RAG/05 - Capstone - Production Multimodal RAG|Note 05 — Capstone]]
- [[10 - Cloud, Infra y Backend/33 - Vector Databases and Semantic Search|Vector Databases]]