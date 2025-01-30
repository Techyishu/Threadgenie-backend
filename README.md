# Backend for YouTube to Twitter Thread Converter

This backend service converts YouTube videos to Twitter threads using yt-dlp for video processing and Claude AI for thread generation.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Running the Server

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The server will run on `http://localhost:8000`

## API Endpoints

### POST /generate-thread
Converts a YouTube video to a Twitter thread.

Request body:
```json
{
    "video_url": "https://www.youtube.com/watch?v=..."
}
```

Response:
```json
{
    "thread": [
        "Tweet 1",
        "Tweet 2",
        ...
    ]
}
``` 