from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Debug: Check environment variables
logger.info("Checking environment variables...")
logger.info(f"OPENAI_API_KEY present: {'OPENAI_API_KEY' in os.environ}")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContentRequest(BaseModel):
    content: str
    thread_length: int = 5  # Default to 5 tweets
    tone: str = "neutral"  # Default tone
    writing_style: str = ""  # Optional writing style

def generate_thread(content: str, thread_length: int = 5, tone: str = "neutral", writing_style: str = "") -> list:
    try:
        logger.info(f"Attempting to generate thread with GPT (length: {thread_length}, tone: {tone})")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        client = OpenAI(
            api_key=api_key,
        )
        
        tone_instructions = {
            "neutral": "Use a balanced and objective tone",
            "formal": "Use a professional and academic tone",
            "casual": "Use a friendly and conversational tone",
            "enthusiastic": "Use an energetic and excited tone"
        }
        
        style_instruction = ""
        if writing_style:
            style_instruction = f"""
            Match this writing style/voice in the generated tweets:
            {writing_style}
            """
        
        prompt = f"""Create a focused Twitter thread that directly addresses and expands on the given content. 
        Format each tweet with a number prefix like "1." at the start.

        {style_instruction}

        Key Requirements:
        1. First tweet (1. 🧵) must directly introduce the main topic from the content
        2. Each tweet must be under 280 characters
        3. Stay focused on the specific topic/content provided
        4. Don't include generic advice or unrelated points
        5. Use concrete examples and points from the given content
        6. Each tweet should build on the specific topic, not general statements
        
        Tweet Structure:
        • Tweet 1: Hook that directly states what will be covered, based on the input content
        • Middle Tweets: Specific points, examples, and insights from the content
        • Final Tweet: Conclusion with key takeaways from the content + relevant hashtags
        
        Format Guidelines:
        • Start each tweet with its number (1., 2., 3., etc.)
        • Use 1-2 relevant emojis per tweet
        • Include bullet points or numbered lists when listing multiple points
        • Keep the focus tight and relevant to the input content
        
        Remember:
        • Don't generate generic advice
        • Stay strictly focused on the input content
        • Each tweet must directly relate to the main topic
        • Number of tweets must be exactly {thread_length}
        • {tone_instructions.get(tone, "Use a balanced tone")}

        Content to transform into a thread:
        {content}"""
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": """You are an expert content strategist who creates highly focused Twitter threads.
                Your threads are known for:
                - Staying strictly on topic
                - Using specific examples from the input
                - Never including generic advice
                - Being concise and information-dense
                - Converting the given content directly into tweet format
                
                You always ensure each tweet directly relates to the input content and avoid generic statements."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Process the tweets
        raw_content = response.choices[0].message.content.strip()
        # Split by numbered tweet markers (1., 2., etc.)
        tweets = []
        current_tweet = []
        
        for line in raw_content.split('\n'):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Check if this line starts a new tweet
            if any(line.startswith(f"{i}.") for i in range(1, thread_length + 1)):
                if current_tweet:  # Save previous tweet if exists
                    tweets.append('\n'.join(current_tweet))
                current_tweet = [line]  # Start new tweet
            elif current_tweet:  # Add line to current tweet if it exists
                current_tweet.append(line)
        
        # Add the last tweet if exists
        if current_tweet:
            tweets.append('\n'.join(current_tweet))
        
        # Clean up tweets and ensure proper length
        tweets = [tweet.strip() for tweet in tweets if tweet.strip()]
        
        # Ensure we have exactly the requested number of tweets
        if len(tweets) > thread_length:
            tweets = tweets[:thread_length]
        elif len(tweets) < thread_length:
            logger.warning(f"Generated only {len(tweets)} tweets, requested {thread_length}")
        
        logger.info(f"Generated {len(tweets)} tweets")
        return tweets
        
    except Exception as e:
        logger.error(f"Error in generate_thread: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating thread: {str(e)}")

@app.post("/generate-thread")
async def create_thread(request: ContentRequest):
    try:
        logger.info(f"Received request with content length: {len(request.content)}")
        thread = generate_thread(
            content=request.content,
            thread_length=request.thread_length,
            tone=request.tone,
            writing_style=request.writing_style
        )
        return {"thread": thread}
        
    except Exception as e:
        logger.error(f"Error in create_thread endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e)) 