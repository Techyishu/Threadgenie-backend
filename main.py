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

class TweetRequest(BaseModel):
    topic: str
    tone: str = "neutral"
    writing_style: str = ""

class BioRequest(BaseModel):
    name: str
    expertise: str
    interests: list[str]
    tone: str = "professional"

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
        
        prompt = f"""Create an engaging Twitter thread that feels natural and conversational while covering the given content.
        Each tweet should start with a number (1., 2., etc.).

        {style_instruction}

        Key Points:
        • First tweet should hook readers naturally - avoid "In this thread..." or "Let's talk about..."
        • Jump straight into the topic with an interesting angle or surprising fact
        • Each tweet must be under 280 characters
        • Focus on specific insights from the content
        • Use real examples and details from the given content
        
        Thread Flow:
        • Tweet 1: Start with an attention-grabbing insight or statement that makes people want to read more
        • Middle Tweets: Break down key points with specific examples
        • Final Tweet: Wrap up with main takeaways + relevant hashtags
        
        Style Guide:
        • Write like you're talking to a friend
        • Use natural language and avoid corporate/formal phrases
        • Include 1-2 fitting emojis per tweet
        • Break complex ideas into digestible points
        
        Important:
        • Keep everything specific to the input content
        • Generate exactly {thread_length} tweets
        • {tone_instructions.get(tone, "Use a balanced tone")}
        • Avoid phrases like "Thread 🧵" or "Let me explain"
        • Don't use generic transitions between tweets

        Content to transform into a thread:
        {content}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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

def generate_single_tweet(topic: str, tone: str = "neutral", writing_style: str = "") -> str:
    try:
        logger.info(f"Generating single tweet for topic: {topic}")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        tone_instructions = {
            "neutral": "Use a balanced and objective tone",
            "formal": "Use a professional and academic tone",
            "casual": "Use a friendly and conversational tone",
            "enthusiastic": "Use an energetic and excited tone"
        }
        
        style_instruction = f"\n{writing_style}" if writing_style else ""
        
        prompt = f"""Create an engaging tweet about the following topic.

        Topic: {topic}

        Requirements:
        • Must be under 280 characters
        • Include 1-2 relevant emojis
        • Be specific and informative
        • {tone_instructions.get(tone, "Use a balanced tone")}
        {style_instruction}

        Style Guide:
        • Write naturally and conversationally
        • Avoid hashtag spam
        • Make it shareable and engaging
        • Focus on providing value
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a social media expert who creates engaging tweets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        tweet = response.choices[0].message.content.strip()
        return tweet
        
    except Exception as e:
        logger.error(f"Error generating tweet: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating tweet: {str(e)}")

def generate_bio(name: str, expertise: str, interests: list[str], tone: str = "professional") -> str:
    try:
        logger.info(f"Generating bio for {name}")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        tone_instructions = {
            "professional": "Keep it formal and business-focused",
            "casual": "Make it friendly and approachable",
            "creative": "Add personality and creative flair",
            "technical": "Focus on technical expertise and achievements"
        }
        
        interests_str = ", ".join(interests)
        
        prompt = f"""Create a compelling Twitter bio for:

        Name: {name}
        Expertise: {expertise}
        Interests: {interests_str}

        Requirements:
        • Maximum 160 characters
        • Include 1-2 relevant emojis
        • {tone_instructions.get(tone, "Keep it professional")}
        • Highlight expertise and personality
        
        Style Guide:
        • Be concise but informative
        • Show personality while maintaining professionalism
        • Include key achievements/roles
        • Make it memorable
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at creating engaging social media bios."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        bio = response.choices[0].message.content.strip()
        return bio
        
    except Exception as e:
        logger.error(f"Error generating bio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating bio: {str(e)}")

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

@app.post("/generate-tweet")
async def create_tweet(request: TweetRequest):
    try:
        tweet = generate_single_tweet(
            topic=request.topic,
            tone=request.tone,
            writing_style=request.writing_style
        )
        return {"tweet": tweet}
    except Exception as e:
        logger.error(f"Error in create_tweet endpoint: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-bio")
async def create_bio(request: BioRequest):
    try:
        bio = generate_bio(
            name=request.name,
            expertise=request.expertise,
            interests=request.interests,
            tone=request.tone
        )
        return {"bio": bio}
    except Exception as e:
        logger.error(f"Error in create_bio endpoint: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e)) 