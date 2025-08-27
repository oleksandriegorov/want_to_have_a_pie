import base64
import os
from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import openai
from PIL import Image
import io


class VisionToolInput(BaseModel):
    """Input schema for VisionTool."""
    image_path: str = Field(..., description="Path to the image file to analyze")
    prompt: str = Field(
        default="Analyze this image and identify all food items visible. For each food item, provide a descriptive name and estimate its weight in grams. Format your response as a list with items, weights, and confidence levels.",
        description="Custom prompt for analyzing the image"
    )


class VisionTool(BaseTool):
    name: str = "Vision Food Analyzer"
    description: str = (
        "Analyzes images to identify and quantify food items using OpenAI's vision model. "
        "Converts images to base64 and sends them directly to GPT-4 vision model for analysis. "
        "Perfect for food recognition, portion estimation, and ingredient identification."
    )
    args_schema: Type[BaseModel] = VisionToolInput

    def _get_openai_client(self):
        """Get OpenAI client instance."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return openai.OpenAI(api_key=api_key)

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string."""
        try:
            # Optimize image size if it's too large
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if image is too large (max 2048px on longest side)
                max_size = 2048
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr = img_byte_arr.getvalue()
                
            return base64.b64encode(img_byte_arr).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to encode image: {str(e)}")

    def _run(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Analyze an image using OpenAI's vision model.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for analysis (optional)
            
        Returns:
            Analysis result from the vision model
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                return f"Error: Image file not found at {image_path}"
            
            # Use default prompt if none provided
            if not prompt:
                prompt = ("Analyze this image and identify all food items visible. "
                         "For each food item, provide:\n"
                         "- Descriptive name (suitable for recipe searches)\n"
                         "- Estimated weight in grams\n"
                         "- Confidence level (percentage)\n\n"
                         "Format as a list like:\n"
                         "- watermelon chunks: 350g (95% confidence)\n"
                         "- banana: 120g (90% confidence)\n\n"
                         "Only include items where confidence is 90% or higher.")
            
            print(f"🔍 Analyzing image: {image_path}")
            print(f"📏 File size: {os.path.getsize(image_path):,} bytes")
            
            # Encode image to base64
            base64_image = self._encode_image(image_path)
            print(f"✅ Image encoded to base64 ({len(base64_image):,} characters)")
            
            # Prepare the vision request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            print("🚀 Sending request to OpenAI vision model...")
            
            # Call OpenAI vision model
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use the vision-capable model
                messages=messages,  # type: ignore
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent results
            )
            
            result = response.choices[0].message.content or "No response received"
            
            print("✅ Received response from OpenAI")
            if response.usage:
                print(f"📊 Token usage: {response.usage.total_tokens} total tokens")
            
            return result
            
        except openai.AuthenticationError:
            return "Error: Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable."
        except openai.RateLimitError:
            return "Error: OpenAI API rate limit exceeded. Please try again later."
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
