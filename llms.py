from google import genai
import openai
import requests
import re
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class OnLineLLMs:
    def __init__(self, model_name: str, model_version: str, base_url: str = None):
        """Initialize model with the specified name, API key, and model version."""
        self.model_name = model_name.lower()
        self.model_version = model_version

        if self.model_name == "gemini":
            print("Make sure you passed Google Cloud service account json file path in .env.")

            CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if CREDENTIALS == None:
                raise ValueError("Missing Google Cloud service account.")
            PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if PROJECT_ID is None: 
                raise ValueError("Missing GOOGLE_CLOUD_PROJECT environment variable")
            LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

            self.client = genai.Client(vertexai=True, project=PROJECT_ID, location = LOCATION)

        elif self.model_name == "openai":
            API_KEY = os.environ.get("OPENAI_API_KEY")
            if API_KEY == None:
                raise ValueError("Missing API key for OpenAI model.")

            self.client = openai.OpenAI(api_key=API_KEY)

        elif self.model_name == "together":
            raise ValueError("Unsupported model at this time.")

        else:
            raise ValueError("Unsupported model name or missing API key or service account.")

    def clean_response(self, text):
        l = text.find('{')
        r = text.rfind("}")
        return text[l:r+1]
    
    def response(self, prompt: str, Response) -> str:
        """Generate content using the online LLM based on the provided prompt.
            input: prompt (str): The prompt to generate content for.
            output: str: The generated content.
        """
        try:
            if self.model_name == "gemini":
                response = self.client.models.generate_content(
                    model=self.model_version,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": Response
                    }
                )
                try:
                    return response.text 
                except Exception as e:
                    print(f"Error generating question: {e}")
                    raise

            elif self.model_name == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_version,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                try:
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Error generating question: {e}")
                    raise

            elif self.model_name == "together":
                raise ValueError(f"This type of model is unsupported at this time.")
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")
        except Exception as e:
            print(f"Error in response generation: {e}")
            raise

class LLMs:
    def __init__(self, type : str, model_version: str, model_name : str = None, **kwargs):
        """Initialize object LocalLLMs or OnlineLLMs with the provided configuration.
            not return 
        """
        if type == "offline":
            raise ValueError(f"Local LLM is not supported at this time.")
        elif type == "online":
            self.llm = OnLineLLMs(model_name=model_name, model_version=model_version)
        else:
            raise ValueError(f"Unsupported LLM type: {type}")

    def response(self, prompt: str, Response) -> str:
        """
        Generate content using the LLM based on the provided prompt.
        input: prompt (str): The prompt to generate content for.
        output: str: The generated content.
        """
        return self.llm.response(prompt, Response)
