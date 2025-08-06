from google import genai
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../../coursemind-ai-7ab33cfdfd24.json"

PROJECT_ID = "coursemind-ai"
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")


class Gemini_Vertex():
    def __init__(self, MODEL_ID):
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        self.MODEL_ID = MODEL_ID
    
    def clean_response(self, text):
        l = text.find('{')
        r = text.rfind("}")
        return text[l:r+1]

    def response(self, question : str, Response) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.MODEL_ID,
                contents=question,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Response,
                },
            )

            # print(self.clean_response(response.text))
            return response.text
        except Exception as e:
            print(f"Error generating question for product: {e}")
            raise