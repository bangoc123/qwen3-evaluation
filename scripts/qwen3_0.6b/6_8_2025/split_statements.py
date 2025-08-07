import os
import asyncio
from dotenv import load_dotenv
import json 
from pydantic import BaseModel
from vertex import Gemini_Vertex
load_dotenv()
import random
import time 

def retry_request(fn, retries=7):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            wait = (2 ** i) + random.uniform(0, 1)
            print(f"Error occurred: {e}, retrying in {wait:.2f} seconds... (attempt {i + 1}/{retries})")
            time.sleep(wait)
    print("Max retries exceeded.")
    return None

class SplitStatement(BaseModel):
    statement: str
    

class ResponseSplitStatement(BaseModel):
    statements: list[SplitStatement]


class SplitStatements():
    def __init__(self, MODEL_ID):
        self.gemini = Gemini_Vertex(MODEL_ID)
    
    def get_json(self,statements):
        l = statements.find('{')
        r = statements.rfind('}')
        return json.loads(statements[l:r+1])
        
    def split_statements(self, question: str, response: str) -> list[str]:
        prompt = f"""
            You are given a user question, an AI-generated answer.

            Your task has one parts:

            1. **Decompose the Answer:**
            Break down the answer into a list of standalone factual statements. Each statement must:
            - Not use any pronouns (e.g., "he", "she", "it", "they").
            - Be a complete and understandable sentence that conveys one atomic idea.
            - Provide a short `"reason"` for each statement.
            - The language of the statement must be in the same language as the question.

            Return the result in the following JSON format:
            {{
            "statements": [
                {{
                "statement": "...",
                }},
                ...
            ]
            }}
            ---

            ### Example

            **Question:**  
            What are the key features of the display on the TCL 40 NXTPAPER 8GB/256GB mobile phone?

            **Answer:**  
            Based on the provided information, the display of the "TCL 40 NXTPaper 8GB/256GB" phone stands out with the following features:

                90Hz refresh rate

                Resolution of 2460x1080 pixels

                50MP main camera with f/1.8 aperture, 5MP ultra-wide camera with f/2.2 aperture, 2MP macro camera with f/2.4 aperture

                256GB of storage, 8GB of RAM.
                These detailed specifications are taken from the first product listing.

            **Output:**
            {{
            "statements": [
                {{
                "statement": "The display of the TCL 40 NXTPaper 8GB/256GB phone has a 90Hz refresh rate."
                }},
                {{
                "statement": "The display of the TCL 40 NXTPaper 8GB/256GB phone has a resolution of 2460x1080 pixels."
                }},
                {{
                "statement": "The TCL 40 NXTPaper 8GB/256GB phone has a 50MP main camera with an f/1.8 aperture."
                }},
                {{
                "statement": "The TCL 40 NXTPaper 8GB/256GB phone has a 5MP ultra-wide camera with an f/2.2 aperture."
                }},
                {{
                "statement": "The TCL 40 NXTPaper 8GB/256GB phone has a 2MP macro camera with an f/2.4 aperture."
                }},
                {{
                "statement": "The TCL 40 NXTPaper 8GB/256GB phone has 256GB of internal storage."
                }},
                {{
                "statement": "The TCL 40 NXTPaper 8GB/256GB phone has 8GB of RAM."
                }},
                {{
                "statement": "These specifications are taken from the first product listing."
                }}
            ]
            }}


            ---

            Input:
            Question: {question}

            Answer: {response}

            """

        
        response = retry_request(lambda: self.gemini.response(prompt,ResponseSplitStatement))
        state_json = json.loads(response)
        return [s["statement"] for s in state_json["statements"]]
    

    
