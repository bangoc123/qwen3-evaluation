import os
import asyncio
from dotenv import load_dotenv
import openai 
import json 

load_dotenv()


class SplitStatements():
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.llm = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o"
    
    def get_json(self,statements):
        l = statements.find('{')
        r = statements.rfind('}')
        return json.loads(statements[l:r+1])
        
    def split_statements(self, question: str, response: str) -> list[str]:
        prompt = f"""
            You are given a user question, an AI-generated answer.

            Your task has two parts:

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
                "reason": "...",
                }},
                ...
            ]
            }}
            ---

            ### Example

            **Question:**  
            Who is Marie Curie and what is she famous for?

            **Answer:**  
            She was a physicist and chemist who won two Nobel Prizes. She discovered radium and polonium. She also taught at Sorbonne University in Paris.


            **Output:**
            {{
            "statements": [
                {{
                "statement": "Marie Curie was a physicist and chemist who won two Nobel Prizes.",
                "reason": "The context confirms that Marie Curie was a physicist and chemist and that she won two Nobel Prizes in Physics and Chemistry.",
                }},
                {{
                "statement": "Marie Curie discovered radium and polonium.",
                "reason": "The context clearly states that she discovered the radioactive elements radium and polonium.",
                }},
                {{
                "statement": "Marie Curie taught at Sorbonne University in Paris.",
                "reason": "The context does not mention anything about Marie Curie teaching at the Sorbonne or any other university.",
                }}
            ]
            }}

            ---

            Input:
            Question: {question}

            Answer: {response}

            """

        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        state_json = self.get_json(response.choices[0].message.content)
        return [s["statement"] for s in state_json["statements"]]
    

    
