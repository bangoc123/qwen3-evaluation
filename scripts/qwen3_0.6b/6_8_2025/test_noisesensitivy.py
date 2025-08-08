import unittest
import pandas as pd
import time
import os
import asyncio
from dotenv import load_dotenv
import openai 
import google.generativeai as genai
from split_statements import SplitStatements
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from vertex import Gemini_Vertex
import random

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

class Review(BaseModel):
    statement: str
    reason: list
    verdict: int

class Reviews(BaseModel):
    statements: list[Review]

# Load environment variables from .env file
load_dotenv()
class NoisesensitivyTest(unittest.TestCase):
    """Test suite for calculating noisesensitivy scores"""

    @classmethod
    def setUpClass(cls):
        File_Data = os.getenv('FILE_DATA', "../../../data/deepseek_685b/output_partial_685B_with.csv")
        File_Output = os.getenv('FILE_OUTPUT', "../../../results/log_noisesensitivy_test_results_685B.csv")
        Model_Name = os.getenv('MODEL_NAME', "deepseek-ai/DeepSeek-R1-0528-tput")
        MODEL_ID = os.getenv('MODEL_ID')

        cls.gemini = Gemini_Vertex(MODEL_ID)
        cls.split_statements = SplitStatements(MODEL_ID)
        cls.Model_Name = Model_Name
        cls.File_Output = File_Output

        # Load CSV data
        try:
            current_path = os.path.dirname(__file__)
            file_path = os.path.join(current_path, File_Data)
            cls.df = pd.read_csv(file_path)
            print(f"Loaded {len(cls.df)} test cases from CSV")
        except FileNotFoundError:
            raise unittest.SkipTest(f"{File_Data} not found")
    
    def fill_prompt(self, question: str, response: list[str], context: list) -> str:
        """Fill the prompt template"""
        GROUNDING_AUTORATER_PROMPT = """
                    You are given a user question, the list AI-generated statements of answer, and a reference context.

                    1. **Evaluate Faithfulness:**
                    For each extracted statement, determine whether it is supported by the context:
                    - Assign `"verdict": 1` if the statement can be directly inferred from the context.
                    - Assign `"verdict": 0` if the statement cannot be directly inferred.
                    - Provide a short `"reason"` for each verdict.

                    Return the result in the following JSON format:
                    {{
                    "statements": [
                        {{
                        "statement": "...",
                        "reason": "...",
                        "verdict": 1 or 0
                        }},
                        ...
                    ]
                    }}
                    ---

                    ### ✅ Example

                    **Question:**  
                    Who is Marie Curie and what is she famous for?

                    **Answer:**  
                    ["Marie Curie was a physicist.", "Marie Curie was a chemist.", "Marie Curie won two Nobel Prizes.", "Marie Curie discovered radium.", "Marie Curie discovered polonium.", "Marie Curie taught at Sorbonne University in Paris."]

                    **Ground Truth:**  
                    Marie Curie was a pioneering physicist and chemist who conducted research on radioactivity. She was awarded two Nobel Prizes: one in Physics and one in Chemistry. She is known for discovering the radioactive elements radium and polonium.

                    **Output:**
                    {{
                    "statements": [
                        {{
                        "statement": "Marie Curie was a physicist.",
                        "reason": "The ground truth confirms that Marie Curie was a physicist.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie was a chemist.",
                        "reason": "The ground truth confirms that Marie Curie was a chemist.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie won two Nobel Prizes.",
                        "reason": "The ground truth confirms that Marie Curie won two Nobel Prizes in Physics and Chemistry.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie discovered radium.",
                        "reason": "The ground truth clearly states that she discovered the radioactive elements radium and polonium.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie discovered polonium.",
                        "reason": "The ground truth clearly states that she discovered the radioactive elements radium and polonium.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie taught at Sorbonne University in Paris.",
                        "reason": "The ground truth does not mention anything about Marie Curie teaching at the Sorbonne or any other university.",
                        "verdict": 0
                        }}
                    ]
                    }}
                    ---

                    Input:
                    Question: {question}

                    Answer: {response}

                    Context: {context}
                    """
        return GROUNDING_AUTORATER_PROMPT.format(question= question, response=response, context=context)

    def label_statements(self, question: str, response: list[str], contexts: list) -> list:
        """label statements based on the context"""
        try:
            if not question or not response or not contexts:
                return []

            full_prompt = self.fill_prompt(question, response, contexts)
            response = retry_request(lambda: self.gemini.response(full_prompt,Reviews))
            if not response:
                return []
            
            response_json = json.loads(response)
  

            return response_json["statements"]
        except Exception as e:
            print(f"Error calculating noisesensitivy score: {e}")
            return []

    def calculate_noisesensitivy_score(self, question: str, response: list[str], contexts: list) -> tuple[list,float]:
        """NoiseSensitivy score calculation"""
        try:
            if not question or not response or not contexts:
                return [],1.0

            # Get labeled statements
            labeled_statements = self.label_statements(question, response, contexts)
            if not labeled_statements:
                return [],1.0

            # Calculate NoiseSensitivy score
            total_statements = len(labeled_statements)
            if total_statements == 0:
                return [],1.0
            
            correct_statements = sum(1 for stmt in labeled_statements if stmt.get("verdict") == 0)
            noise_sensitivity_score = correct_statements / total_statements

            return labeled_statements,round(noise_sensitivity_score, 2)
        except Exception as e:
            print(f"Error calculating NoiseSensitivy score: {e}")
            return [],1.0
        

    def save_result_to_csv(self, result: dict, filename: str = "log_noisesensitivy_test_results.csv"):
        """Save test result to CSV file"""
        try:
            file_exists = os.path.isfile(filename)
            result_df = pd.DataFrame([result])
            result_df.to_csv(filename, mode='a', header=not file_exists, index=False)
        except Exception as e:
            print(f"Warning: Failed to save result to CSV: {e}")

    def test_noisesensitivy_scores_all_queries(self):
        """Test all queries from CSV and calculate noisesensitivy scores"""
        
        if os.path.exists(self.File_Output):
            os.remove(self.File_Output)

        total_noisesensitivy_scores = []

        print(f"\nStarting Noisesensitivy score evaluation for {len(self.df)} test cases...")
        print("=" * 80)

        model_column = f"{self.Model_Name}_answer"  

        def process_row(index, row):
                    query = str(row.get('generated_question', '')).strip()
                    expected_answer = str(row.get('answer', '')).strip()
                    context = [str(row.get('reference_str', '')).strip()] 
                    model_answer = str(row.get(model_column, '')).strip()
                    # print(model_column)

                    if not query or not context[0] or not model_answer:
                        print(f"Row {index}: Skipping - missing query, context, or model answer")

                    print(f"Row {index}: Testing query: '{query[:50]}...'")

                    try:
                        # Calculate noisesensitivy score
                        noisesensitivy_start_time = time.time()
                        statements_of_response = self.split_statements.split_statements(query, model_answer)
                        labeled_statement,noisesensitivy_score = self.calculate_noisesensitivy_score(query, statements_of_response, context)
                        noisesensitivy_calc_time = time.time() - noisesensitivy_start_time

                        total_noisesensitivy_scores.append(noisesensitivy_score)

                        # Prepare result data
                        result = {
                            "row_index": index,
                            "generated_question": query,
                            "expected_answer": expected_answer,
                            "context": context,
                            "model_answer": model_answer,
                            "label_statements": labeled_statement,
                            "noisesensitivy_score": round(noisesensitivy_score,2),
                            "noisesensitivy_calc_time_seconds": round(noisesensitivy_calc_time, 2),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }

                        # # Save result
                        self.save_result_to_csv(result, self.File_Output)

                        time.sleep(1)
                    except Exception as e:
                        print(f"Row {index}: Error processing query '{query}': {e}")

        # Adjust the max_workers based on your rate limit and CPU
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row, index, row) for index, row in self.df.iterrows()]
            for future in as_completed(futures):
                future.result()  # To raise exceptions if any

