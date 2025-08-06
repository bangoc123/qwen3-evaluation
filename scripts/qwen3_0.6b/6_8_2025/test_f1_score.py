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
# Load environment variables from .env file
load_dotenv()

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

class F1Test(unittest.TestCase):
    """Test suite for calculating f1 scores"""

    @classmethod
    def setUpClass(cls):
        
        File_Data = os.getenv('FILE_DATA', "../../../data/deepseek_685b/output_partial_685B_with.csv")
        File_Output = os.getenv('FILE_OUTPUT', "../../../results/log_f1_score_test_results_685B.csv")
        Model_Name = os.getenv('MODEL_NAME', "deepseek-ai/DeepSeek-R1-0528-tput")
        MODEL_ID = os.getenv('MODEL_ID')
        
        cls.gemini = Gemini_Vertex(MODEL_ID)
        cls.split_statements = SplitStatements(MODEL_ID = MODEL_ID)
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
    
    def fill_prompt(self, question: str, response_llm: list[str], ground_truth: list) -> str:
        """Fill the prompt template"""
        GROUNDING_AUTORATER_PROMPT = """
                    You are given a user question, the list AI-generated statements of answer, and the list statements of ground_truth.

                    1. **Evaluate accuracy:**
                    For each extracted AI-generated statements of answer, determine whether it is supported by the list statements of ground_truth:
                    - Assign `"verdict": 1` if the statement can be directly inferred from the ground_truth.
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
                    ["Marie Curie was a physicist and chemist who won two Nobel Prizes.",  "Marie Curie discovered radium and polonium.", "Marie Curie taught at Sorbonne University in Paris."]

                    **Ground Truth:**  
                    ["Marie Curie was a pioneering physicist and chemist who won two Nobel Prizes", "She discovered radium and polonium", "She conducted research on radioactivity"]

                    **Output:**
                    {{
                    "statements": [
                        {{
                        "statement": "Marie Curie was a physicist and chemist who won two Nobel Prizes.",
                        "reason": "The ground truth confirms that Marie Curie was a physicist and chemist and that she won two Nobel Prizes in Physics and Chemistry.",
                        "verdict": 1
                        }},
                        {{
                        "statement": "Marie Curie discovered radium and polonium.",
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

                    Answer: {response_llm}

                    Ground Truth: {ground_truth}
                    """
        return GROUNDING_AUTORATER_PROMPT.format(question= question, response_llm=response_llm, ground_truth=ground_truth)

    def label_statements(self, question: str, response_llm: list[str], ground_truth: list) -> list:
        """label statements based on the context"""
        try:
            if not question or not response_llm or not ground_truth:
                return []

            full_prompt = self.fill_prompt(question, response_llm, ground_truth)
            response = retry_request(lambda: self.gemini.response(full_prompt,Reviews))
            if response is None or not isinstance(response, str) or response.strip() == "":
                print("Warning: Empty or invalid response received from Gemini.")
                return []

            
            response_json = json.loads(response)

            return response_json["statements"]
        except Exception as e:
            print(f"Error calculating f1 score: {e}")
            return []

    def calculate_score(self, question: str, response_llm: list[str], ground_truth: list) -> tuple[list, float, float, float]:
        """f1 score calculation"""
        try:
            if not question or not response_llm or not ground_truth:
                return [], 0.0, 0.0, 0.0

            # Get labeled statements
            labeled_statements = self.label_statements(question, response_llm, ground_truth)
            if not labeled_statements:
                return [], 0.0, 0.0, 0.0

            # Calculate f1 score
            total_statements = len(labeled_statements)
            if total_statements == 0:
                return [], 0.0, 0.0, 0.0
            
            correct_statements = sum(1 for stmt in labeled_statements if stmt.get("verdict") == 1)
            precision = correct_statements / len(response_llm) if len(response_llm) > 0 else 0.0
            recall = correct_statements / len(ground_truth) if len(ground_truth) > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return labeled_statements,round(precision, 2), round(recall, 2), round(f1_score, 2)
        except Exception as e:
            print(f"Error calculating f1 score: {e}")
            return [], 0.0, 0.0, 0.0
        

    def save_result_to_csv(self, result: dict, filename: str = "log_f1_score_test_results.csv"):
        """Save test result to CSV file"""
        try:
            file_exists = os.path.isfile(filename)
            result_df = pd.DataFrame([result])
            result_df.to_csv(filename, mode='a', header=not file_exists, index=False)
        except Exception as e:
            print(f"Warning: Failed to save result to CSV: {e}")

    def test_f1_scores_all_queries(self):
        """Test all queries from CSV and calculate f1 scores"""
        
        if os.path.exists(self.File_Output):
            os.remove(self.File_Output)


        print(f"\nStarting f1 score evaluation for {len(self.df)} test cases...")
        print("=" * 80)

        model_column = f"{self.Model_Name}_answer"  

        def process_row(index, row):
                query = str(row.get('generated_question', '')).strip()
                model_answer = str(row.get(model_column, '')).strip()
                ground_truth = str(row.get('answer', '')).strip()

                if not query or not ground_truth or not model_answer:
                    print(f"Row {index}: Skipping - missing query, context, or model answer")
              

                print(f"Row {index}: Testing query: '{query[:50]}...'")

                try:
                    # Calculate f1 score
                    score_start_time = time.time()
                    statements_of_response_llm = self.split_statements.split_statements(query, model_answer)
                    statements_of_ground_truth = self.split_statements.split_statements(query, ground_truth)
                    labeled_statement,precision, recall, f1_score = self.calculate_score(query, statements_of_response_llm, statements_of_ground_truth)
                    score_calc_time = time.time() - score_start_time


                    # Prepare result data
                    result = {
                        "row_index": index,
                        "query": query,
                        "expected_answer": ground_truth,
                        "model_answer": model_answer,
                        "label_statements": labeled_statement,
                        "precision": round(precision, 2),
                        "recall": round(recall, 2),
                        "f1_score": round(f1_score,2),
                        "score_calc_time_seconds": round(score_calc_time, 2),
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
    

