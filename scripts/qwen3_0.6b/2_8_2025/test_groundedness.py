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

# Load environment variables from .env file
load_dotenv()

class GroundednessTest(unittest.TestCase):
    """Test suite for calculating Groundedness scores"""

    @classmethod
    def setUpClass(cls):
        """Setup test data and RAGAS Groundedness scorer"""
        api_key = os.getenv('OPENAI_API_KEY')
        File_Data = os.getenv('FILE_DATA', "../../../data/deepseek_685b/output_partial_685B_with.csv")
        File_Output = os.getenv('FILE_OUTPUT', "../../../results/log_groundedness_test_results_685B.csv")
        Model_Name = os.getenv('MODEL_NAME', "deepseek-ai/DeepSeek-R1-0528-tput")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        cls.evaluator_llm = openai.OpenAI(api_key=api_key)
        cls.split_statements = SplitStatements()
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
    
    def fill_prompt(self, response: list[str], context: list) -> str:
        """Fill the prompt template"""
        GROUNDING_AUTORATER_PROMPT = """
                    You are a helpful and harmless AI assistant. You will be provided with a textual
                    context and the list model-generated statements of response.
                    Your task is to analyze the list statements of response and classify each
                    statement according to its relationship with the provided context.

                    **Instructions:**
                    1. **For each statement, assign one of the following labels:**
                        * **`supported`**: The statement is entailed by the given context.  Provide a
                        supporting excerpt from the context. The supporting excerpt must *fully*
                        entail the statement. If you need to cite multiple supporting excerpts,
                        simply concatenate them.
                        * **`unsupported`**: The statement is not entailed by the given context. No
                        excerpt is needed for this label.
                        * **`contradictory`**: The statement is falsified by the given context.
                        Provide a contradicting excerpt from the context.
                        * **`no_rad`**: The statement does not require factual attribution (e.g.,
                        opinions, greetings, questions, disclaimers).  No excerpt is needed for
                        this label.
                    2. **For each label, provide a short rationale explaining your decision.**
                    The rationale should be separate from the excerpt.
                    3. **Be very strict with your `supported` and `contradictory` decisions.**
                    Unless you can find straightforward, indisputable evidence excerpts *in the
                    context* that a sentence is `supported` or `contradictory`, consider it
                    `unsupported`. You should not employ world knowledge unless it is truly
                    trivial.

                    **Input Format:**

                    The input will consist of two parts, clearly separated:

                    * **Context:**  The textual context used to generate the response.
                    * **Response:** The list model-generated statements of response.

                    **Output Format:**
                    Return the result in the following JSON format:
                    {{
                    "statements": [
                        {{
                        "sentence": "...",
                        "label": "...",
                        "rationale": "...",
                        "excerpt": "..."
                        }},
                        ...
                    ]
                    }}
                    For each sentence in the response, output with the following
                    fields:

                    * `"sentence"`: The sentence being analyzed.
                    * `"label"`: One of `supported`, `unsupported`, `contradictory`, or `no_rad`.
                    * `"rationale"`: A brief explanation for the assigned label.
                    * `"excerpt"`:  A relevant excerpt from the context. Only required for
                    `supported` and `contradictory` labels.

                    **Example:**

                    **Input:**

                    ```
                    Context:
                    Apples are red fruits. Bananas are yellow fruits.

                    Response:
                    [Apples are red, Bananas are green, Bananas are cheaper than apples,Enjoy your fruit!]
                    ```

                    **Output:**
                    {{
                    "statements":
                        [
                        {{"sentence": "Apples are red.", "label": "supported", "rationale": "The context explicitly states that apples are red.", "excerpt": "Apples are red fruits."}}
                        {{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The context states that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}}
                        {{"sentence": "Bananas are cheaper than apples.", "label": "unsupported", "rationale": "The context does not mention the price of bananas or apples.", "excerpt": null}}
                        {{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a general expression and does not require factual attribution.", "excerpt": null}}
                        ]
                    }}
                    **Now, please analyze the following context and response:**

                    **Context:**
                    {context}

                    **Response:**
                    {response}
                    """
        return GROUNDING_AUTORATER_PROMPT.format(context=context, response=response)

    def label_statements(self, response: list[str], contexts: list) -> list:
        """label statements of response based on context"""
        try:
            if not response or not contexts:
                return []

            full_prompt = self.fill_prompt(response, contexts)
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            response = self.evaluator_llm.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            # Extract the JSON response
            if not response.choices or not response.choices[0].message:
                return []
            if not response.choices[0].message.content:
                return []
            
            try:
                response_json = self.split_statements.get_json(response.choices[0].message.content)
            except json.JSONDecodeError:
                print(f"Error decoding JSON response: {response.choices[0].message.content}")
                return response.choices[0].message.content  

            return response_json["statements"]
        except Exception as e:
            print(f"Error label statements: {e}")
            return []

    def calculate_groundedness_score(self, response: list[str], contexts: list) -> tuple[list,float]:
        """calculate Groundedness score"""
        label_statements = self.label_statements(response, contexts)
        label_cal_score = [item for item in label_statements if item["label"] != "no_rad"] 
        if len(label_cal_score) == 0:
            return [],0.0
        score = sum(1 for item in label_cal_score if item["label"] == "supported") / len(label_cal_score)
        return label_statements,score

    def save_result_to_csv(self, result: dict, filename: str = "log_groundedness_test_results.csv"):
        """Save test result to CSV file"""
        try:
            file_exists = os.path.isfile(filename)
            result_df = pd.DataFrame([result])
            result_df.to_csv(filename, mode='a', header=not file_exists, index=False)
        except Exception as e:
            print(f"Warning: Failed to save result to CSV: {e}")

    def test_groundedness_scores_all_queries(self):
        """Test all queries from CSV in parallel and calculate Groundedness scores"""

        if os.path.exists(self.File_Output):
            os.remove(self.File_Output)

        print(f"\nStarting Groundedness score evaluation using RAGAS for {len(self.df)} test cases...")
        print("=" * 80)

        model_column = f"{self.Model_Name}_answer"

        def process_row(index, row):
            query = str(row.get('generated_question', '')).strip()
            expected_answer = str(row.get('answer', '')).strip()
            context = [str(row.get('reference_str', '')).strip()]
            model_answer = str(row.get(model_column, '')).strip()

            if not query or not context[0] or not model_answer:
                print(f"Row {index}: Skipping - missing query, context, or model answer")
                return

            print(f"Row {index}: Testing query: '{query[:50]}...'")

            try:
                groundedness_start_time = time.time()
                statements_response = self.split_statements.split_statements(query, model_answer)
                label_statement, groundedness_score = self.calculate_groundedness_score(statements_response, context)
                groundedness_calc_time = time.time() - groundedness_start_time

                result = {
                    "row_index": index,
                    "generated_question": query,
                    "expected_answer": expected_answer,
                    "context": context,
                    "model_answer": model_answer,
                    "label_statements": label_statement,
                    "groundedness_score": round(groundedness_score, 2),
                    "groundedness_calc_time_seconds": round(groundedness_calc_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                self.save_result_to_csv(result, self.File_Output)

            except Exception as e:
                print(f"Row {index}: Error processing query '{query}': {e}")
                
        # Adjust the max_workers based on your rate limit and CPU
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row, index, row) for index, row in self.df.iterrows()]
            for future in as_completed(futures):
                future.result()  # To raise exceptions if any


    

