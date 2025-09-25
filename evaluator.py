import json
from typing import Optional
from prompts import PROMPT_SPLIT_STATEMENTS, F1_SCORE_LABEL_STATEMENT_PROMPT, GROUNDEDNESS_LABEL_STATEMENT_PROMPT
from models import ResponseSplitStatement, Reviews, ReviewsGroundedness
from retry import retry_request

class Evaluator():
    def __init__(self, metrics: list, llm: LLM):
        self.metrics = metrics
        self.llm = llm

    def _split_statements(self, question: str, response: str) -> Optional[list[str]]:
        try:
            if not question or not response :
                return None
            prompt = PROMPT_SPLIT_STATEMENTS.format(question, response)
            response = retry_request(lambda: self.gemini.response(prompt,ResponseSplitStatement))
            if(response is None):
                return None
            state_json = json.loads(response)
            return [s["statement"] for s in state_json["statements"]]
        except:
            return None 

    def _label_statements(self, prompt, Response, question: str, statements: list[str], reference_text: str) -> list:
        try:
            if not question or not statements or not reference_text:
                return []

            full_prompt = prompt.format(question, statements, reference_text)
            response = retry_request(lambda: self.llm.response(full_prompt,Response))
            if response is None or not isinstance(response, str) or response.strip() == "":
                print("Warning: Empty or invalid response received from Gemini.")
                return []

            
            response_json = json.loads(response)

            return response_json["statements"]
        except Exception as e:
            print(f"Error calculating f1 score: {e}")
            return []
    
    def _precision_cal(self, question: str, response_llm: str, ground_truth: str) -> Optional[tuple[list, float]]:
        try:
            if not question or not response_llm or not ground_truth:
                return [], 0.0

            statements_of_response_llm = self._split_statements(question, response_llm)
            # Get labeled statements
            labeled_statements = self._label_statements(F1_SCORE_LABEL_STATEMENT_PROMPT, Reviews, question, statements_of_response_llm, ground_truth)
            if not labeled_statements:
                return [], 0.0

            # Calculate precision score
            total_statements = len(labeled_statements)
            if total_statements == 0:
                return [], 0.0
            
            correct_statements = sum(1 for stmt in labeled_statements if stmt.get("verdict") == 1)
            precision = correct_statements / len(response_llm) if len(response_llm) > 0 else 0.0

            return labeled_statements,round(precision, 2)
        except Exception as e:
            print(f"Error calculating precision score: {e}")
            return [], 0.0
    
    def _recall_cal(self, question: str, response_llm: str, ground_truth: str) -> Optional[tuple[list, float]]:
        try:
            if not question or not response_llm or not ground_truth:
                return [], 0.0

            statements_of_ground_truth = self._split_statements(question, ground_truth)
            # Get labeled statements
            labeled_statements = self.label_statements(F1_SCORE_LABEL_STATEMENT_PROMPT, Reviews, question,statements_of_ground_truth,response_llm)
            if not labeled_statements:
                return [], 0.0

            # Calculate recall score
            total_statements = len(labeled_statements)
            if total_statements == 0:
                return [], 0.0
            
            correct_statements = sum(1 for stmt in labeled_statements if stmt.get("verdict") == 1)
            recall = correct_statements / len(ground_truth) if len(ground_truth) > 0 else 0.0

            return labeled_statements,round(recall, 2)
        except Exception as e:
            print(f"Error calculating recall score: {e}")
            return [], 0.0
    
    def _f1_score_cal(self, question: str, response_llm: str, ground_truth: str) -> float:
        precision = self._precision_cal(question, response_llm, ground_truth)
        recall = self._recall_cal(question, response_llm, ground_truth)
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1_score
    
    def _groundedness_cal(self, question:str, response_llm: str, contexts: list) -> Optional[tuple[list,float]]:
        statements_response = self._split_statements(question, response_llm)
        label_statements = self._label_statements(response, contexts)
        label_cal_score = [item for item in label_statements if item["label"] != "no_rad"] 
        if len(label_cal_score) == 0:
            return [],0.0
        score = sum(1 for item in label_cal_score if item["label"] == "supported") / len(label_cal_score)
        return label_statements,score

    def _noisesensitivy(self, question:str, response_llm: str, contexts: list) -> Optional[tuple[list,float]]:
        pass 
    
    def eval(self):
        pass