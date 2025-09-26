import json
from typing import Optional
from prompts import (
    PROMPT_SPLIT_STATEMENTS, 
    F1_SCORE_LABEL_STATEMENT_PROMPT, 
    GROUNDEDNESS_LABEL_STATEMENT_PROMPT, 
    NOISESENSITIVY_LABEL_STATEMENT_PROMPT
)
from models import ResponseSplitStatement, Reviews, ReviewsGroundedness
from retry import retry_request
from constants import AVAILABLE_METRICS, PRECISION, RECALL, F1, GROUNDEDNESS, NOISE_SENSITIVITY  
from llms import LLMs
import uuid




class Evaluator():
    def __init__(self, 
                metrics: list = [PRECISION, RECALL, F1, GROUNDEDNESS, NOISE_SENSITIVITY], 
                model_type: str = "online", 
                model_name: str = "gemini", 
                model_version: str = "gemini-2.5-pro"):
        self.metrics = metrics 
        self.model_type = model_type
        self.model_name = model_name
        self.model_version = model_version
        self.llm = LLMs(type=model_type, model_name=model_name, model_version=model_version)


    def _split_statements(self, question: str, response: str) -> Optional[list[str]]:
        try:
            if not question or not response :
                return None
            prompt = PROMPT_SPLIT_STATEMENTS.format(question, response)
            response = retry_request(lambda: self.llm.response(prompt,ResponseSplitStatement))
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
                print(f"Warning: Empty or invalid response received from {self.model_name}.")
                return []

            response_json = json.loads(response)

            return response_json["statements"]
        except Exception as e:
            print(f"Error when generating label statements: {e}")
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
            precision = correct_statements / len(statements_of_response_llm) if len(statements_of_response_llm) > 0 else 0.0

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
            labeled_statements = self._label_statements(F1_SCORE_LABEL_STATEMENT_PROMPT, Reviews, question,statements_of_ground_truth,response_llm)
            if not labeled_statements:
                return [], 0.0

            # Calculate recall score
            total_statements = len(labeled_statements)
            if total_statements == 0:
                return [], 0.0
            
            correct_statements = sum(1 for stmt in labeled_statements if stmt.get("verdict") == 1)
            recall = correct_statements / len(statements_of_ground_truth) if len(statements_of_ground_truth) > 0 else 0.0

            return labeled_statements,round(recall, 2)
        except Exception as e:
            print(f"Error calculating recall score: {e}")
            return [], 0.0
    
    def _f1_score_cal(self, question: str, response_llm: str, ground_truth: str) -> float:
        try:
            _, precision = self._precision_cal(question, response_llm, ground_truth)
            _, recall = self._recall_cal(question, response_llm, ground_truth)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return f1_score
        except Exception as e:
            print(f"Error calculating recall score: {e}")
            return 0.0
    
    def _groundedness_cal(self, question:str, response_llm: str, context: list) -> Optional[tuple[list,float]]:
        try:
            statements_response = self._split_statements(question, response_llm)
            label_statements = self._label_statements(GROUNDEDNESS_LABEL_STATEMENT_PROMPT, ReviewsGroundedness, question ,statements_response, context)
            label_cal_score = [item for item in label_statements if item["label"] != "no_rad"] 
            if len(label_cal_score) == 0:
                return [],0.0
            score = sum(1 for item in label_cal_score if item["label"] == "supported") / len(label_cal_score)
            return label_statements,score
        except Exception as e:
            print(f"Error calculating recall score: {e}")
            return [], 0.0

    def _noise_sensitivity_cal(self, question:str, response_llm: str, context: list) -> Optional[tuple[list,float]]:
        try:
            if not question or not response_llm or not context:
                return [], 1.0

            statements_of_response = self._split_statements(question, response_llm)
            # Get labeled statements
            labeled_statements = self._label_statements(NOISESENSITIVY_LABEL_STATEMENT_PROMPT, Reviews, question, statements_of_response, context)
            if not labeled_statements:
                return [], 1.0

            # Calculate NoiseSensitivy score
            total_statements = len(labeled_statements)
            if total_statements == 0:
                return [], 1.0
             
            correct_statements = sum(1 for stmt in labeled_statements if stmt.get("verdict") == 0)
            noise_sensitivity_score = correct_statements / total_statements

            return labeled_statements,round(noise_sensitivity_score, 2)
        except Exception as e:
            print(f"Error calculating NoiseSensitivy score: {e}")
            return [], 1.0
        
    
    def eval(self,
            questions: list = None, 
            response_llms: list = None,
            ground_truths: list = None,
            contexts: list = None):

        try:
            invalid = [m for m in self.metrics if m not in AVAILABLE_METRICS]
            if invalid:
                raise ValueError(f"Unsupported metric(s): {invalid}. Available: {AVAILABLE_METRICS}")

            metric_requirements = {
                PRECISION: {"questions", "response_llms", "ground_truths"},
                RECALL: {"questions", "response_llms", "ground_truths"},
                F1: {"questions", "response_llms", "ground_truths"},
                GROUNDEDNESS: {"questions", "response_llms", "contexts"},
                NOISE_SENSITIVITY: {"questions", "response_llms", "contexts"},
            }

            required_fields = set()
            for m in self.metrics:
                required_fields.update(metric_requirements.get(m, set()))    

            field_map = {
                "questions": questions,
                "response_llms": response_llms,
                "ground_truths": ground_truths,
                "contexts": contexts,
            }

            missing = [f for f in required_fields if not isinstance(field_map.get(f), list)]
            if missing:
                raise ValueError(f"The following required inputs are missing or not lists: {missing} "
                                f"(required by requested metrics: {self.metrics})")
               
            lengths = {len(field_map[f]) for f in required_fields}
            if len(lengths) > 1:
                length_info = {f: len(field_map[f]) for f in required_fields}
                raise ValueError(f"Input lists have mismatched lengths: {length_info}")

            n = next(iter(lengths)) if lengths else 0
            if n == 0:
                raise ValueError("No data to evaluate (length is 0).")

            results = []
            
            for i in range(n):
                question = questions[i] if questions is not None else None
                response_llm = response_llms[i] if response_llms is not None else None
                ground_truth = ground_truths[i] if ground_truths is not None else None
                context = contexts[i] if contexts is not None else None

                result = {}
                result['_id'] = str(uuid.uuid4())
                result['question'] = question
                result['response_llm'] = response_llm
                result['ground_truth'] = ground_truth
                result['context'] = context

                for metric in self.metrics:
                    if metric == PRECISION:
                        _, precision = self._precision_cal(question=question, response_llm=response_llm, ground_truth=ground_truth)
                        result[metric] = precision

                    elif metric == RECALL:
                        _, recall = self._recall_cal(question=question, response_llm=response_llm, ground_truth=ground_truth)
                        result[metric] = recall
                    
                    elif metric == F1:
                        f1_score = self._f1_score_cal(question=question, response_llm=response_llm, ground_truth=ground_truth)
                        result[metric] = f1_score
                    
                    elif metric == GROUNDEDNESS:
                        _, groundedness_score = self._groundedness_cal(question=question, response_llm=response_llm, context=context)
                        result[metric] = groundedness_score
                    
                    elif metric == NOISE_SENSITIVITY :
                        _, noise_sensitivity_score = self._noise_sensitivity_cal(question=question, response_llm=response_llm, context=context)
                        result[metric] = noise_sensitivity_score
                
                results.append(result)
            
            return results

        except Exception as e:
            raise ValueError(f"Error when evaluating: {e}")
