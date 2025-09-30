from pydantic import BaseModel
from typing import List, Optional, Union, Callable, Any

class SplitStatement(BaseModel):
    statement: str
    

class ResponseSplitStatement(BaseModel):
    statements: list[SplitStatement]


class Review(BaseModel):
    statement: str
    reason: list
    verdict: int

class Reviews(BaseModel):
    statements: list[Review]


class ReviewGroundedness(BaseModel):
    sentence: str
    label: str
    rationale: str
    excerpt : list

class ReviewsGroundedness(BaseModel):
    statements: list[ReviewGroundedness]

class EvalRequest(BaseModel):
    metrics: list[str]
    questions: list[str] = None
    response_llms: list[str] = None 
    ground_truths: list[str] = None
    contexts: list[str] = None

class EvalResponse(BaseModel):
    results: list[dict]
