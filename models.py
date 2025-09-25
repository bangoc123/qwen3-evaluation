from pydantic import BaseModel

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
    statements: list[Review]