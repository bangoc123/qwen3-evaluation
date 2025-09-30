from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Union, Callable, Any
from dotenv import load_dotenv
from models import EvalRequest, EvalResponse 
import os, gc, json, asyncio, logging
from datetime import datetime
import time
import psutil
from evaluator import Evaluator
from constants import AVAILABLE_METRICS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("", response_model = EvalResponse)
@router.post("/", response_model = EvalResponse)
async def eval(request: EvalRequest):
    start = datetime.now()

    try:
        metrics = request.metrics
        questions = request.questions
        response_llms = request.response_llms
        ground_truths = request.ground_truths
        contexts = request.contexts

        invalid = [m for m in metrics if m not in AVAILABLE_METRICS]
        if len(invalid) > 0:
            raise HTTPException(status_code=400, detail=f"The following metrics {invalid} are not available. Please choose available metrics: {AVAILABLE_METRICS}")
        
        evaluator = Evaluator(
            metrics = metrics
        )

        logger.info(f"Evaluation request received with metrics: {metrics}")

        results = evaluator.eval(questions=questions, response_llms=response_llms, ground_truths=ground_truths, contexts=contexts)


        try: 
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    evaluator.eval,
                    questions=questions,
                    response_llms=response_llms,
                    ground_truths=ground_truths,
                    contexts=contexts
                ),
                timeout=36000
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Evaluation timeout")

        return EvalResponse(
            results=results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Evaluation error")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

@app.api_route("/health-check", methods=["GET", "HEAD", "OPTIONS"])
async def health_check():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss / 1024 / 1024
        return {
            "status": "ok",
            "message": "Evaluation service is healthy",
            "memory_usage_mb": round(mem, 2),
        }
    except Exception:
        return {"status": "ok", "model_loaded": local_model is not None}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "80"))  # set API_PORT=80 to run on :80
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    # uvicorn.run(app, host=host, port=port, reload=reload, workers=1)
    uvicorn.run(
    app, 
    host=host, 
    port=port, 
    reload=reload,
    workers=1,  # CRITICAL: Single worker only
    log_level="info",
    access_log=True,
    timeout_keep_alive=30,
    limit_concurrency=100
    )
