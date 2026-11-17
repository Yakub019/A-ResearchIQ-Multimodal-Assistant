import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault(
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    "2000",
)

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase

from app.retrieval.retriever import get_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "goldens" / "retriever_golden_dataset.json"
THRESHOLD = 0.7
TOP_K = 4

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing.")

judge_model = OpenRouterModel(
    model="openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

if not GOLDEN_PATH.is_file():
    raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_PATH}")

with GOLDEN_PATH.open("r", encoding="utf-8") as file:
    goldens = json.load(file)

if not isinstance(goldens, list):
    raise ValueError(f"Expected a list of golden cases in {GOLDEN_PATH}")

retriever = get_retriever(k=TOP_K)
test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g["query"])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieval_context,
            actual_output="(generator not evaluated in this run)",
        )
    )



metrics = [
    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=False,
    ),
    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=False,
    ),
]

print(f"Evaluating {len(test_cases)} golden queries with top_k={TOP_K}.")

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=False,
        max_concurrent=1,
        throttle_value=3,
    ),
)
