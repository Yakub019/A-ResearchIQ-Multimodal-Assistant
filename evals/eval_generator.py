
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configure timeout before DeepEval imports
os.environ.setdefault(
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    "2000"
)

from deepeval import evaluate
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.evaluate import AsyncConfig

# ResearchIQ imports
from app.retrieval.retriever import get_retriever, format_documents
from app.chains.qa_chain import get_qa_chain


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "goldens"
    / "faithfulness_golden_dataset.json"
)

THRESHOLD = 0.7



# --------------------------------------------------
# 2. INITIALIZE JUDGE MODEL
# --------------------------------------------------

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing.")

judge_model = OpenRouterModel(
    model="openai/gpt-4o-mini",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    generation_kwargs={
        "max_tokens": 2048
    },
)


# --------------------------------------------------
# 3. LOAD GOLDEN DATASET
# --------------------------------------------------

if not GOLDEN_PATH.exists():
    raise FileNotFoundError(
        f"Golden dataset not found: {GOLDEN_PATH}"
    )

with open(GOLDEN_PATH, "r", encoding="utf-8") as file:
    goldens = json.load(file)

print(f"Total golden questions: {len(goldens)}")


# --------------------------------------------------
# 4. INITIALIZE APPLICATION COMPONENTS
# --------------------------------------------------

retriever = get_retriever(k=4)

generation_chain = get_qa_chain()


# --------------------------------------------------
# 5. CREATE DEEPEVAL TEST CASES
# --------------------------------------------------

test_cases = []

for golden in goldens:
    query = golden["query"]
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        raise ValueError(
            f"No documents retrieved for golden question "
            f"{golden.get('id', query)!r}"
        )

    context = format_documents(retrieved_docs)
    answer = generation_chain.invoke(
        {
            "question": query,
            "context": context,
        }
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=[
            document.page_content for document in retrieved_docs
        ],
    )

    test_cases.append(test_case)


# --------------------------------------------------
# 6. DEFINE EVALUATION METRICS
# --------------------------------------------------

metrics = [

    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=True,
    ),

    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge_model,
        include_reason=True,
    ),

]


# --------------------------------------------------
# 7. RUN EVALUATION
# --------------------------------------------------

print("\nStarting ResearchIQ Generator Evaluation...\n")

if not test_cases:
    raise ValueError("No valid test cases available for evaluation.")

result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=False,
        max_concurrent=1,
        throttle_value=3,
    ),
)

print("\nResearchIQ Generator Evaluation Completed.")
