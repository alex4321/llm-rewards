import functools
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Tuple
from llm_rewards import reasoner_rewards as rewards
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
import pytest


def test_format_reward():
    reward = rewards.build_format_reward()
    assert [0, 1] == reward(
        completions=[
            [
                {
                    "role": "assistant",
                    "content": "Hello, world!"
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": "<think>Okay, let me gree user</think> Hello, {name}!"
                }
            ]
        ]
    )


def llm_cache(function):

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        fname = f"{function.__name__}.sqlite"
        fname_full = os.path.join(os.path.dirname(__file__), "llm_cache", fname)
        cache = SQLiteCache(database_path=fname_full)
        set_llm_cache(cache)
        return function(*args, **kwargs)
    
    return wrapper


@pytest.fixture
def openrouter_api_key() -> str:
    response = os.getenv("OPENROUTER_API_KEY", None)
    assert response is not None, "OPENROUTER_API_KEY is not set"
    return response


@pytest.fixture
def hf_token() -> str:
    response = os.getenv("HFTOKEN", None)
    assert response is not None, "HFTOKEN is not set"
    return response


def get_multiple_run_average_rewards(reward, prompt, completions, other_params, run_count):
    generation_count = len(completions)
    prompts = [prompt] * generation_count
    rewards_average = [0] * generation_count
    for i in range(run_count):
        rewards = reward(prompts=prompts, completions=completions, **other_params)
        for i in range(generation_count):
            rewards_average[i] += rewards[i] / run_count
    return rewards_average


def build_llm(openrouter_api_key: str, hf_token: str) -> Tuple[ChatOpenAI, PreTrainedTokenizerFast]:
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        model="meta-llama/llama-3.3-70b-instruct",
        temperature=0.1
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-3.3-70b-instruct", token=hf_token)
    return llm, tokenizer


def assert_rewards_average(rewards_average, expected_rewards_average):
    threshold = 0.3
    for item_average, item_expected in zip(rewards_average, expected_rewards_average):
        min_expected = max(item_expected - threshold, 0)
        max_expected = min(item_expected + threshold, 1)
        assert min_expected <= item_average <= max_expected, f"Got {item_average} instead of something like {item_expected}. " + \
            f"All reward it got is {rewards_average} while expected is {expected_rewards_average}"


@llm_cache
def test_qa_correct_answer_reward(openrouter_api_key: str, hf_token: str):
    llm, tokenizer = build_llm(openrouter_api_key, hf_token)
    prompt = [
        {
            "role": "user",
            "content": "What is the capital of France?"
        },
    ]
    completions = [
        [
            {
                "role": "assistant",
                "content": "Paris"
            }
        ],
        [
            {
                "role": "assistant",
                "content": "London"
            }
        ],
        [
            {
                "role": "assistant",
                "content": "ABCDEF"
            }
        ]
    ]
    reference_answer = ["Paris", "Paris", "Paris"]
    rewards_average = get_multiple_run_average_rewards(
        rewards.build_qa_correct_answer_reward(llm, tokenizer, max_length=100, max_restarts=5, raise_on_failure=True),
        prompt,
        completions,
        {"reference_answer": reference_answer},
        run_count=5
    )
    assert_rewards_average(rewards_average, [1, 0, 0])


@llm_cache
def test_qa_consistent_thoughts_reward(openrouter_api_key: str, hf_token: str):
    llm, tokenizer = build_llm(openrouter_api_key, hf_token)
    prompt = [
        {
            "role": "user",
            "content": "Which of these countries capital name starts with 'L' - France or UK?"
        },
    ]
    completions = [
        [
            {
                "role": "assistant",
                "content": "<think>France capital is Paris, and UK capital is London; starts with L - means London; London - means UK; so answer is UK</think> UK"
            },
        ],
        [
            {
                "role": "assistant",
                "content": "<think>France capital is Paris, and UK capital is London, starts with L - means London; London - means UK; so answer is UK</think> France"
            },
        ],
        [
            {
                "role": "assistant",
                "content": "<think>Let's think about self-awareness instead</think>"
            },
        ],
    ]
    rewards_average = get_multiple_run_average_rewards(
        rewards.build_qa_consistent_thoughts_reward(llm, tokenizer, max_length=100, max_restarts=5, raise_on_failure=True),
        prompt,
        completions,
        {},
        run_count=5
    )
    assert_rewards_average(rewards_average, [1, 0, 0])
