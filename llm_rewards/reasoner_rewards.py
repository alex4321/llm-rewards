from typing import Callable, Dict, List, Any
import operator
import functools
import concurrent.futures
from transformers import PreTrainedTokenizerFast
from langchain.chat_models.base import BaseChatModel
from .scoring import Scoring


def batch_function(f: Callable[..., float]) -> Callable[..., List[float]]:
    @functools.wraps(f)
    def _inner(**kwargs):
        keys = sorted(kwargs.keys())
        count = len(kwargs[keys[0]])
        with concurrent.futures.ThreadPoolExecutor() as pool:
            scores = pool.map(
                lambda i: f(
                    **{
                        key: kwargs[key][i]
                        for key in keys
                    },
                    i=i,
                ),
                range(count)
            )
            return list(scores)

    return _inner


# region Formatting rewards
def build_think_length_reward(min_length: int, max_length: int, tokenizer: PreTrainedTokenizerFast) -> Callable[..., float]:
    def think_length_reward(completions, **kwargs) -> float:
        message, = completions
        content = message["content"].strip()
        thoughts = content.split("<think>")[-1].split("</think>")[0]
        length = len(tokenizer.tokenize(thoughts))
        if length < min_length:
            return max(0.0, length / min_length)
        elif length > max_length:
            return max(0.0, 0.5 - 0.5 * (length - max_length) / max_length)
        else:
            return 1.0 - 0.5 * (length - min_length) / (max_length - min_length)
        
    return batch_function(think_length_reward)


def build_format_reward() -> Callable[..., float]:
    def format_reward(completions, **kwargs) -> float:
        message, = completions
        content = message["content"].strip()
        if not content.startswith("<think>"):
            return 0.0
        if not "</think>" in content:
            return 0.5
        if content.endswith("</think>"):
            return 0.5
        return 1.0
    
    return batch_function(format_reward)


def build_model_function_name(llm: BaseChatModel, function_name: str) -> str:
    if hasattr(llm, "model_name"):
        return f"{llm.model_name}_{function_name}"
    else:
        return function_name
# endregion Formatting rewards


# region Helpers
def stringify_session(chat_session: List[Dict[str, str]], tokenizer: PreTrainedTokenizerFast, max_length: int) -> str:
    transcript_lines = [
        f"{item['role']}: {item['content']}"
        for item in chat_session
    ]
    transcript_line_sizes = [
        len(tokenizer.tokenize(line))
        for line in transcript_lines
    ]
    while True:
        total_len = sum(transcript_line_sizes)
        if total_len == 0:
            break
        if total_len <= max_length:
            break
        index = len(transcript_lines) // 2
        del transcript_lines[index]
        del transcript_line_sizes[index]
    return "\n\n".join(transcript_lines)


def extract_thoughts(completions: List[Dict[str, str]]) -> str | None:
    message, = completions
    content = message["content"]
    if "<think>" not in content:
        return None
    if "</think>" not in content:
        return None
    return content.split("<think>")[-1].split("</think>")[0]


def extract_response(completions: List[Dict[str, str]]) -> str | None:
    message, = completions
    content = message["content"]
    return content.split("</think>")[-1]
# endregion Helpers


# region QA rewards
_REWARD_QA_CORRECT_ANSWER_SYSTEM = "You're QA response correctness checking system"
_REWARD_QA_CORRECT_ANSWER_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
{answer}
```
As well as reference response:
```
{reference_answer}
```
Score the correctness of the model final response - it should be same as reference one or equal.
From "model was utterly wrong" to "totally aligned with the reference response".
""".strip()


def build_qa_correct_answer_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        system_prompt=_REWARD_QA_CORRECT_ANSWER_SYSTEM,
        user_prompt=_REWARD_QA_CORRECT_ANSWER_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "reference_answer": operator.itemgetter("reference_answer"),
        },
        function_name=build_model_function_name(llm, "qa_correct_answer"),
        **kwargs
    )
    return batch_function(scorer.score_function)


_REWARD_THOUGHTS_CONSISTENT_SYSTEM = """
You're a QA system reasoning consistency checker
""".strip()
_REWARD_THOUGHTS_CONSISTENT_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Tell if model have a well-readable correct step-by-step logic decomposing the task and leading to the answer.
From "logic is either wrong or incomprehensible" to "totally fine logic".
""".strip()


def build_qa_consistent_thoughts_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        system_prompt=_REWARD_THOUGHTS_CONSISTENT_SYSTEM,
        user_prompt=_REWARD_THOUGHTS_CONSISTENT_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        function_name=build_model_function_name(llm, "qa_consistent_thoughts"),
        **kwargs,
    )
    return batch_function(scorer.score_function)


_REWARD_QA_CORNER_CASES_SYSTEM = """
You're a QA system corner cases checker
""".strip()
_REWARD_QA_CORNER_CASES_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Check if thoughts and response covers all the corner cases of the task correctly.
From "corner cases are either not covered or covered incorrectly" to "corner cases are covered correctly".
""".strip()


def build_qa_corner_cases_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        system_prompt=_REWARD_QA_CORNER_CASES_SYSTEM,
        user_prompt=_REWARD_QA_CORNER_CASES_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        function_name=build_model_function_name(llm, "qa_corner_cases"),
        **kwargs
    )
    return batch_function(scorer.score_function)


_REWARD_QA_LANGUAGE_STYLE_SYSTEM = """
You're a language style checker
""".strip()
_REWARD_QA_LANGUAGE_STYLE_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Score the language style of the model final response
From "gibberish" to "perfectly fine language".
""".strip()
def build_qa_language_style_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        system_prompt=_REWARD_QA_LANGUAGE_STYLE_SYSTEM,
        user_prompt=_REWARD_QA_LANGUAGE_STYLE_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        function_name=build_model_function_name(llm, "qa_language_style"),
        **kwargs
    )
    return batch_function(scorer.score_function)


def build_qa_rewards(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> List[Callable[..., float]]:
    return [
        build_qa_correct_answer_reward(llm, tokenizer, max_length, **kwargs),
        build_qa_consistent_thoughts_reward(llm, tokenizer, max_length, **kwargs),
        build_qa_corner_cases_reward(llm, tokenizer, max_length, **kwargs),
        build_qa_language_style_reward(llm, tokenizer, max_length, **kwargs),
    ]
# endregion QA rewards


# region Roleplay rewards
_REWARD_ROLEPLAY_FITS_CHARACTER_SYSTEM = """
You are a roleplay quality measurement system
""".strip()
_REWARD_ROLEPLAY_FITS_CHARACTER_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Tell if these thoughts and response style fits character and system instructions well
(1 - means thoughts and response is either not aligned with character / instructions, 5 - perfectly fine).
""".strip()


def build_roleplay_fits_character_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        function_name=build_model_function_name(llm, "roleplay_fits_character"),
        system_prompt=_REWARD_ROLEPLAY_FITS_CHARACTER_SYSTEM,
        user_prompt=_REWARD_ROLEPLAY_FITS_CHARACTER_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        **kwargs
    )
    return batch_function(scorer.score_function)


_REWARD_ROLEPLAY_ENVIRONMENT_INTEGRITY_SYSTEM = """
You are a roleplay quality measurement system
""".strip()
_REWARD_ROLEPLAY_ENVIRONMENT_INTEGRITY_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Tell if these thoughts and response fits the environment described earlier
(1 - means thoughts and response contradicts the environment, 5 - perfectly fine).
""".strip()


def build_roleplay_environment_integrity_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        function_name=build_model_function_name(llm, "roleplay_environment_integrity"),
        system_prompt=_REWARD_ROLEPLAY_ENVIRONMENT_INTEGRITY_SYSTEM,
        user_prompt=_REWARD_ROLEPLAY_ENVIRONMENT_INTEGRITY_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        **kwargs
    )
    return batch_function(scorer.score_function)


_REWARD_ROLEPLAY_LANGUAGE_STYLE_SYSTEM = """
You are a roleplay quality measurement system
""".strip()
_REWARD_ROLEPLAY_LANGUAGE_STYLE_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Tell if these thoughts and response (language especially) fits with the character earlier description regards the language style and sounds native
(1 - gibberish or non-fitting language, 5 - perfectly aligned with character).
""".strip()


def build_roleplay_language_style_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        function_name=build_model_function_name(llm, "roleplay_language_style"),
        system_prompt=_REWARD_ROLEPLAY_LANGUAGE_STYLE_SYSTEM,
        user_prompt=_REWARD_ROLEPLAY_LANGUAGE_STYLE_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        **kwargs
    )
    return batch_function(scorer.score_function)


_REWARD_ROLEPLAY_CONSISTENT_SYSTEM = """
You are a roleplay quality measurement system
""".strip()
_REWARD_ROLEPLAY_CONSISTENT_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Tell if the thoughts and response are consistent with each other
(1 - means thoughts and response contradicts each other, 5 - perfectly consistent).
""".strip()


def build_roleplay_consistent_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        function_name=build_model_function_name(llm, "roleplay_consistent"),
        system_prompt=_REWARD_ROLEPLAY_CONSISTENT_SYSTEM,
        user_prompt=_REWARD_ROLEPLAY_CONSISTENT_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        **kwargs
    )
    return batch_function(scorer.score_function)


def build_roleplay_rewards(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> List[Callable[..., float]]:
    return [
        build_roleplay_fits_character_reward(llm, tokenizer, max_length, **kwargs),
        build_roleplay_environment_integrity_reward(llm, tokenizer, max_length, **kwargs),
        build_roleplay_language_style_reward(llm, tokenizer, max_length, **kwargs),
        build_roleplay_consistent_reward(llm, tokenizer, max_length),
    ]
# endregion Roleplay rewards


# region Creative writing rewards
_REWARD_CREATIVE_WRITING_FIT_INSTRUCTION_SYSTEM = """
You are a creative writing quality measurement system
""".strip()
_REWARD_CREATIVE_WRITING_FIT_INSTRUCTION_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Tell if these thoughts and response fits the instructions well
(1 - means thoughts and response contradicts the instructions, 5 - perfectly fine).
""".strip()


def build_creative_writing_fit_instruction_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        function_name=build_model_function_name(llm, "creative_writing_fit_instruction"),
        system_prompt=_REWARD_CREATIVE_WRITING_FIT_INSTRUCTION_SYSTEM,
        user_prompt=_REWARD_CREATIVE_WRITING_FIT_INSTRUCTION_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        **kwargs
    )
    return batch_function(scorer.score_function)


_REWARD_CREATIVE_WRITING_FIT_STYLE_SYSTEM = """
You are a creative writing quality measurement system
""".strip()
_REWARD_CREATIVE_WRITING_FIT_STYLE_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Tell if the language style of thoughts and response feels native and fits the instructions
(1 - gibberish, dry or non-fitting language, 5 - perfectly aligned with instructions).
""".strip()


def build_creative_writing_fit_style_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        function_name=build_model_function_name(llm, "creative_writing_fit_style"),
        system_prompt=_REWARD_CREATIVE_WRITING_FIT_STYLE_SYSTEM,
        user_prompt=_REWARD_CREATIVE_WRITING_FIT_STYLE_INSTRUCTION,
        preprocessors={ 
            "i": operator.itemgetter("i"),
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        **kwargs
    )
    return batch_function(scorer.score_function)


_REWARD_CREATIVE_WRITING_CONSISTENT_SYSTEM = """
You are a creative writing quality measurement system
""".strip()
_REWARD_CREATIVE_WRITING_CONSISTENT_INSTRUCTION = """
Given this chat history:
```
{history}
```
And this model final response:
```
<think>
{thoughts}
</think>
{answer}
```
Tell if the thoughts and response are consistent with each other
(1 - means thoughts and response contradicts each other, 5 - perfectly consistent).
""".strip()


def build_creative_writing_consistent_reward(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> Callable[..., float]:
    scorer = Scoring(
        llm=llm,
        function_name=build_model_function_name(llm, "creative_writing_consistent"),
        system_prompt=_REWARD_CREATIVE_WRITING_CONSISTENT_SYSTEM,
        user_prompt=_REWARD_CREATIVE_WRITING_CONSISTENT_INSTRUCTION,
        preprocessors={
            "history": lambda data: stringify_session(data["prompts"], tokenizer=tokenizer, max_length=max_length),
            "answer": lambda data: extract_response(data["completions"]),
            "thoughts": lambda data: extract_thoughts(data["completions"]),
        },
        **kwargs
    )
    scorer_function = scorer.function()
    return batch_function(scorer_function)


def build_creative_writing_rewards(llm: BaseChatModel, tokenizer: PreTrainedTokenizerFast, max_length: int, **kwargs: dict) -> List[Callable[..., float]]:
    return [
        build_creative_writing_fit_instruction_reward(llm, tokenizer, max_length, **kwargs),
        build_creative_writing_fit_style_reward(llm, tokenizer, max_length, **kwargs),
        build_creative_writing_consistent_reward(llm, tokenizer, max_length, **kwargs),
    ]
# endregion Creative writing rewards
