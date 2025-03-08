import traceback
from typing import Dict, Callable, Any
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import pydantic


SCORE_MIN = 1
SCORE_MAX = 5


SCORER_RESPONSE_DESCRIPTION = f"""
Use the following output JSON format (pay attention to syntax)
```json
{{{{
  "thoughts": [
    "A list of thoughts regards your scoring",
    "like 3-4 sentences"
  ],
  "score": {SCORE_MIN}...{SCORE_MAX}
}}}}
```
{SCORE_MIN} is lowest score, {SCORE_MAX} is highest.
"""


class ScorerResponse(pydantic.BaseModel):
    thoughts: list[str]
    score: int

    @pydantic.field_validator("score")
    @classmethod
    def validate_score(cls, value: int) -> int:
        if not SCORE_MIN <= value <= SCORE_MAX:
            raise ValueError(f"Score must be between {SCORE_MIN} and {SCORE_MAX}")
        return value

    @property
    def score_float(self):
        return (self.score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


class Scoring:
    def __init__(self, llm: BaseChatModel, system_prompt: str | None, user_prompt: str,
                 preprocessors: Dict[str, Callable[[str], str]],
                 max_restarts: int, raise_on_failure: bool, function_name: str):
        prompt_messages = []
        if system_prompt is not None:
            prompt_messages.append(SystemMessagePromptTemplate.from_template(system_prompt))
        prompt_messages.append(HumanMessagePromptTemplate.from_template(
            user_prompt.strip() + "\n" + SCORER_RESPONSE_DESCRIPTION
        ))
        prompt = ChatPromptTemplate(messages=prompt_messages)
        parser = PydanticOutputParser(pydantic_object=ScorerResponse)
        chain = preprocessors | prompt | llm | parser
        self._chain = chain
        self._max_restarts = max_restarts
        self._raise_on_failure = raise_on_failure
        self._function_name = function_name
    
    def score(self, **kwargs) -> float:
        error = None
        for i in range(self._max_restarts):
            try:
                inputs = dict(**kwargs, error=error)
                result: ScorerResponse = self._chain.invoke(inputs)
                return result.score_float
            except Exception as err:
                if i == self._max_restarts - 1:
                    if self._raise_on_failure:
                        raise err
                    else:
                        return 0
                else:
                    error = traceback.format_exc()

    @property
    def score_function(self) -> Callable[..., float]:
        def function(**kwargs) -> float:
            return self.score(**kwargs)
        
        function.__name__ = self._function_name
        function.__qualname__ = self._function_name
        return function
