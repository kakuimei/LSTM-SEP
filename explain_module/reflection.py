# sourcery skip: dont-import-test-modules
from rich import print
import logging
import guardrails as gd
from datetime import date
from .run_type import RunMode
from pydantic import BaseModel, Field
from httpx import HTTPStatusError
from guardrails.validators import ValidChoices
from typing import List, Callable, Dict, Union, Any, Tuple
from .chat import LongerThanContextError
from .prompts import (
    short_memory_id_desc,
    mid_memory_id_desc,
    long_memory_id_desc,
    reflection_memory_id_desc,
    train_prompt,
    train_memory_id_extract_prompt,
    train_trade_reason_summary,
    train_investment_info_prefix,

)
import re



def _format_memories(
    short_memory: Union[List[str], None] = None,
    short_memory_id: Union[List[int], None] = None,
    mid_memory: Union[List[str], None] = None,
    mid_memory_id: Union[List[int], None] = None,
    long_memory: Union[List[str], None] = None,
    long_memory_id: Union[List[int], None] = None,
    reflection_memory: Union[List[str], None] = None,
    reflection_memory_id: Union[List[int], None] = None,
) -> Tuple[
    List[str],
    List[int],
    List[str],
    List[int],
    List[str],
    List[int],
    List[str],
    List[int],
]:
    # add placeholder information if not memory is available
    # each memory has a duplicate because guardrails::ValidChoices does not support single choice
    if (short_memory is None) or len(short_memory) == 0:
        short_memory = ["No short-term information.", "No short-term information."]
        short_memory_id = [-1, -1]
    elif len(short_memory) == 1:
        short_memory = [short_memory[0], short_memory[0]]
        short_memory_id = [short_memory_id[0], short_memory_id[0]]  # type: ignore
    if (mid_memory is None) or len(mid_memory) == 0:
        mid_memory = ["No mid-term information.", "No mid-term information."]
        mid_memory_id = [-1, -1]
    elif len(mid_memory) == 1:
        mid_memory = [mid_memory[0], mid_memory[0]]
        mid_memory_id = [mid_memory_id[0], mid_memory_id[0]]  # type: ignore
    if (long_memory is None) or len(long_memory) == 0:
        long_memory = ["No long-term information.", "No long-term information."]
        long_memory_id = [-1, -1]
    elif len(long_memory) == 1:
        long_memory = [long_memory[0], long_memory[0]]
        long_memory_id = [long_memory_id[0], long_memory_id[0]]  # type: ignore
    if (reflection_memory is None) or len(reflection_memory) == 0:
        reflection_memory = [
            "No reflection-term information.",
            "No reflection-term information.",
        ]
        reflection_memory_id = [-1, -1]
    elif len(reflection_memory) == 1:
        reflection_memory = [reflection_memory[0], reflection_memory[0]]
        reflection_memory_id = [reflection_memory_id[0], reflection_memory_id[0]]  # type: ignore

    return (
        short_memory,
        short_memory_id,
        mid_memory,
        mid_memory_id,
        long_memory,
        long_memory_id,
        reflection_memory,
        reflection_memory_id,
    )


def build_investment_info_prompt(
    cur_date: date,
    symbol: str,
    future_record: Dict[str, float | str],
    short_memory: List[str],
    short_memory_id: List[int],
    mid_memory: List[str],
    mid_memory_id: List[int],
    long_memory: List[str],
    long_memory_id: List[int],
    reflection_memory: List[str],
    reflection_memory_id: List[int],
) -> str:
    investment_info = train_investment_info_prefix.format(
        cur_date=cur_date, symbol=symbol, future_record=future_record
    )

    def format_memories(title, mems, ids):
        if mems:
            lines = [f"{i}. {text.strip()}" for i, text in zip(ids, mems)]
            return f"{title}\n" + "\n".join(lines) + "\n\n"
        return ""

    investment_info += format_memories("The short-term information:", short_memory, short_memory_id)
    investment_info += format_memories("The mid-term information:", mid_memory, mid_memory_id)
    investment_info += format_memories("The long-term information:", long_memory, long_memory_id)
    investment_info += format_memories("The reflection-term information:", reflection_memory, reflection_memory_id)

    return investment_info



def trading_reflection(
    cur_date: date,
    reflect_llm: Callable[[str], str],
    symbol: str,
    run_mode: RunMode,
    logger: logging.Logger,
    future_record: Union[Dict[str, float | str], None] = None,
    short_memory: Union[List[str], None] = None,
    short_memory_id: Union[List[int], None] = None,
    mid_memory: Union[List[str], None] = None,
    mid_memory_id: Union[List[int], None] = None,
    long_memory: Union[List[str], None] = None,
    long_memory_id: Union[List[int], None] = None,
    reflection_memory: Union[List[str], None] = None,
    reflection_memory_id: Union[List[int], None] = None,
) -> Dict[str, Any]:
    # format memories
    (
        short_memory,
        short_memory_id,
        mid_memory,
        mid_memory_id,
        long_memory,
        long_memory_id,
        reflection_memory,
        reflection_memory_id,
    ) = _format_memories(
        short_memory=short_memory,
        short_memory_id=short_memory_id,
        mid_memory=mid_memory,
        mid_memory_id=mid_memory_id,
        long_memory=long_memory,
        long_memory_id=long_memory_id,
        reflection_memory=reflection_memory,
        reflection_memory_id=reflection_memory_id,
    )

    investment_info = build_investment_info_prompt(
        cur_date=cur_date,
        symbol=symbol,
        future_record=future_record,
        short_memory=short_memory,
        short_memory_id=short_memory_id,
        mid_memory=mid_memory,
        mid_memory_id=mid_memory_id,
        long_memory=long_memory,
        long_memory_id=long_memory_id,
        reflection_memory=reflection_memory,
        reflection_memory_id=reflection_memory_id,
    )

    if run_mode == RunMode.Train:
        prompt = train_prompt.format(investment_info=investment_info)

    try:
        response = reflect_llm(prompt)
        logger.info(f"Raw LLM output:\n{response}")

        match = re.search(r"summary_reason\s*:\s*(.+)", response)
        reason = match.group(1).strip() if match else "[Invalid response]"

        # 构造回传结构（注意：你可以根据实际格式自定义）
        return {
            "summary_reason": reason,
            "short_memory_index": [{"memory_index": i} for i in short_memory_id],
            "middle_memory_index": [{"memory_index": i} for i in mid_memory_id],
            "long_memory_index": [{"memory_index": i} for i in long_memory_id],
            "reflection_memory_index": [{"memory_index": i} for i in reflection_memory_id],
        }

    except Exception as e:
        logger.error(f"Reflection failed: {e}")
        return {
            "summary_reason": "[Exception]",
            "short_memory_index": None,
            "middle_memory_index": None,
            "long_memory_index": None,
            "reflection_memory_index": None,
        }
