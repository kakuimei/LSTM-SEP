from typing import List, Union, Literal, Dict, Any
from utils.llm import OpenAILLM, NShotLLM, DeepSeekLLM, FastChatLLM
from utils.prompts import REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER, REFLECTION_QUERY_INSTRUCTION 
from utils.fewshots import PREDICT_EXAMPLES
from memory_module.memorydb import BrainDB
from datetime import date
from .reflection import trading_reflection
from .run_type import RunMode
import logging
import os
import shutil
import pickle
from unittest.mock import MagicMock
from typing import Optional

class PredictAgent:
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 predict_llm: Optional[DeepSeekLLM] = None,
                 ) -> None:
        
        self.ticker = ticker
        self.summary = summary
        self.cur_record = target
        self.prediction = ''

        self.predict_prompt = PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES
        self.llm = predict_llm or DeepSeekLLM()

        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        facts = "Facts:\n" + self.summary + "\n\nPrice Movement: "
        self.scratchpad += facts
        print(facts, end="")

        self.scratchpad += self.prompt_agent()
        response = self.scratchpad.split('Price Movement: ')[-1]
        self.prediction = response.split()[0]
        print(response, end="\n\n\n\n")

        self.finished = True
    def prompt_agent(self) -> str:
        return self.llm(self._build_agent_prompt())
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        print(f'[Info] Prediction: {self.prediction}, Target: {self.cur_record}')
        if self.cur_record is None or self.prediction is None:
            return False
        return EM(self.cur_record, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''


class PredictReflectAgent(PredictAgent):
    def __init__(self,
                 ticker: str,
                 date: date,
                 summary: str,
                 target: str,
                 predict_llm: Optional[DeepSeekLLM] = None,
                 reflect_llm: Optional[DeepSeekLLM] = None,
                 brain_db: Optional[BrainDB] = None
                 ) -> None:

        super().__init__(ticker, summary, target, predict_llm)
        self.predict_llm = predict_llm or DeepSeekLLM()
        self.reflect_llm = reflect_llm or DeepSeekLLM()
        self.reflect_prompt = REFLECT_INSTRUCTION
        self.agent_prompt = PREDICT_REFLECT_INSTRUCTION
        self.reflect_query_prompt = REFLECTION_QUERY_INSTRUCTION
        self.reflections = []
        self.reflections_str: str = ''
        self.reflection_result_series_dict = {}
        self.brain_db = brain_db if brain_db else BrainDB()
        self.cur_date = date
    
    def run(self, reset=True) -> None:
        if self.is_finished() and not self.is_correct():
            self.reflect()

        PredictAgent.run(self, reset=reset)

    def reflect(self) -> None:
        print('[Info] Reflecting...')
        reflection = self.prompt_reflection()
        print(f'[Info] Generated Reflection: \n{reflection}')
        self.brain_db.add_memory_reflection(
                symbol=self.ticker,
                date=self.cur_date,
                text=reflection
            )
        print(f'[Info] Added reflection to reflection memory for {self.ticker} on {self.cur_date}')
        query_prompt = self.reflect_query_prompt.format(
            ticker=self.ticker,
            scratchpad = self.scratchpad,
            date_str = self.cur_date
        )
        query_text = self.reflect_llm(query_prompt)
        print(f'[Info] Reflection Query: \n{query_text}')
        texts, _ = self.brain_db.query_reflection(
                query_text=query_text,
                top_k=5,
                symbol=self.ticker
            )
        self.reflections_str = format_reflections(texts)
        print(f'[Info] Reflection: \n{self.reflections_str}',end="\n\n\n\n")

    def prompt_reflection(self) -> str:
        return self.reflect_llm(self._build_reflection_prompt())

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            ticker = self.ticker,
                            scratchpad = self.scratchpad)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            reflections = self.reflections_str,
                            summary = self.summary)

    def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
        self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
        PredictAgent.run(self, reset=reset)

def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def EM(prediction, sentiment) -> bool:
    return prediction.lower() == sentiment.lower()
