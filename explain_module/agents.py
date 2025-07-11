from typing import List, Union, Literal
from utils.llm import OpenAILLM, NShotLLM, DeepSeekLLM, FastChatLLM
from utils.prompts import REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
from utils.fewshots import PREDICT_EXAMPLES
from memory_module import BrainDB
from datetime import date


class PredictAgent:
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 memory_db: BrainDB = None,
                 memory_symbol: str = None,
                 predict_llm = DeepSeekLLM()
                 ) -> None:
        
        self.memory_db = memory_db
        self.memory_symbol = memory_symbol
        self.ticker = ticker
        self.summary = summary
        self.target = target
        self.prediction = ''

        self.predict_prompt = PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES
        self.llm = predict_llm

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

        if self.memory_db and self.memory_symbol:
            # 记录本次推理过程到记忆库
            self.memory_db.add_memory_short(
                symbol=self.memory_symbol,
                date=date.today(),
                text=f"Summary: {self.summary}\nPrediction: {self.prediction}\nReflection: {getattr(self, 'reflections_str', '')}"
            )
        
        if self.memory_db and self.memory_symbol:
            # 假设刚刚写入的记忆id已知或可查
            # 这里假设最新的id为N-1
            last_id = self.memory_db.short_term_memory.id_generator.current_id - 1
            feedback = 1 if self.is_correct() else -1
            self.memory_db.update_access_count_with_feed_back(
                symbol=self.memory_symbol, ids=[last_id], feedback=feedback
            )

    def prompt_agent(self) -> str:
        return self.llm(self._build_agent_prompt())

    def _build_agent_prompt(self) -> str:
        memory_context = ""
        if self.memory_db and self.memory_symbol:
            # 检索相关记忆（如top3）
            memory_texts, _ = self.memory_db.query_short(
                query_text=self.summary, top_k=3, symbol=self.memory_symbol
            )
            if memory_texts:
                memory_context = "\n".join(memory_texts) + "\n"
        return self.predict_prompt.format(
            ticker=self.ticker,
            examples=self.predict_examples,
            summary=memory_context + self.summary  # 把记忆加到摘要前
        )

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''


class PredictReflectAgent(PredictAgent):
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 predict_llm = DeepSeekLLM(),
                 reflect_llm = DeepSeekLLM()
                 ) -> None:

        super().__init__(ticker, summary, target, predict_llm)
        self.predict_llm = predict_llm
        self.reflect_llm = reflect_llm
        self.reflect_prompt = REFLECT_INSTRUCTION
        self.agent_prompt = PREDICT_REFLECT_INSTRUCTION
        self.reflections = []
        self.reflections_str: str = ''

    def run(self, reset=True) -> None:
        if self.is_finished() and not self.is_correct():
            self.reflect()

        PredictAgent.run(self, reset=reset)

    def reflect(self) -> None:
        print('Reflecting...\n')
        reflection = self.prompt_reflection()
        self.reflections += [reflection]
        self.reflections_str = format_reflections(self.reflections)
        print(self.reflections_str, end="\n\n\n\n")

    def prompt_reflection(self) -> str:
        return self.reflect_llm(self._build_reflection_prompt())

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            ticker = self.ticker,
                            scratchpad = self.scratchpad)

    def _build_agent_prompt(self) -> str:
        prompt = self.agent_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            reflections = self.reflections_str,
                            summary = self.summary)
        return prompt

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
