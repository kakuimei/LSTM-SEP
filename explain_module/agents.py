from typing import List, Union, Literal, Dict, Any
from utils.llm import OpenAILLM, NShotLLM, DeepSeekLLM, FastChatLLM
from utils.prompts import REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
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
        print(f'DEBUG: Prediction: {self.prediction}, Target: {self.cur_record}')
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
        self.reflections = []
        self.reflections_str: str = ''
        self.reflection_result_series_dict = {}
        self.brain = brain_db
        self.cur_date = date
        self.top_k = 5
        self.character_string = "You are a stock trading agent. You have access to the following information about the stock market and your past trades: \n"

        # logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(
            os.path.join(
                "data",
                "04_model_output_log",
                f"{self.ticker}_run.log",
            ),
            mode="a",
        )
        file_handler.setFormatter(logging_formatter)
        self.logger.addHandler(file_handler)

    def run(self, reset=True) -> None:
        if self.is_finished() and not self.is_correct():
            self.reflect()
            self.step(RunMode.Train)

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

    def __update_access_counter_sub(self, cur_memory, layer_index_name, feedback):
        if cur_memory[layer_index_name] is not None:
            cur_ids = []
            for i in cur_memory[layer_index_name]:
                cur_id = i["memory_index"]
                if cur_id not in cur_ids:
                    cur_ids.append(cur_id)
            self.brain.update_access_count_with_feed_back(
                symbol=self.ticker,
                ids=cur_ids,
                feedback=feedback["feedback"],
            )

    def __update_short_memory_access_counter(
        self,
        feedback: Dict[str, Union[int, date]],
        cur_memory: Dict[str, Any],
    ) -> None:
        if "short_memory_index" in cur_memory:
            self.__update_access_counter_sub(
                cur_memory=cur_memory,
                layer_index_name="short_memory_index",
                feedback=feedback,
            )

    def __update_mid_memory_access_counter(
        self,
        feedback: Dict[str, Union[int, date]],
        cur_memory: Dict[str, Any],
    ) -> None:
        if "middle_memory_index" in cur_memory:
            self.__update_access_counter_sub(
                cur_memory=cur_memory,
                layer_index_name="middle_memory_index",
                feedback=feedback,
            )

    def __update_long_memory_access_counter(
        self,
        feedback: Dict[str, Union[int, date]],
        cur_memory: Dict[str, Any],
    ) -> None:
        if "long_memory_index" in cur_memory:
            self.__update_access_counter_sub(
                cur_memory=cur_memory,
                layer_index_name="long_memory_index",
                feedback=feedback,
            )

    def __update_reflection_memory_access_counter(
        self,
        feedback: Dict[str, Union[int, date]],
        cur_memory: Dict[str, Any],
    ) -> None:
        if "reflection_memory_index" in cur_memory:
            self.__update_access_counter_sub(
                cur_memory=cur_memory,
                layer_index_name="reflection_memory_index",
                feedback=feedback,
            )

    def _update_access_counter(self):
        feedback = {}
        if self.is_correct():
            feedback = {"feedback": 1, "date": self.cur_date}
        else:
            feedback = {"feedback": 0, "date": self.cur_date}
        cur_memory = self.reflection_result_series_dict[self.cur_date]
        self.__update_short_memory_access_counter(
            feedback=feedback, cur_memory=cur_memory
        )
        self.__update_mid_memory_access_counter(
            feedback=feedback, cur_memory=cur_memory
        )
        self.__update_long_memory_access_counter(
            feedback=feedback, cur_memory=cur_memory
        )
        self.__update_reflection_memory_access_counter(
            feedback=feedback, cur_memory=cur_memory
        )
            
    def __query_info_for_reflection(self, run_mode: RunMode):

        self.logger.info(f"Symbol: {self.ticker}\n")

        # query the brain for top-k memories
        cur_short_queried, cur_short_memory_id = self.brain.query_short(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.ticker,
        )
        for cur_id, cur_memory in zip(cur_short_memory_id, cur_short_queried):
            self.logger.info(f"Top-k Short: {cur_id}: {cur_memory}\n")

        cur_mid_queried, cur_mid_memory_id = self.brain.query_mid(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.ticker,
        )
        for cur_id, cur_memory in zip(cur_mid_memory_id, cur_mid_queried):
            self.logger.info(f"Top-k Mid: {cur_id}: {cur_memory}\n")

        cur_long_queried, cur_long_memory_id = self.brain.query_long(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.ticker,
        )
        for cur_id, cur_memory in zip(cur_long_memory_id, cur_long_queried):
            self.logger.info(f"Top-k Long: {cur_id}: {cur_memory}\n")

        cur_reflection_queried, cur_reflection_memory_id = self.brain.query_reflection(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.ticker,
        )
        for cur_id, cur_memory in zip(cur_reflection_memory_id, cur_reflection_queried):
            self.logger.info(f"Top-k Reflection: {cur_id}: {cur_memory}\n")

        if run_mode == RunMode.Train:
            return (
                cur_short_queried,
                cur_short_memory_id,
                cur_mid_queried,
                cur_mid_memory_id,
                cur_long_queried,
                cur_long_memory_id,
                cur_reflection_queried,
                cur_reflection_memory_id,
            )

    def __reflection_on_record(
        self,
        cur_date: date,
        run_mode: RunMode,
        cur_record: Union[float, None] = None,
    ) -> Dict[str, Any]:
        if (run_mode == RunMode.Train) and (not cur_record):
            self.logger.info("No record\n")
            return {}
        # reflection
        if run_mode == RunMode.Train:
            (
                cur_short_queried,
                cur_short_memory_id,
                cur_mid_queried,
                cur_mid_memory_id,
                cur_long_queried,
                cur_long_memory_id,
                cur_reflection_queried,
                cur_reflection_memory_id,
            ) = self.__query_info_for_reflection(  # type: ignore
                run_mode=run_mode
            )
            reflection_result = trading_reflection(
                cur_date=cur_date,
                symbol=self.ticker,
                run_mode=run_mode,
                reflect_llm=self.reflect_llm,
                short_memory=cur_short_queried,
                short_memory_id=cur_short_memory_id,
                mid_memory=cur_mid_queried,
                mid_memory_id=cur_mid_memory_id,
                long_memory=cur_long_queried,
                long_memory_id=cur_long_memory_id,
                reflection_memory=cur_reflection_queried,
                reflection_memory_id=cur_reflection_memory_id,
                future_record=cur_record,  # type: ignore
                logger=self.logger,
            )

        if (reflection_result is not {}) and ("summary_reason" in reflection_result):
            self.brain.add_memory_reflection(
                symbol=self.ticker,
                date=cur_date,
                text=reflection_result["summary_reason"],
            )
        else:
            self.logger.info("No reflection result , not converged\n")
        return reflection_result

    def _reflect(
        self,
        cur_date: date,
        run_mode: RunMode,
        cur_record: Union[float, None] = None,
    ) -> None:
        if run_mode == RunMode.Train:
            reflection_result_cur_date = self.__reflection_on_record(
                cur_date=cur_date,
                cur_record=cur_record,
                run_mode=run_mode,
            )
        self.reflection_result_series_dict[cur_date] = reflection_result_cur_date
        if run_mode == RunMode.Train:
            self.logger.info(
                f"{self.ticker}-Day {cur_date}\nreflection summary: {reflection_result_cur_date.get('summary_reason')}\n\n"
            )


    def step(
        self,
        run_mode: RunMode,
    ) -> None:
        # mode assertion
        if run_mode not in [RunMode.Train, RunMode.Test]:
            raise ValueError("run_mode should be either Train or Test")
        self._reflect(
            cur_date=self.cur_date,
            run_mode=run_mode,
            cur_record=self.cur_record,
        )
        # update the access counter if need to
        self._update_access_counter()
        print('update access counter completed')
        # brain step
        self.brain.step()
        print('DEBUG: Agent step completed')

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        path = os.path.join(path, self.agent_name)
        if os.path.exists(path):
            if force:
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Path {path} already exists")
        os.mkdir(path)
        os.mkdir(os.path.join(path, "brain"))
        state_dict = {
            "agent_name": self.agent_name,
            "character_string": self.character_string,
            "top_k": self.top_k,
            "counter": self.counter,
            "ticker": self.ticker,
            "portfolio": self.portfolio,
            "chat_config": self.chat_config_save,
            "reflection_result_series_dict": self.reflection_result_series_dict,  #
            "access_counter": self.access_counter,
        }
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)
        self.brain.save_checkpoint(path=os.path.join(path, "brain"), force=force)


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def EM(prediction, sentiment) -> bool:
    return prediction.lower() == sentiment.lower()
