from utils.llm import OpenAILLM
from utils.prompts import SUMMARIZE_INSTRUCTION, WEEKLY_SUMMARIZE_INSTRUCTION, MONTHLY_SUMMARIZE_INSTRUCTION
from utils.fewshots import SUMMARIZE_EXAMPLES, WEEKLY_SUMMARIZE_EXAMPLES, MONTHLY_SUMMARIZE_EXAMPLES
import tiktoken
import re
from utils.llm import DeepSeekLLM
class DeepSeekSummarizer:
    def __init__(self, model="deepseek-chat"):
        self.llm = DeepSeekLLM(model=model)
        self.summarize_prompt = SUMMARIZE_INSTRUCTION
        self.summarize_examples = SUMMARIZE_EXAMPLES

        self.weekly_summarize_prompt = WEEKLY_SUMMARIZE_INSTRUCTION
        self.weekly_summarize_examples = WEEKLY_SUMMARIZE_EXAMPLES

        self.monthly_summarize_prompt = MONTHLY_SUMMARIZE_INSTRUCTION
        self.monthly_summarize_examples = MONTHLY_SUMMARIZE_EXAMPLES

        # 使用通用编码器 cl100k_base
        self.enc = tiktoken.get_encoding("cl100k_base")

    def get_summary(self, ticker, tweets):
        if not tweets:
            return None
        prompt = self.summarize_prompt.format(
            ticker=ticker,
            examples=self.summarize_examples,
            tweets="\n".join(tweets)
        )
        # 保证 token 数不超限制
        while len(self.enc.encode(prompt)) > 16000:
            tweets = tweets[:-1]
            prompt = self.summarize_prompt.format(
                ticker=ticker,
                examples=self.summarize_examples,
                tweets="\n".join(tweets)
            )
        return self.llm(prompt)

    def get_weekly_summary(self, ticker, week_start, week_end, raw_weekly_report):
        summary = None
        if raw_weekly_report:
            prompt = self.weekly_summarize_prompt.format(
                                    ticker = ticker,
                                    week_start = week_start,
                                    week_end = week_end,
                                    daily_summaries = raw_weekly_report
                                    )
            summary = self.llm(prompt)

        return summary

    def get_monthly_summary(self, ticker, month_start, month_end, raw_monthly_report):
        summary = None
        if raw_monthly_report:
            prompt = self.weekly_summarize_prompt.format(
                                    ticker = ticker,
                                    month_start = month_start,
                                    month_end = month_end,
                                    weekly_summaries = raw_monthly_report
                                    )
            summary = self.llm(prompt)

        return summary
    
    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)
class Summarizer:
    def __init__(self):
        self.summarize_prompt = SUMMARIZE_INSTRUCTION
        self.summarize_examples = SUMMARIZE_EXAMPLES
        self.llm = OpenAILLM()
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

    def get_summary(self, ticker, tweets):
        summary = None
        if tweets != []:
            prompt = self.summarize_prompt.format(
                                    ticker = ticker,
                                    examples = self.summarize_examples,
                                    tweets = "\n".join(tweets))

            while len(self.enc.encode(prompt)) > 16385:
                tweets = tweets[:-1]
                prompt = self.summarize_prompt.format(
                                        ticker = ticker,
                                        examples = self.summarize_examples,
                                        tweets = "\n".join(tweets))

            summary = self.llm(prompt)

        return summary

    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)