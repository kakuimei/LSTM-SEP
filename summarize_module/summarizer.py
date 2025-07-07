from utils.llm import OpenAILLM
from utils.prompts import SUMMARIZE_INSTRUCTION
from utils.fewshots import SUMMARIZE_EXAMPLES
import tiktoken
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

class DeepSeekSummarizer:
    def __init__(self, model="deepseek-chat"):
        # DeepSeek API 与 OpenAI SDK 兼容，可通过 base_url 使用
        self.client = OpenAI(api_key='sk-8dd3b65ec33940bd978b195d462189df', base_url="https://api.deepseek.com")
        self.model = model  # 可选 "deepseek-chat" 或 "deepseek-reasoner"
        self.summarize_prompt = SUMMARIZE_INSTRUCTION
        self.summarize_examples = SUMMARIZE_EXAMPLES
        self.enc = tiktoken.get_encoding("cl100k_base")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def llm_call(self, prompt):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return resp.choices[0].message.content

    def get_summary(self, ticker, tweets):
        if not tweets:
            return None
        prompt = self.summarize_prompt.format(
            ticker=ticker,
            examples=self.summarize_examples,
            tweets="\n".join(tweets)
        )
        # 保证 prompt token 不超过模型上下文限制（近似 16k token）
        while len(self.enc.encode(prompt)) > 16000:
            tweets = tweets[:-1]
            prompt = self.summarize_prompt.format(
                ticker=ticker,
                examples=self.summarize_examples,
                tweets="\n".join(tweets)
            )
        return self.llm_call(prompt)

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