SUMMARIZE_INSTRUCTION = """Given a list of tweets, summarize all key facts regarding {ticker} stock.
Here are some examples:
{examples}
(END OF EXAMPLES)

Tweets:
{tweets}

Facts:"""


PREDICT_INSTRUCTION = """Given a list of facts, estimate their overall impact on the price movement of {ticker} stock. Give your response in this format:
(1) Price Movement, which should be either Positive or Negative.
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

Facts:
{summary}

Price Movement:"""


PREDICT_REFLECT_INSTRUCTION = """Given a list of facts, estimate their overall impact on the price movement of {ticker} stock. Give your response in this format:
(1) Price Movement, which should be either Positive or Negative.
(2) Explanation, which should be in a single, short paragraph.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Facts:
{summary}

Price Movement:"""


REFLECTION_HEADER = 'You have attempted to tackle the following task before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly tackling the given task.\n'


REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to a list of facts to assess their overall impact on the price movement of {ticker} stock. You were unsuccessful in tackling the task because you gave the wrong price movement. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Previous trial:
{scratchpad}

Reflection:"""

WEEKLY_SUMMARIZE_INSTRUCTION = """
Given daily tweet summaries for {ticker} over the week from {week_start} to {week_end}, generate a concise weekly report that:

1. Highlights the top 3–5 market-moving events or themes  
2. Describes the overall sentiment trend  
3. Provides one actionable insight or prediction based on these observations  

Use complete sentences or numbered bullets.  
Daily Summaries:
{daily_summaries}

Weekly Report:
"""

MONTHLY_SUMMARIZE_INSTRUCTION = """
Given the weekly summaries for {ticker} over the month from {month_start} to {month_end}, generate a concise monthly report that:

1. Highlights the top 3–5 major market themes or events  
2. Describes the overall sentiment trend across the weeks  
3. Offers one strategic insight or recommendation for the coming month  

Use complete sentences or numbered bullets.  
Weekly Summaries:
{weekly_summaries}

Monthly Report:
"""