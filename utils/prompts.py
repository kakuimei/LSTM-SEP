SUMMARIZE_INSTRUCTION = """Given a list of tweets, summarize all key facts regarding {ticker} stock.
Here are some examples (STRICTLY FOLLOW the format below, do NOT add any introduction, explanation, or bullet styling like "*", "-", or numbering):
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
Given the following daily tweet summaries for {ticker} during the week from {week_start} to {week_end}, write a concise but comprehensive weekly report.

Requirements:
1. Start with the line: "Weekly Report:".
2. Use numbered bullet points (1., 2., 3., …). No free-form narrative outside this structure.
3. Capture key facts, major events, and notable trends across the whole week (not just repetition of daily items).
4. Avoid generic filler text. Do not add introductions or conclusions outside "Weekly Report".
5. The tone should match the examples below.

Here are some examples:
{examples}
(END OF EXAMPLES)

Daily Summaries:
{daily_summaries}

Weekly Report:
"""

MONTHLY_SUMMARIZE_INSTRUCTION = """
Given the weekly summaries for {ticker} during the month from {month_start} to {month_end}, write a comprehensive monthly report.

Requirements:
1. Start the output with: "Monthly Report:".
2. Use numbered bullet points (1., 2., 3., …) OR short paragraphs in a consistent format.
3. Capture major themes and cumulative trends across the whole month (not just repeating each weekly summary).
4. Cover key business developments, financial performance, strategic moves, market sentiment, and recurring issues.
5. Do NOT add generic introductions, conclusions, or commentary outside the "Monthly Report".
6. Follow the style of the examples below.

Here are some examples:
{examples}
(END OF EXAMPLES)

Weekly Summaries:
{weekly_summaries}

Monthly Report:
"""