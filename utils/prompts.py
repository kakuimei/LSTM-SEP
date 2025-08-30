REFLECTION_QUERY_INSTRUCTION = """
You are drafting FAISS-style search snippets to retrieve prior reflections that can prevent repeating the same mistake for {ticker} on {date_str}.

Context:
{scratchpad}

Output rules:
- Return 6–8 lines, each a compact keyword string (<= 12 tokens).
- No full sentences. No punctuation except spaces and #/$.
- Each line MUST include: {ticker} and {date_str}.
- Cover multiple intents across the lines: supply chain; earnings/guidance; products/launch; regulation/lawsuit; analysts/price targets; partnerships/M&A; sentiment/flows.
- Prefer salient entities/phrases from the facts (e.g., Foxconn, guidance cut, options flow).
- Include 1–2 lines that encode the label mismatch (e.g., "negative vs positive", "misread premarket drop").

Return ONLY the lines, one per line, no numbering or extra text.
"""

QUERY_INSTRUCTION = """
You are writing FAISS search queries to retrieve the most relevant prior memory for stock analysis.
Constraints:
- No full sentences; return compact search strings (<= 12 tokens each).
- Include the ticker {ticker} and the date {date_str}.
- Cover different intents: earnings/guidance, products, supply chain, regulation/lawsuit, analysts/price targets, partnerships/M&A, and sentiment/buzz.
- Incorporate salient nouns/hashtags/cashtags from tweets if any.
Input tweets:
{tweets}
Return 6-8 search queries, one per line, no numbering.
"""

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


PREDICT_REFLECT_INSTRUCTION = """Given a list of facts, estimate their overall impact on the price movement of {ticker} stock. You MUST follow this exact output format:

Line 1 → Price Movement: Positive | Negative
Line 2 → Explanation: <one short paragraph. No numbering. Do not prefix with (1) or (2).>

Rules:
- Output EXACTLY two lines.
- Do NOT include any numbering or parentheses like "(1)" or "(2)" anywhere.
- The first line must start with: "Price Movement: "
- The second line must start with: "Explanation: "

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Facts:
{summary}

Price Movement:"""


REFLECTION_HEADER = 'You have attempted to tackle the following task before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly tackling the given task.\n'


REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self-reflection.  
You will be given a previous reasoning trial in which you were asked to assess the price movement of {ticker} stock.  
The trial was incorrect because you predicted the wrong price movement.  

Your task:  
- Diagnose the possible reason for failure.  
- Provide reflection in one paragraph.  
- Do not output any plan or strategy.  

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