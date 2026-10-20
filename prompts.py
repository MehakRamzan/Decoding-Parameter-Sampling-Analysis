
PROMPTS = {
    "factual": """
Name the three astronauts who flew on Apollo 11 in 1969,
and identify which two walked on the Moon.
If uncertain, explicitly state your uncertainty.
Answer in no more than 80 words.
""",

    "creative": """
Write a short science fiction story of 100 words about
a robot experiencing human emotions for the first time.
Include an unexpected ending.
""",

    "reasoning": """
A shop sells notebooks for $4 each and pens for $2 each.
Sara buys 3 notebooks and 5 pens.
She pays $30.

Calculate the total cost and the change she receives.
Show your calculations.
""",

    "structured": """
Extract information from the following sentence:

"Ali is 24 years old and lives in Lahore."

Return only a valid JSON object with exactly these keys:
"name", "age", "city".

The age must be an integer.
Do not include explanations or Markdown formatting.
"""
}

