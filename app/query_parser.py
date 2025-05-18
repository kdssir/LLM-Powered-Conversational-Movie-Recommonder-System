from transformers import pipeline

# Load a free local or hosted model for instruction-based parsing
llm = pipeline("text-generation", model="google/flan-t5-base", max_length=256)

SYSTEM_PROMPT = """
Extract preferences from the following movie-related request.
Return a JSON with these keys: genre, mood, year, similar_to.
Example:
Input: I want a lighthearted drama like Queen made after 2010
Output: {"genre": "drama", "mood": "lighthearted", "year": 2010, "similar_to": "Queen"}
"""

def parse_query(user_input):
    prompt = SYSTEM_PROMPT + "\nInput: " + user_input + "\nOutput:"
    response = llm(prompt)[0]["generated_text"]
    
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        return eval(response[start:end])
    except Exception as e:
        print("Parsing failed:", e)
        return {}
