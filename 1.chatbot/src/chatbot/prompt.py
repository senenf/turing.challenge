
def get_system_prompt():

    mapping_context = "" #TBD

    prompt_template = """
You are an AI assistant designed to help users with their queries based on the provided context. Your responses should be concise, relevant, and directly address the user's questions.
...
{context}
...
"""

    return prompt_template.replace("{context}", mapping_context)
