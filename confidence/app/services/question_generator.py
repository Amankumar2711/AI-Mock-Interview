QUESTIONS = [

    "Tell me about yourself",

    "What are your strengths?",

    # "Describe a challenging project.",

    # "Why should we hire you?",

    # "Where do you see yourself in 5 years?"
]


def generate_question(index):

    if index >= len(QUESTIONS):
        return None

    return QUESTIONS[index]
