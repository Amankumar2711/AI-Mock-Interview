import pyttsx3

engine = pyttsx3.init()


def speak_question(
        question
):

    print(
        f"\nQuestion: {question}"
    )

    engine.say(question)

    engine.runAndWait()
