# src/core/quizzes.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file to access API keys or other configurations
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Maximum number of quiz questions to generate
MAX_QUESTIONS = 50  # Adjustable


def generate_quiz(transcription_text):
    """
    Generates a quiz based on the provided lecture transcription using OpenAI's API.
    Args:
        transcription_text (str): The lecture transcription text to generate quiz questions from.
    Returns:
        str: Raw quiz text as a JSON-formatted string containing questions and answers.
    """
    # Get the directory of this script and construct the relative path for the quiz prompt file
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
    prompt_file_path = os.path.join(current_dir, 'prompts/quiz_generation_json.txt')  # Combine directory with the file name

    # Read the quiz generation prompt template from the file
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()

    # Replace the placeholder in the prompt template with the actual transcription text
    prompt = prompt_template.replace("TRANSCRIPTION_HERE", transcription_text)

    # Send the formatted prompt to OpenAI's API (gpt4o) to generate the quiz
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert teacher skilled in producing detailed and correct student assessments."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=14000, # Maximum tokens for response
        temperature=0.7, # Controls creativity in response
    )
    # Extract and return the generated quiz text
    quiz = response.choices[0].message.content.strip()
    return quiz

def grade_quizzes(*args):
    """
    Grades the quiz answers provided by a user.
    Args:
        *args: A variable-length argument list containing:
            - user_answers (list): List of user's answers to quiz questions.
            - quizzes (list): List of quiz questions and their correct answers.
    Returns:
        str: A summary of the grading results, indicating correct and incorrect answers.
    """
    # Separate user answers and quizzes from the arguments
    *user_answers, quizzes = args
    # Handle the case where no quizzes are provided
    if not quizzes or not isinstance(quizzes, list) or len(quizzes) == 0:
        return "No quizzes to grade."

    result = [] # Initialize a list to store grading results
    # Iterate through the quiz questions and grade each one
    for idx, q in enumerate(quizzes):
        if idx >= MAX_QUESTIONS:  # Stop grading if maximum questions are reached
            break
        user_ans = user_answers[idx] # User's answer for the current question
        correct = q["answer"] # Correct answer for the current question
        # Compare user's answer with the correct answer
        if user_ans == correct:
            result.append(f"**Question {idx+1}**: Correct! ✅")
        else:
            result.append(f"**Question {idx+1}**: Incorrect ❌ (Correct answer: {correct})")
            
    # Return the grading results as a formatted string
    return "\n".join(result)

