# LecChurro: From Lecture to Learning

LecChurro is an AI-powered application designed to revolutionize the way students engage with lecture content. By leveraging cutting-edge audio and video processing techniques, speech recognition, and natural language processing (NLP), LecChurro helps students focus on learning and comprehension instead of transcription. It transforms lecture videos into detailed summaries, timestamps, quizzes, and flashcards, providing an interactive and accessible study experience.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Technologies and Libraries Used](#technologies-and-libraries-used)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Prompts](#prompts)
8. [Roadmap](#roadmap)
9. [Acknowledgments](#acknowledgments)
---

## Project Overview

**Goal**:  
LecChurro addresses the challenges students face when processing lecture content by using artificial intelligence to generate structured and meaningful study materials. The application ensures that students can focus on comprehension by extracting the most relevant information from lectures. LecChurro achieves this by analyzing both the audio and visual elements of lecture videos.

**Team Members**:
- **Ibraheem Refai**
- **Maggie Chen**
- **Matthew Rispler**

**Supervisor**:
- **David Gordo Gomez**

**Date Last Updated**: December 16, 2024

**Repository**: [https://github.com/ibbeh/LecChurro/](https://github.com/ibbeh/LecChurro/)

**Key Features**:
- Converts lecture audio into text using state-of-the-art speech recognition.
- Automatically generates:
  - **Summaries**: Detailed and concise lecture overviews.
  - **Timestamps**: Conceptual groupings with accurate timestamps.
  - **Quizzes**: Interactive assessments with multiple-choice and true/false questions.
  - **Flashcards**: Engaging tools for active learning and retention.

**Similar Solutions**:
- **Otter.ai**: Live transcription and note-taking.
- **NotebookLM**: Interaction with uploaded documents and summarization.
- **LectureSummarizer**: Video segmentation and caption generation.
  
LecChurro aims to go beyond these solutions by integrating advanced features like interactive quizzes, timestamps, and flashcards, making the learning experience more personalized and engaging.

---

## Features

1. **Speech-to-Text Transcription**  
   Converts audio recordings into text using Whisper, a robust automatic speech recognition (ASR) model.

2. **Text Summarization**  
   Summarizes transcription data into structured notes with headings, bolded terms, and bullet points for clarity.

3. **Conceptual Timestamps**  
   Groups related lecture segments into clusters with titles, summaries, and timestamps for easier navigation.

4. **Interactive Quizzes**  
   Automatically generates challenging quizzes with multiple-choice and true/false questions, complete with an answer key.

5. **Flashcards**  
   Extracts key lecture concepts and formats them into "Front" and "Back" flashcards for active recall practice.

6. **Data Management**  
   Stores audio, transcription, and output data in a structured directory for easy access and retrieval.

---

## Directory Structure
```
LecChurro/    
├── config/                  # Configuration files    
│   └── .gitkeep  
├── data/                    # Directory for processed data  
│   ├── audio/               # Audio files extracted from video  
│   ├── text/                # Transcription text  
│   ├── text_timestamps/     # Timestamped notes  
│   └── video/               # Uploaded lecture videos  
├── docs/                    # Documentation files  
│   └── README.md  
├── images/                  # Project images  
│   └── LecChurroLogo.jpg  
├── lib/                     # External libraries  
│   └── requirements.txt  
├── src/                     # Source code  
│   ├── app.py               # Main application entry point  
│   └── core/                # Core functionalities  
│       ├── prompts/         # GPT prompts for various features  
│       │   ├── flashcards_prompt.txt  
│       │   ├── group_concepts_prompt.txt  
│       │   ├── quiz_generation_json.txt  
│       │   └── summarization_prompt.txt  
│       ├── flashcards.py    # Flashcards generation  
│       ├── quizzes.py       # Quizzes generation and grading  
│       ├── summaries.py     # Summarization logic  
│       └── timestamps.py    # Timestamp generation  
├── tests/                   # Test scripts  
│   └── test_whisper.py      # Whisper model testing  
├── .gitignore               # Git ignored files  
├── requirements.txt         # Python dependencies  
└── README.md                # Project documentation
```

---

## Technologies and Libraries Used

### Core Frameworks and Libraries
- **[Gradio](https://gradio.app/)**: For creating the user interface.
- **[FFmpeg](https://ffmpeg.org/)**: For audio extraction from videos.
- **[OpenAI Whisper](https://github.com/openai/whisper)**: For speech-to-text transcription.
- **[OpenAI GPT](https://platform.openai.com/)**: For text summarization, quiz, and flashcard generation.
- **[Python-dotenv](https://pypi.org/project/python-dotenv/)**: For managing environment variables.

### Additional Dependencies
- **Torch**: Deep learning framework.
- **Pandas**: Data manipulation and analysis.
- **Faster-Whisper**: Optimized version of Whisper for faster processing.

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg installed on your system ([Installation Guide](https://ffmpeg.org/download.html))
- OpenAI API Key ([Sign up for OpenAI](https://platform.openai.com/signup/))

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ibbeh/LecChurro.git
   cd LecChurro
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Setup your .env file
   ```
   touch .env
   ```
   Add your OpenAI API key to the .env file:
   ```
   OPENAI_API_KEY=<your_openai_api_key>\
   ```
5. Ensure the integrity of the directory structure under data/
   ```bash
   mkdir -p data/audio data/text data/text_timestamps data/video
6. Launch the application
   ```bash
   python src/app.py
7. Open the Gradio interface through the link provided in the command line output:
   ```
   Launching Gradio interface...
   * Running on local URL:  http://127.0.0.1:7860
   ```
   
---

## Usage

1. Upload a lecture video through the Gradio interface.
2. Click "Transcribe Now" to process the video.
3. Wait for the transcription and generation to process
4. Navigate through the tabs for:
   * Summary/Notes: View the summarized lecture content.
   * Timestamps: Organized conceptual clusters with timestamps.
   * Quizzes: Interactive quizzes for self-assessment.
   * Flashcards: Digital flashcards for active recall.
4. Interact with Features:
   * Select quiz answers and submit to receive feedback.
   * Use flashcards to practice, recall, and reinforce learning. 
6. Download outputs as needed and repeat for other lectures

---

## Prompts

The prompts used for generating summaries, quizzes, flashcards, and timestamps are meticulously designed to guide the GPT model for optimal results. 
These prompts are located in the ```src/core/prompts/``` directory:  

* flashcards_prompt.txt: For generating flashcards.  
* group_concepts_prompt.txt: For organizing lecture segments into conceptual groups.  
* quiz_generation_json.txt: For creating quizzes in JSON format.  
* summarization_prompt.txt: For summarizing lecture transcriptions.  

---

## Roadmap

1. Phase 0: Research

   * Analyze existing solutions (Otter.ai, NotebookLM, LectureSummarizer).
   * Identify their limitations and areas for improvement.

2. Phase 1: Project Setup

   * Create GitHub Repository.
   * Collect and compile example lectures.
   * Organize files in a structured folder format with metadata annotations.

3. Phase 2: Transcribe Data

   * Extract audio from video files using FFmpeg.
   * Transcribe sample lectures into plain text using Whisper.
   * Store transcriptions in JSON format with a consistent naming convention.

4. Phase 3: Evaluation and Iteration

   * Evaluate model performance.
   * Iterate based on evaluation results.

5. Phase 4: Mock Feature Development

   * Speech-to-text transcription.
   * Summarization and note generation.
   * Timestamped notes and concept grouping.
   * Quiz and flashcard generation.

6. Phase 5: Deployment and User Interface

   * Build a simple user interface with Gradio.
   * Deploy the application for user access.

7. Phase 6: Feature Development and Deployment in Iterations

   * Summarization: Enhance summarization capabilities with prompt engineering.
   * Quiz Generation: Improve quiz generation through refined prompts.
   * Video Analysis: Use OpenCV to extract key visual information.
   * Timestamped Notes: Link transcribed notes to specific audio timestamps.

8. Phase 7: Documentation and Finalization

   * Complete comprehensive documentation.
   * Gather feedback and iterate for improvements.
  
---

## Acknowledgments

Special thanks to David Gordo Gomez for overseeing this project and helping us tremendously.

We acknowledge the use of various open-source libraries and tools that made this project possible.


