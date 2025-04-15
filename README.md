# SchizoNegSympAI

## Overview

**SchizoNegSympAI** is the official implementation of the research paper: "*Analyzing Generative AI and Machine Learning in Auto-Assessing Schizophrenia's Negative Symptoms*."

This study presents an automated system for the assessment of negative symptoms in schizophrenia based on the Clinical Assessment Interview for Negative Symptoms (CAINS) scale. The system evaluates two primary domains:

- **EXP Domain (Expression)**: Extracts and analyzes features from video and audio data using ensemble learning that combines ordinal and nominal classifiers.
- **MAP Domain (Motivation and Pleasure)**: Leverages large language models with zero-shot prompting techniques to analyze clinical interview transcripts.

## Repository Structure

```
/
├── assets/                    # Resource files
└── src/  
    ├── core/                  # Shared utilities and helper functions
    ├── ensemble/              # Ensemble learning components
    │   ├── feature_extractor/ # Multimodal feature extraction
    │   │   ├── spkr_diarization.py    # Speaker diarization and separation
    │   │   ├── holistic_tracking.py   # Facial and pose analysis
    │   │   └── encoder.py             # Feature encoding
    │   └── estimator/         # Classification models
    ├── llm/                   # LLM-based MAP assessment
    │   ├── transcriber.py     # Audio-to-text transcription
    │   └── estimator.py       # LLM-based assessment logic
    ├── config.yaml            # Configuration file
    └── .env.example           # Environment variable template
```

## Installation

1. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Copy the environment template:
    ```bash
    cp .env.example .env
    ```

3. Edit `.env` to set up your API keys.

## EXP Assessment Pipeline

This pipeline evaluates expression-related symptoms (facial expression, expressive gestures, vocal expression, and quantity of speech):

1. Prepare your audio/video files.

2. Extract features:
    ```bash
    python src/ensemble/feature_extractor/spkr_diarization.py /path/to/audio.wav
    python src/ensemble/feature_extractor/holistic_tracking.py /path/to/video.mp4
    python src/ensemble/feature_extractor/encoder.py
    ```

3. Train the ensemble classifier:
    ```bash
    python src/ensemble/estimator/train.py
    ```

4. Run inference:
    ```bash
    python src/ensemble/estimator/inference.py
    ```

## MAP Assessment Pipeline

This pipeline assesses motivation-related symptoms (social relationships, work/school activities, and recreational activities) from clinical interview transcripts:

1. Prepare your audio recordings.

2. Transcribe audio to text:
    ```bash
    python src/llm/transcriber.py path/to/audio.mp3
    ```

3. Perform MAP scoring with an LLM:
    ```bash
    python src/llm/estimator.py path/to/text_file.txt --model gpt-4
    ```

## External Dependencies

Key technologies and libraries used:

- [Claude 3 Haiku](https://www.anthropic.com/news/claude-3-haiku)
- [fairseq (Facebook AI)](https://github.com/facebookresearch/fairseq)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs/models)
- [Hartley Spectral Pooling](https://github.com/AlbertZhangHIT/Hartley-spectral-pooling)
- [MediaPipe (Google)](https://github.com/google-ai-edge/mediapipe)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [OpenAI GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3.5-turbo)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Ordinal Classifier](https://github.com/garyongguanjie/Ordinal-Classifier)
- [Random Forest](https://github.com/wangyuhsin/random-forest)

## License

Licensed under the [MIT License](LICENSE).
