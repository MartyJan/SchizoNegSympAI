"""
Run LLM evaluation on transcripts using models like GPT, Claude, Gemini, or Ollama.

Usage:
    # Process a single transcript file with specified model
    python estimator.py --input /path/to/transcript.txt --model gpt-4

    # Process a single transcript with custom output directory
    python estimator.py --input /path/to/transcript.txt --model claude-3-haiku-20240307 --output-dir /path/to/results

    # Process all transcripts in the default directory (if --input not specified)
    python estimator.py --model gemini-pro

    # Process all transcripts in a custom directory with custom output directory
    python estimator.py --input /path/to/transcripts/directory --model gpt-3.5-turbo --output-dir /path/to/results
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import tiktoken
from dotenv import load_dotenv
from google.generativeai.generative_models import safety_types
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import read_config
from core.log import get_logger
from llm.constants import ITEM_ABBR_TO_NAME, ITEM_TYPES_TO_ABBR_LIST

# Load environment variables
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)


config = read_config()
logger = get_logger()


def get_transcript(file_path: Path) -> str:
    """
    Reads and returns text content from a file.

    Args:
        file_path: Path to the transcript file

    Returns:
        str: The contents of the transcript file
    """
    with file_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def get_prompt(input_path: Path, item_type: str) -> List[BaseMessage]:
    """
    Generates prompt messages using the transcript and a prompt template.

    Args:
        input_path: Path to the transcript file
        item_type: Type of evaluation items to use

    Returns:
        List[BaseMessage]: Formatted messages for the LLM prompt
    """
    eval_rule_path = (
        Path(config["llm"]["path"]["prompt_template_dir"]) / f"{item_type}.txt"
    )
    instruct_path = Path(config["llm"]["path"]["prompt_template_dir"]) / "template.txt"

    with eval_rule_path.open("r", encoding="utf-8") as f:
        eval_rule = f.read().strip()

    with instruct_path.open("r", encoding="utf-8") as f:
        instruct = f.read().strip()

    items = [ITEM_ABBR_TO_NAME[item] for item in ITEM_TYPES_TO_ABBR_LIST[item_type]]
    conversation = get_transcript(input_path)

    chat_template = ChatPromptTemplate.from_messages([("human", instruct)])
    return chat_template.format_messages(
        item_count=str(len(items)),
        items="、".join(items),
        max_score=config["llm"]["estimator"]["max_output_score"],
        eval_rule=eval_rule,
        conversation=conversation,
    )


def get_execution_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a summary of the model output and prompt parameters.

    Args:
        context: Dictionary containing model execution context including output, model name,
                temperature, token count, and messages

    Returns:
        Dict[str, Any]: Formatted summary of the execution context
    """
    output = context["output"]
    output_dict = {"content": output} if isinstance(output, str) else output.dict()

    return {
        "model": context["model"],
        "temperature": context["temperature"],
        "prompt_token_count": context["token_count"],
        "prompt": [msg.dict() for msg in context["messages"]],
        "output": output_dict,
    }


def run_single(input_path: Path, output_dir: Path, model_id: str) -> None:
    """
    Runs LLM evaluation for a single input file.

    Args:
        input_path: Path to the transcript file
        output_dir: Directory to save the evaluation results
        model_id: Identifier of the LLM to use

    Returns:
        None
    """
    # now = datetime.now().astimezone()
    filename = input_path.stem.split("_")[0]

    prompt_messages = get_prompt(input_path, "C1-9")

    temperature = config["llm"]["estimator"]["temperature"]
    max_output_tokens = config["llm"]["estimator"]["max_output_tokens"]

    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(prompt_messages[0].content))

    max_input_tokens = config["llm"]["estimator"]["max_input_tokens"]
    if token_count > max_input_tokens:
        raise RuntimeError(
            f"[{filename}] Token count {token_count} exceeds limit ({max_input_tokens}), skipping."
        )

    if model_id.startswith("gpt"):
        chat = ChatOpenAI(
            model_name=model_id, temperature=temperature, max_tokens=max_output_tokens
        )
    elif model_id.startswith("gemini"):
        chat = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            max_tokens=max_output_tokens,
            safety_settings={
                safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: safety_types.HarmBlockThreshold.BLOCK_NONE,
                safety_types.HarmCategory.HARM_CATEGORY_HARASSMENT: safety_types.HarmBlockThreshold.BLOCK_NONE,
                safety_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: safety_types.HarmBlockThreshold.BLOCK_NONE,
                safety_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
        )
    elif model_id.startswith("claude"):
        chat = ChatAnthropic(
            model_name=model_id, temperature=temperature, max_tokens=max_output_tokens
        )
    else:
        chat = ChatOllama(
            model=model_id,
            temperature=temperature,
            verbose=True,
            max_tokens=max_output_tokens,
        )

    output = chat.invoke(prompt_messages)
    summary = get_execution_summary(
        {
            "model": model_id,
            "temperature": temperature,
            "messages": prompt_messages,
            "token_count": token_count,
            "output": output,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)
    logger.info(f"[{filename}] Output saved to {output_path}")


def run_folder(input_path: Path, output_dir: Path, model_id: str) -> None:
    """
    Runs LLM evaluation for all `.txt` files in a folder.

    Args:
        input_path: Directory containing transcript files
        output_dir: Directory to save the evaluation results
        model_id: Identifier of the LLM to use

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(input_path.iterdir()):
        if file_path.is_file() and file_path.suffix == ".txt":
            logger.info(f"Processing: {file_path.name}...")
            try:
                run_single(file_path, output_dir, model_id)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    now = datetime.now().astimezone()
    default_output_dir = (
        Path(config["llm"]["path"]["result_dir"])
        / f"{config['llm']['estimator']['experiment_name']}_{now.strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    parser = argparse.ArgumentParser(
        description="Run LLM evaluation on audio transcripts."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=config["llm"]["path"]["transcript_dir"],
        help="File or folder of input text",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model ID (e.g., claude-3-haiku-20240307, gemini-pro, gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help="Directory to save results",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to processes arguments and runs the appropriate evaluation.

    Returns:
        None
    """
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if input_path.is_file():
        run_single(input_path, output_dir, args.model)
    elif input_path.is_dir():
        run_folder(input_path, output_dir, args.model)
    else:
        raise ValueError(
            f"Input path '{input_path}' is neither a file nor a directory."
        )


if __name__ == "__main__":
    main()
