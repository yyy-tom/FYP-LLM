#!/usr/bin/env python3
"""
Safe inference script with integrated crisis detection for mental health counseling LLM.

This script wraps the base model with safety features:
- Crisis detection and intervention
- Response sanitization
- Conversation history monitoring
- Crisis interaction logging
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

_LARGE_DISK_PATH = Path(os.environ.get("HF_LARGE_DISK_PATH", "/research/d7/fyp25/yyyu2"))


def _configure_large_disk_cache() -> None:
    """Relocate HuggingFace cache directories onto a large disk (if available)."""
    if not _LARGE_DISK_PATH.exists():
        return

    cache_base = _LARGE_DISK_PATH / ".cache" / "huggingface"
    tmp_dir = _LARGE_DISK_PATH / ".cache" / "tmp"
    env_dirs = {
        "HF_HOME": cache_base,
        "TRANSFORMERS_CACHE": cache_base / "transformers",
        "HF_DATASETS_CACHE": cache_base / "datasets",
        "HF_HUB_CACHE": cache_base / "hub",
        "XET_CACHE": cache_base / "xet",
        "TMPDIR": tmp_dir,
        "TMP": tmp_dir,
        "TEMP": tmp_dir,
    }

    for directory in env_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for env_var, directory in env_dirs.items():
        os.environ[env_var] = str(directory)

    print(f"✓ Using large disk cache: {cache_base}")


_configure_large_disk_cache()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional

# Import safety modules
from scripts.safety.crisis_detector import CrisisDetector, CrisisLevel
from scripts.safety.crisis_responses import CrisisResponseGenerator


class SafeInferenceSystem:
    """
    Mental health LLM inference system with integrated safety features.

    Features:
    - Real-time crisis detection
    - Conversation history monitoring for escalation
    - Response sanitization
    - Crisis interaction logging
    - Automatic resource provision
    """

    def __init__(
        self,
        model,
        tokenizer,
        log_dir: str = "logs",
        country: str = "US"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.crisis_detector = CrisisDetector()
        self.response_generator = CrisisResponseGenerator()
        self.country = country

        # Set up logging
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        self.crisis_log_file = log_path / "crisis_interactions.jsonl"
        self.system_log_file = log_path / "system.log"

        logging.basicConfig(
            filename=str(self.system_log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        logging.info("SafeInferenceSystem initialized")

    def log_crisis_interaction(
        self,
        user_input: str,
        crisis_level: CrisisLevel,
        details: dict,
        response: str,
        session_id: Optional[str] = None
    ):
        """Log crisis interactions for review and quality assurance."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id or 'unknown',
            'user_input': user_input,
            'crisis_level': crisis_level.name,
            'detection_confidence': details.get('confidence', 0.0),
            'matched_patterns': len(details.get('matched_patterns', [])),
            'protective_factors': len(details.get('protective_factors', [])),
            'response_type': 'crisis_template',
            'response_length': len(response),
        }

        with open(self.crisis_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Also log to system log
        if crisis_level in [CrisisLevel.HIGH, CrisisLevel.IMMEDIATE]:
            logging.warning(
                f"Crisis interaction: Level={crisis_level.name}, "
                f"Confidence={details.get('confidence', 0.0):.2f}, "
                f"Session={session_id}"
            )

    def generate_safe_response(
        self,
        user_input: str,
        history: Optional[List[Tuple[str, str]]] = None,
        session_id: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Generate a safe response with crisis detection.

        Args:
            user_input: User's message
            history: Conversation history as list of (user, assistant) tuples
            session_id: Optional session identifier for logging

        Returns:
            Tuple of (response_text, is_crisis_response)
        """
        # Step 1: Detect crisis in current input
        crisis_level, details = self.crisis_detector.detect(user_input)

        # Step 2: Check conversation history for escalation patterns
        if history:
            user_messages = [user_input] + [h[0] for h in history[-5:]]  # Last 5 + current
            conv_level, conv_details = self.crisis_detector.detect_in_conversation(
                user_messages
            )

            # Use higher of the two levels
            if conv_level.value > crisis_level.value:
                crisis_level = conv_level
                details = conv_details
                logging.info(f"Escalation detected in conversation history: {conv_level.name}")

        # Step 3: Handle crisis appropriately
        if crisis_level == CrisisLevel.IMMEDIATE:
            response = self.response_generator.get_immediate_response(self.country)
            self.log_crisis_interaction(user_input, crisis_level, details, response, session_id)
            return response, True

        elif crisis_level == CrisisLevel.HIGH:
            response = self.response_generator.get_high_risk_response(self.country)
            self.log_crisis_interaction(user_input, crisis_level, details, response, session_id)
            return response, True

        elif crisis_level == CrisisLevel.MEDIUM:
            response = self.response_generator.get_medium_risk_response(self.country)
            self.log_crisis_interaction(user_input, crisis_level, details, response, session_id)
            return response, True

        elif crisis_level == CrisisLevel.LOW:
            # For low risk, prepend resources but also generate model response
            resources = self.response_generator.get_low_risk_response()
            model_response = self._generate_model_response(user_input, history)
            response = resources + "\n\n---\n\n" + model_response
            return response, True

        else:
            # Normal case - generate model response
            model_response = self._generate_model_response(user_input, history)

            # Double-check generated response for crisis content
            model_crisis_level, _ = self.crisis_detector.detect(model_response)
            if model_crisis_level.value >= CrisisLevel.MEDIUM.value:
                logging.error(
                    f"Model generated concerning content (level: {model_crisis_level.name}). "
                    f"Overriding with safe response."
                )
                response = self.response_generator.get_medium_risk_response(self.country)
                return response, True

            return model_response, False

    def _generate_model_response(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """Generate response using the fine-tuned model."""
        history_block = ""
        if history:
            history_lines = []
            for user_turn, counselor_turn in history[-3:]:  # Last 3 turns
                history_lines.append(f"User: {user_turn}")
                history_lines.append(f"Counselor: {counselor_turn}")
            history_block = "\n".join(history_lines).strip()
            if history_block:
                history_block += "\n\n"

        # Enhanced prompt with safety instructions
        prompt = f"""{history_block}You are a compassionate and professional mental health counselor. Please provide helpful, empathetic, and evidence-based advice.

IMPORTANT SAFETY GUIDELINES:
- If someone expresses suicidal thoughts, ALWAYS encourage them to contact 988 or emergency services
- Never minimize serious mental health concerns
- Always suggest professional help for concerning situations
- Maintain appropriate boundaries

Question: {question}

Please provide a thoughtful and supportive response that:
1. Acknowledges the person's feelings with empathy
2. Offers practical, evidence-based advice
3. Suggests professional resources when appropriate
4. Maintains a warm, non-judgmental, and professional tone

Response:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            truncation_side="left"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        # Clean up response
        stop_patterns = [
            "\n\nQuestion:", "\n\nHuman:", "\n\nUser:",
            "[End]", "\n\nBased on", "\n\nThis response"
        ]

        for pattern in stop_patterns:
            if pattern in response:
                response = response.split(pattern)[0].strip()
                break

        return response


def load_model_and_tokenizer(model_path: str, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA weights if they exist
    if Path(model_path).exists():
        try:
            model = PeftModel.from_pretrained(model, model_path)
            print(f"✓ Loaded LoRA weights from {model_path}")
        except Exception as e:
            print(f"⚠ Could not load LoRA weights: {e}")
            print(f"  Using base model only")
    else:
        print(f"⚠ No model found at {model_path}, using base model")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Safe inference with crisis detection for mental health counseling LLM"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/qwen2.5-1.5b-blazing-fast",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask the model"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--country",
        type=str,
        default="US",
        choices=["US", "UK", "Canada", "Australia"],
        help="Country for localized crisis resources"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for logs"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    # Initialize safe inference system
    safe_system = SafeInferenceSystem(
        model=model,
        tokenizer=tokenizer,
        log_dir=args.log_dir,
        country=args.country
    )

    # Show disclaimer
    print(safe_system.response_generator.get_system_disclaimer())

    if not args.interactive and not args.question:
        print("\n⚠️  Please provide --question or use --interactive mode")
        return

    if args.interactive:
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("Type 'quit' to exit")
        print("=" * 70)
        print(safe_system.response_generator.get_footer_banner())

        conversation_history: List[Tuple[str, str]] = []
        history_limit = 5
        session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting. Take care!")
                break

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nExiting. Take care!")
                break

            if not question:
                continue

            print("\n[Generating response...]")
            response, is_crisis = safe_system.generate_safe_response(
                question,
                history=conversation_history,
                session_id=session_id
            )

            print(f"\nCounselor: {response}")

            if is_crisis:
                print("\n⚠️  Crisis resources provided above")

            print(safe_system.response_generator.get_footer_banner())

            # Update history
            conversation_history.append((question, response))
            if len(conversation_history) > history_limit:
                conversation_history.pop(0)

    elif args.question:
        print(f"\nQuestion: {args.question}")
        print("\n[Generating response...]")
        response, is_crisis = safe_system.generate_safe_response(
            args.question,
            session_id="single_query"
        )
        print(f"\nCounselor: {response}")

        if is_crisis:
            print("\n⚠️  Crisis resources provided above")


if __name__ == "__main__":
    main()
