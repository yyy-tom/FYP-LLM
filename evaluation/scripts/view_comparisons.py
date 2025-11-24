
"""
Simple script to view model comparison results from evaluation JSON.
"""

import json
import sys
import textwrap
import re
from pathlib import Path


def format_text(text: str, width: int = 80, indent: str = "   ") -> str:
    """Format text with proper wrapping, preserving word boundaries."""
    if not text:
        return ""
    
    # Use textwrap with proper settings to avoid breaking words
    wrapped_lines = textwrap.wrap(
        text,
        width=width - len(indent),
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
        expand_tabs=True
    )
    
    return '\n'.join(wrapped_lines) if wrapped_lines else indent + text


def format_prompt(input_text: str) -> str:
    """Format prompt text, preserving structure like Context: and Question:"""
    if not input_text:
        return "   (Empty - check dataset format)"
    
    # Normalize whitespace first - replace multiple spaces with single space
    text = re.sub(r' +', ' ', input_text)
    
    # Ensure markers are on their own lines for better readability
    # Match any whitespace (including newlines) before the marker
    text = re.sub(r'(\s+)(Context:)', r'\n\n\2', text)
    text = re.sub(r'(\s+)(Question:)', r'\n\n\2', text)
    text = re.sub(r'(\s+)(Please provide)', r'\n\n\2', text)
    text = re.sub(r'(\s+)(Response:)', r'\n\n\2', text)
    
    # Clean up any triple+ newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Split by sections (double newlines)
    sections = [s.strip() for s in text.split('\n\n') if s.strip()]
    
    result_lines = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if section starts with a marker
        marker_match = re.match(r'^(Context:|Question:|Please provide|Response:)', section)
        if marker_match:
            # Split marker from content
            marker = marker_match.group(1)
            content = section[len(marker):].strip()
            
            # Format the marker line
            if marker.endswith(':'):
                # For markers with colons, put marker and first part of content on same line if short
                if content and len(marker + ' ' + content.split('\n')[0]) <= 77:
                    first_line = content.split('\n')[0]
                    result_lines.append(f"   {marker} {first_line}")
                    # If there's more content, wrap it
                    remaining = '\n'.join(content.split('\n')[1:]).strip()
                    if remaining:
                        wrapped = textwrap.wrap(
                            remaining,
                            width=77,
                            initial_indent="   ",
                            subsequent_indent="   ",
                            break_long_words=False,
                            break_on_hyphens=False
                        )
                        result_lines.extend(wrapped)
                else:
                    result_lines.append(f"   {marker}")
                    if content:
                        wrapped = textwrap.wrap(
                            content,
                            width=77,
                            initial_indent="   ",
                            subsequent_indent="   ",
                            break_long_words=False,
                            break_on_hyphens=False
                        )
                        result_lines.extend(wrapped)
            else:
                # For "Please provide", keep it on its own line
                result_lines.append(f"   {marker}")
                if content:
                    wrapped = textwrap.wrap(
                        content,
                        width=77,
                        initial_indent="   ",
                        subsequent_indent="   ",
                        break_long_words=False,
                        break_on_hyphens=False
                    )
                    result_lines.extend(wrapped)
        else:
            # Regular text section, wrap normally
            wrapped = textwrap.wrap(
                section,
                width=77,
                initial_indent="   ",
                subsequent_indent="   ",
                break_long_words=False,
                break_on_hyphens=False
            )
            result_lines.extend(wrapped)
    
    return '\n'.join(result_lines) if result_lines else "   (Empty)"


def view_comparisons(json_path: str, max_examples: int = 10):
    """View comparison results in a readable format."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    comparisons = data.get('comparisons', [])
    if not comparisons:
        print("No comparisons found in the results file.")
        print("Make sure you ran evaluation with --compare_with_base flag.")
        return
    
    print("=" * 80)
    print(f"Model Comparison Results")
    print("=" * 80)
    print(f"Base Model: {data.get('base_model', 'Unknown')}")
    print(f"Fine-tuned Model: {data.get('model_path', 'Unknown')}")
    print(f"Test Dataset: {data.get('test_dataset_path', 'Unknown')}")
    
    # Show evaluation mode
    eval_mode = data.get('evaluation_mode', 'single-turn')
    print(f"Evaluation Mode: {eval_mode.upper()}")
    
    # Show conversational mode info if available
    conv_settings = data.get('conversational_settings')
    if conv_settings:
        print(f"\n💬 Conversational Evaluation Settings:")
        print(f"   Enabled: {conv_settings.get('enabled', False)}")
        if conv_settings.get('enabled'):
            print(f"   Min Turns: {conv_settings.get('min_turns', 'N/A')}")
            print(f"   Max Turns: {conv_settings.get('max_turns', 'N/A')}")
            print(f"   Conversations Found: {conv_settings.get('conversations_found', 'N/A')}")
    
    # Check if conversations data exists
    conversations = data.get('conversations', [])
    if conversations:
        print(f"\n💬 Multi-turn Conversations: {len(conversations)} conversations evaluated")
        total_turns = sum(c.get('num_turns', 0) for c in conversations)
        print(f"   Total turns across all conversations: {total_turns}")
        print(f"   ⚠️  NOTE: Comparisons shown below are from single-turn mode.")
        print(f"   To view multi-turn conversations, check the 'conversations' field in the JSON.")
    else:
        if eval_mode == 'conversational':
            print(f"\n⚠️  WARNING: Evaluation mode is 'conversational' but no conversations found in results.")
            print(f"   This suggests conversational mode was enabled but no multi-turn conversations were detected.")
        else:
            print(f"\n📝 Single-turn evaluation mode (no conversation history)")
    
    print(f"\nTotal Comparisons: {len(comparisons)}")
    print("=" * 80)
    print()
    
    for i, comp in enumerate(comparisons[:max_examples]):
        print(f"\n{'='*80}")
        print(f"Example {comp.get('sample_id', i)}")
        print(f"{'='*80}")
        
        input_text = comp.get('input', '')
        
        # Extract the question from the input if it contains a prompt template
        question = ""
        if input_text:
            # Try to extract the question part
            if "Question:" in input_text:
                # Extract everything after "Question:"
                question_start = input_text.find("Question:")
                if question_start != -1:
                    question = input_text[question_start + len("Question:"):].strip()
                    # Remove any trailing prompt instructions
                    if "Please provide" in question or "Response:" in question:
                        for marker in ["Please provide", "Response:", "\n\nResponse:"]:
                            if marker in question:
                                question = question.split(marker)[0].strip()
                    # If question is still very long, it might include the full prompt
                    # In that case, just show the input as-is
                    if len(question) > 500:
                        question = ""
        
        # Show dataset source if available
        dataset_source = comp.get('dataset_source', '')
        if dataset_source:
            print(f"\n📂 Dataset Source: {dataset_source}")
        
        # Show if this is from a conversation (check if there's turn info)
        turn_number = comp.get('turn_number', None)
        conversation_id = comp.get('conversation_id', None)
        eval_mode_comp = comp.get('evaluation_mode', 'single-turn')
        
        if turn_number is not None or conversation_id:
            print(f"\n💬 Conversation Info:")
            if conversation_id:
                print(f"   Conversation ID: {conversation_id}")
            if turn_number is not None:
                print(f"   Turn Number: {turn_number} (this is turn {turn_number + 1} in the conversation)")
            print(f"   Mode: {eval_mode_comp.upper()}")
        elif eval_mode_comp == 'conversational':
            print(f"\n💬 Mode: CONVERSATIONAL (but no turn info - may be from conversation summary)")
        
        print(f"\n📝 Full Input/Prompt:")
        print(format_prompt(input_text))
        
        # Show extracted question separately if found
        if question and question != input_text and len(question) < len(input_text):
            print(f"\n❓ Question (Extracted):")
            print(format_text(question, width=80, indent="   "))
        
        print(f"\n📋 Reference (Expected Response):")
        reference = comp.get('reference', '')
        if reference:
            print(format_text(reference, width=80, indent="   "))
        else:
            print("   (Empty)")
        
        print(f"\n🤖 Base Model Response:")
        base_resp = comp.get('base_response', '')
        if base_resp:
            print(format_text(base_resp, width=80, indent="   "))
        else:
            print("   (Empty)")
        
        print(f"\n✨ Fine-tuned Model Response:")
        ft_resp = comp.get('finetuned_response', '')
        if ft_resp:
            print(format_text(ft_resp, width=80, indent="   "))
        else:
            print("   (Empty)")
        
        # Metrics
        print(f"\n📊 Metrics:")
        base_rouge = comp.get('base_rouge', {})
        ft_rouge = comp.get('finetuned_rouge', {})
        print(f"   Base ROUGE-1: {base_rouge.get('rouge1', 0):.4f} | Fine-tuned: {ft_rouge.get('rouge1', 0):.4f}")
        print(f"   Base BLEU: {comp.get('base_bleu', 0):.4f} | Fine-tuned: {comp.get('finetuned_bleu', 0):.4f}")
        
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_comparisons.py <results_json> [max_examples]")
        print("\nExample:")
        print("  python view_comparisons.py evaluation/results/comparison_results.json 5")
        sys.exit(1)
    
    json_path = sys.argv[1]
    max_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    view_comparisons(json_path, max_examples)

