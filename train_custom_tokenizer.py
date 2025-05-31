"""
Training script for custom BPE tokenizers

This script allows you to train a BPE tokenizer on your own text data.
It supports various input formats and provides detailed analysis of the results.

Usage:
    python train_custom_tokenizer.py --input data.txt --vocab_size 50000 --output my_tokenizer.pkl
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
import time

from bpe_tokenizer import BPETokenizer


def load_text_file(filepath: str) -> str:
    """Load text from a file with automatic encoding detection."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Could not decode file {filepath} with any common encoding")


def load_training_data(input_path: str, max_files: int = None) -> List[str]:
    """
    Load training data from file(s).

    Args:
        input_path: Path to text file or directory containing text files
        max_files: Maximum number of files to load (None for all)

    Returns:
        List of text strings for training
    """
    path = Path(input_path)
    texts = []

    if path.is_file():
        print(f"Loading text from file: {input_path}")
        text = load_text_file(input_path)

        # Split large files into chunks to improve training
        chunk_size = 10000  # characters per chunk
        if len(text) > chunk_size:
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if chunk.strip():  # Only add non-empty chunks
                    texts.append(chunk)
        else:
            texts.append(text)

    elif path.is_dir():
        print(f"Loading text files from directory: {input_path}")

        # Find all text files
        text_files = []
        for ext in ['*.txt', '*.text', '*.md', '*.py', '*.js', '*.html', '*.json']:
            text_files.extend(path.glob(ext))
            text_files.extend(path.glob(f"**/{ext}"))  # recursive

        if max_files:
            text_files = text_files[:max_files]

        print(f"Found {len(text_files)} text files")

        for file_path in text_files:
            try:
                text = load_text_file(str(file_path))
                if text.strip():  # Only add non-empty files
                    texts.append(text)
                    if len(texts) % 100 == 0:
                        print(f"  Loaded {len(texts)} files...")
            except Exception as e:
                print(f"  Warning: Could not load {file_path}: {e}")

    else:
        raise ValueError(
            f"Input path {input_path} is neither a file nor directory")

    print(f"Loaded {len(texts)} text chunks for training")
    return texts


def analyze_tokenizer_performance(tokenizer: BPETokenizer, test_texts: List[str]) -> None:
    """Analyze tokenizer performance on test data."""

    print("\n=== Tokenizer Performance Analysis ===")
    print("-" * 50)

    total_chars = 0
    total_tokens = 0
    total_words = 0

    # Analyze sample of test texts
    sample_size = min(10, len(test_texts))
    sample_texts = test_texts[:sample_size]

    print(f"Analyzing {sample_size} sample texts...")

    for i, text in enumerate(sample_texts):
        if len(text.strip()) == 0:
            continue

        # Basic stats
        chars = len(text)
        words = len(text.split())
        tokens = len(tokenizer.encode(text))

        total_chars += chars
        total_tokens += tokens
        total_words += words

        if i < 3:  # Show details for first few texts
            print(f"\nSample {i+1}:")
            print(f"  Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            print(f"  Characters: {chars}")
            print(f"  Words: {words}")
            print(f"  Tokens: {tokens}")
            print(f"  Chars/token: {chars/tokens:.2f}")
            print(f"  Tokens/word: {tokens/words:.2f}")

    # Overall statistics
    if total_tokens > 0:
        print(f"\nOverall Statistics:")
        print(f"  Average chars per token: {total_chars/total_tokens:.2f}")
        print(f"  Average tokens per word: {total_tokens/total_words:.2f}")
        print(f"  Compression ratio: {total_words/total_tokens:.2f}x")


def show_vocabulary_analysis(tokenizer: BPETokenizer) -> None:
    """Show analysis of the learned vocabulary."""

    print("\n=== Vocabulary Analysis ===")
    print("-" * 50)

    vocab = tokenizer.get_vocab()

    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Number of BPE merges: {len(tokenizer.bpe_merges)}")

    # Categorize tokens
    single_chars = []
    subwords = []
    special_tokens = []

    for token in vocab.keys():
        if token.startswith('<|') and token.endswith('|>'):
            special_tokens.append(token)
        elif len(token) == 1:
            single_chars.append(token)
        else:
            subwords.append(token)

    print(f"\nToken categories:")
    print(f"  Single characters: {len(single_chars)}")
    print(f"  Subwords: {len(subwords)}")
    print(f"  Special tokens: {len(special_tokens)}")

    # Show some example subwords
    if subwords:
        print(f"\nExample subwords (first 20):")
        for i, token in enumerate(subwords[:20]):
            print(f"  '{token}' -> {vocab[token]}")

    # Show special tokens
    if special_tokens:
        print(f"\nSpecial tokens:")
        for token in special_tokens:
            print(f"  '{token}' -> {vocab[token]}")

    # Show some recent merges
    if tokenizer.bpe_merges:
        print(f"\nRecent BPE merges (last 10):")
        for i, (token1, token2) in enumerate(tokenizer.bpe_merges[-10:]):
            merge_num = len(tokenizer.bpe_merges) - 10 + i + 1
            print(
                f"  {merge_num:4d}: '{token1}' + '{token2}' -> '{token1 + token2}'")


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom BPE tokenizer")
    parser.add_argument("--input", "-i", required=True,
                        help="Input text file or directory containing text files")
    parser.add_argument("--vocab_size", "-v", type=int, default=50000,
                        help="Target vocabulary size (default: 50000)")
    parser.add_argument("--output", "-o", default="tokenizer.pkl",
                        help="Output file for trained tokenizer (default: tokenizer.pkl)")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to load from directory")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ratio of data to use for testing (default: 0.1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed training progress")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)

    if args.vocab_size < 256:
        print("Error: Vocabulary size must be at least 256")
        sys.exit(1)

    print("=== Custom BPE Tokenizer Training ===\n")
    print(f"Input: {args.input}")
    print(f"Target vocabulary size: {args.vocab_size}")
    print(f"Output: {args.output}")
    print()

    # Load training data
    try:
        print("Loading training data...")
        all_texts = load_training_data(args.input, args.max_files)

        if not all_texts:
            print("Error: No text data found")
            sys.exit(1)

        # Split into training and test sets
        test_size = max(1, int(len(all_texts) * args.test_ratio))
        test_texts = all_texts[:test_size]
        train_texts = all_texts[test_size:]

        # Ensure we have training data - if only one chunk, use it for both training and testing
        if len(train_texts) == 0:
            train_texts = all_texts

        print(f"Training texts: {len(train_texts)}")
        print(f"Test texts: {len(test_texts)}")

        # Calculate total size
        total_chars = sum(len(text) for text in all_texts)
        print(f"Total characters: {total_chars:,}")

    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)

    # Train tokenizer
    try:
        print(f"\nTraining BPE tokenizer...")
        start_time = time.time()

        tokenizer = BPETokenizer()
        tokenizer.train(train_texts, vocab_size=args.vocab_size,
                        verbose=args.verbose)

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    # Analyze results
    show_vocabulary_analysis(tokenizer)
    analyze_tokenizer_performance(tokenizer, test_texts)

    # Save tokenizer
    try:
        print(f"\nSaving tokenizer to {args.output}...")
        tokenizer.save(args.output)

        # Save vocabulary as JSON for inspection
        vocab_file = args.output.replace('.pkl', '_vocab.json')
        import json
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer.get_vocab(), f, indent=2, ensure_ascii=False)
        print(f"Vocabulary saved to {vocab_file}")

    except Exception as e:
        print(f"Error saving tokenizer: {e}")
        sys.exit(1)

    print(f"\n=== Training Complete ===")
    print(f"Tokenizer saved to: {args.output}")
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
