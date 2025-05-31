"""
Comparison script between our BPE tokenizer and other popular tokenizers.

This script helps understand how different tokenizers work and their trade-offs.
It compares our implementation with conceptually similar approaches.

Note: This script requires optional dependencies for full comparison.
Run: pip install transformers tiktoken (optional)
"""

from bpe_tokenizer import BPETokenizer
import time
from typing import Dict, List


def simple_word_tokenizer(text: str) -> List[str]:
    """Simple whitespace-based tokenizer for comparison."""
    return text.split()


def character_tokenizer(text: str) -> List[str]:
    """Character-level tokenizer for comparison."""
    return list(text)


def analyze_tokenizer_performance(name: str, tokenize_func, texts: List[str]) -> Dict:
    """Analyze performance of a tokenizer."""
    print(f"\n--- {name} ---")

    total_chars = 0
    total_tokens = 0
    total_words = 0
    encoding_times = []

    for text in texts:
        total_chars += len(text)
        total_words += len(text.split())

        start_time = time.time()
        try:
            if hasattr(tokenize_func, 'encode'):
                tokens = tokenize_func.encode(text)
            else:
                tokens = tokenize_func(text)
            encoding_time = time.time() - start_time
            encoding_times.append(encoding_time)

            total_tokens += len(tokens)

        except Exception as e:
            print(f"  Error with {name}: {e}")
            continue

    if total_tokens > 0:
        stats = {
            'avg_chars_per_token': total_chars / total_tokens,
            'avg_tokens_per_word': total_tokens / total_words,
            'compression_ratio': total_words / total_tokens,
            'avg_encoding_time': sum(encoding_times) / len(encoding_times) if encoding_times else 0,
            'vocab_size': getattr(tokenize_func, 'get_vocab_size', lambda: 'Unknown')()
        }

        print(f"  Avg chars per token: {stats['avg_chars_per_token']:.2f}")
        print(f"  Avg tokens per word: {stats['avg_tokens_per_word']:.2f}")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Avg encoding time: {stats['avg_encoding_time']*1000:.2f}ms")
        print(f"  Vocabulary size: {stats['vocab_size']}")

        return stats
    else:
        print(f"  No tokens generated")
        return {}


def demonstrate_tokenization_differences():
    """Show how different tokenizers handle the same text."""

    test_texts = [
        "Hello, world! This is a test of tokenization.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large datasets for training.",
        "Natural language processing is a subfield of artificial intelligence.",
        "Transformers revolutionized the field of NLP in 2017.",
    ]

    print("=== Tokenization Comparison ===")
    print("=" * 50)

    # Create and train our BPE tokenizer
    print("Training our BPE tokenizer...")
    our_tokenizer = BPETokenizer()

    # Use expanded training data
    training_data = test_texts * 20  # Repeat for better training
    training_data.extend([
        "programming languages like Python and JavaScript",
        "data science and machine learning algorithms",
        "artificial intelligence and deep learning",
        "natural language processing and computer vision",
        "tokenization and text preprocessing techniques",
    ] * 15)

    our_tokenizer.train(training_data, vocab_size=800, verbose=False)

    # Analyze different tokenizers
    tokenizers = [
        ("Character-level", character_tokenizer),
        ("Word-level", simple_word_tokenizer),
        ("Our BPE", our_tokenizer),
    ]

    # Try to import and add popular tokenizers
    try:
        from transformers import GPT2Tokenizer
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizers.append(("GPT-2 BPE", gpt2_tokenizer))
        print("✓ GPT-2 tokenizer loaded")
    except ImportError:
        print("⚠ transformers not installed - skipping GPT-2 comparison")
    except Exception as e:
        print(f"⚠ Could not load GPT-2 tokenizer: {e}")

    try:
        import tiktoken
        tiktoken_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokenizers.append(("OpenAI tiktoken", tiktoken_tokenizer))
        print("✓ tiktoken loaded")
    except ImportError:
        print("⚠ tiktoken not installed - skipping OpenAI comparison")
    except Exception as e:
        print(f"⚠ Could not load tiktoken: {e}")

    # Performance comparison
    print(f"\nAnalyzing performance on {len(test_texts)} test texts...")

    results = {}
    for name, tokenizer in tokenizers:
        results[name] = analyze_tokenizer_performance(
            name, tokenizer, test_texts)

    # Show detailed tokenization examples
    print(f"\n{'='*50}")
    print("DETAILED TOKENIZATION EXAMPLES")
    print(f"{'='*50}")

    example_text = "The transformer architecture revolutionized NLP."
    print(f"\nExample text: '{example_text}'")
    print("-" * 60)

    for name, tokenizer in tokenizers:
        try:
            if hasattr(tokenizer, 'encode'):
                if name == "Our BPE":
                    tokens = tokenizer.encode(example_text)
                    token_strs = [tokenizer.decoder[tid] for tid in tokens]
                elif name in ["GPT-2 BPE", "OpenAI tiktoken"]:
                    tokens = tokenizer.encode(example_text)
                    if hasattr(tokenizer, 'decode'):
                        # For tiktoken
                        token_strs = [tokenizer.decode(
                            [tid]) for tid in tokens]
                    else:
                        # For transformers tokenizers
                        token_strs = [tokenizer.decode(
                            [tid]) for tid in tokens]
                else:
                    tokens = tokenizer.encode(example_text)
                    token_strs = [str(t) for t in tokens]
            else:
                tokens = tokenizer(example_text)
                token_strs = tokens

            print(f"\n{name}:")
            print(f"  Tokens ({len(tokens)}): {token_strs}")

        except Exception as e:
            print(f"\n{name}: Error - {e}")

    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY COMPARISON")
    print(f"{'='*50}")

    if results:
        print(
            f"{'Tokenizer':<20} {'Chars/Token':<12} {'Compression':<12} {'Vocab Size':<12}")
        print("-" * 60)

        for name, stats in results.items():
            if stats:
                chars_per_token = f"{stats.get('avg_chars_per_token', 0):.2f}"
                compression = f"{stats.get('compression_ratio', 0):.2f}x"
                vocab_size = str(stats.get('vocab_size', 'Unknown'))

                print(
                    f"{name:<20} {chars_per_token:<12} {compression:<12} {vocab_size:<12}")


def educational_insights():
    """Provide educational insights about tokenization."""

    print(f"\n{'='*50}")
    print("EDUCATIONAL INSIGHTS")
    print(f"{'='*50}")

    insights = [
        "1. CHARACTER-LEVEL TOKENIZATION:",
        "   + Can handle any text (no OOV words)",
        "   + Very large vocabulary needed for sequences",
        "   - Sequences become very long",
        "   - Hard for models to learn word-level patterns",
        "",
        "2. WORD-LEVEL TOKENIZATION:",
        "   + Natural linguistic units",
        "   + Short sequences",
        "   - Cannot handle unknown words",
        "   - Very large vocabulary for real-world text",
        "",
        "3. BPE (SUBWORD) TOKENIZATION:",
        "   + Balances vocabulary size and sequence length",
        "   + Can handle unknown words through subword composition",
        "   + Learns frequent patterns from data",
        "   + Used by modern LLMs (GPT, BERT, etc.)",
        "",
        "4. WHY BPE WORKS FOR LLMS:",
        "   + Efficient representation of text",
        "   + Handles multiple languages",
        "   + Balances between characters and words",
        "   + Vocabulary size is controllable",
        "",
        "5. KEY DIFFERENCES IN OUR IMPLEMENTATION:",
        "   + Educational focus with detailed explanations",
        "   + Shows step-by-step BPE merging process",
        "   + Includes vocabulary analysis tools",
        "   + Simpler codebase for learning purposes",
    ]

    for insight in insights:
        print(insight)


def main():
    """Main function to run all comparisons."""

    print("BPE Tokenizer Comparison and Analysis")
    print("=" * 50)
    print("This script compares our BPE implementation with other tokenizers.")
    print("It helps understand the trade-offs and design decisions in tokenization.")
    print()

    # Run the comparison
    demonstrate_tokenization_differences()

    # Educational insights
    educational_insights()

    print(f"\n{'='*50}")
    print("NEXT STEPS FOR LEARNING")
    print(f"{'='*50}")

    next_steps = [
        "1. Experiment with different vocabulary sizes",
        "2. Train on domain-specific text (code, literature, etc.)",
        "3. Analyze learned subwords for your domain",
        "4. Compare compression ratios for different text types",
        "5. Implement other tokenization algorithms (WordPiece, SentencePiece)",
        "6. Study how tokenization affects model performance",
    ]

    for step in next_steps:
        print(step)

    print(f"\nCheck out the other scripts:")
    print("- example_usage.py: Basic usage and training")
    print("- train_custom_tokenizer.py: Train on your own data")
    print("- bpe_tokenizer.py: Core implementation")


if __name__ == "__main__":
    main()
