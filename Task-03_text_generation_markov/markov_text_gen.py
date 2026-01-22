import argparse
import random
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


# ----------------------------
# 1) Tokenization
# ----------------------------
def tokenize(text: str) -> List[str]:
    """
    Converts raw text into tokens (words + punctuation).
    Example:
      "Hello world!" -> ["Hello", "world", "!"]
    """
    text = text.replace("\n", " ")
    return re.findall(r"[A-Za-z0-9']+|[.,!?;:]", text)


def detokenize(tokens: List[str]) -> str:
    """
    Converts tokens back into readable text.
    Punctuation is attached to the previous word without extra space.
    """
    out = []
    for tok in tokens:
        if tok in {".", ",", "!", "?", ";", ":"}:
            if out:
                out[-1] = out[-1] + tok
            else:
                out.append(tok)
        else:
            out.append(tok)
    return " ".join(out)


# ----------------------------
# 2) Build Markov Chain (order-n)
# ----------------------------
def build_markov_chain(tokens: List[str], order: int) -> Dict[Tuple[str, ...], Counter]:
    """
    Builds an order-n (n-gram) Markov chain.

    state  = tuple(previous n tokens)
    value  = Counter(next_token -> frequency)

    Example for order=2:
      state ("I", "am") -> {"happy": 3, "here": 1}
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    chain: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

    # Not enough tokens to build the model
    if len(tokens) <= order:
        return chain

    for i in range(len(tokens) - order):
        state = tuple(tokens[i : i + order])
        nxt = tokens[i + order]
        chain[state][nxt] += 1

    return chain


# ----------------------------
# 3) Sampling (weighted probability)
# ----------------------------
def weighted_choice(counter: Counter) -> str:
    """
    Chooses a next token based on frequency weights.
    If "the" appears 50 times and "a" appears 10 times,
    "the" is more likely to be chosen.
    """
    items = list(counter.items())
    words, weights = zip(*items)
    return random.choices(words, weights=weights, k=1)[0]


# ----------------------------
# 4) Generate text
# ----------------------------
def generate_text(
    chain: Dict[Tuple[str, ...], Counter],
    order: int,
    length: int,
    seed_text: str = "",
) -> str:
    """
    Generates 'length' tokens of text using the Markov chain.
    If seed_text is provided, generation tries to start from it.
    """
    if not chain:
        raise ValueError("Chain is empty. Provide more training text or use a lower order.")

    states = list(chain.keys())

    # Pick start state
    if seed_text.strip():
        seed_tokens = tokenize(seed_text)
        if len(seed_tokens) >= order:
            state = tuple(seed_tokens[-order:])
            if state not in chain:
                state = random.choice(states)  # fallback
        else:
            state = random.choice(states)
    else:
        state = random.choice(states)

    generated = list(state)

    # Generate tokens
    while len(generated) < length:
        options = chain.get(state)
        if not options:
            # If state has no outgoing transitions, restart from a random state
            state = random.choice(states)
            generated.extend(list(state))
            continue

        nxt = weighted_choice(options)
        generated.append(nxt)
        state = tuple(generated[-order:])

    return detokenize(generated[:length])


# ----------------------------
# 5) CLI entry point
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Task-03: Text Generation with Markov Chains")
    parser.add_argument("--input", type=str, default="input.txt", help="Training text file path")
    parser.add_argument("--order", type=int, default=2, help="Markov order (1, 2, 3...)")
    parser.add_argument("--length", type=int, default=120, help="Number of tokens to generate")
    parser.add_argument("--seed", type=str, default="", help="Seed text to start generation (optional)")
    parser.add_argument("--output", type=str, default="", help="Optional output file to save generated text")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Fix randomness (reproducible output)
    random.seed(args.random_seed)

    # Read training text
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenize(text)
    chain = build_markov_chain(tokens, args.order)

    generated = generate_text(chain, args.order, args.length, seed_text=args.seed)

    print("\n--- GENERATED TEXT ---\n")
    print(generated)
    print("\n----------------------\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(generated)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
