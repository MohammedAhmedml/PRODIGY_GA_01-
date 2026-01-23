# Task-03: Text Generation with Markov Chains

## Objective
Implement a simple text generation algorithm using Markov chains that predicts the next word based on previous word(s).

## Approach
- Tokenize the training text into words and punctuation.
- Build an order-n Markov chain:
  - state = previous n tokens
  - transitions = frequency of next token
- Generate text by sampling next tokens using weighted probabilities.

## How to Run
```bash
python markov_text_gen.py --input input.txt --order 2 --length 120 --output sample_output.txt
