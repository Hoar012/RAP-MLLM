import json
import re, string
import argparse

def normalize_answer(text: str) -> str:
    """Normalize a given text by removing articles, punctuation, and white spaces, and converting to lowercase."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lowercase(text))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, help='The path to the output file', required=True)
    args = parser.parse_args()
    with open(args.output_path, 'r') as f:
        outputs = json.load(f)
    scores = []
    for output in outputs:
        pred = output["answer"]
        correct = output["correct_answer"]
        scores.append(normalize_answer(pred) == normalize_answer(correct))
    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"Accuracy: {accuracy:.4f}")