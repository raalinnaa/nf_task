import random
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    with open(file_path, 'r') as f:
        names = f.read().splitlines()
    return names

def create_bigram_probs(names):
    bigram_counts = {}
    for name in names:
        name = '^' + name.lower() + '$'
        for i in range(len(name) - 1):
            bigram = name[i:i+2]
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    total_count = sum(bigram_counts.values())
    bigram_probs = {k: v / total_count for k, v in bigram_counts.items()}
    return bigram_probs

def generate_name(bigram_probs):
    name = '^'
    while name[-1] != '$':
        bigrams = list(bigram_probs.keys())
        probs = list(bigram_probs.values())
        bigram_dist = torch.tensor(probs)
        chosen_bigram = bigrams[bigram_dist.multinomial(1)]
        if chosen_bigram[0] == name[-1]:
            name += chosen_bigram[1]
    return name[1:-1].capitalize()

def visualize_bigram_probs(bigram_probs):
    sorted_bigrams = sorted(bigram_probs.items(), key=lambda x: x[1], reverse=True)
    data = pd.DataFrame(sorted_bigrams, columns=["Bigram", "Probability"])
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Bigram", y="Probability", data=data.head(30))
    plt.title("Top 30 Bigram Probabilities")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("bigram_probabilities.png")
    plt.show()

def main():
    names = read_data('names.txt')
    bigram_probs = create_bigram_probs(names)
    new_name = generate_name(bigram_probs)
    print("Generated name:", new_name)
    visualize_bigram_probs(bigram_probs)

if __name__ == "__main__":
    main()
