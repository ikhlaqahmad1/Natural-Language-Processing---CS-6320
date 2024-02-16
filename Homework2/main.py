# Homework 2 for CS 6320 NLP
# By Ikhlaq Ahmad ixa190000

# Dependencies
import math
import random
import spacy

# Loads a small greek corpus
nlp = spacy.load("el_core_news_sm")

# 2 Greek sentences
sentences = ["Τρώω τυρί και ελιές τα σαββατοκύριακα.",
             "Στην Ελλάδα, οι άνθρωποι απολαμβάνουν μια πλούσια ποικιλία τροφίμων που περιλαμβάνει φρέσκα θαλασσινά,"
             "λαχταριστά παραδοσιακά πιάτα όπως μουσακά και σουβλάκι, αρωματικά μπαχαρικά και βότανα, καθώς και μια "
             "εκπληκτική ποικιλία τυριών και ελιών, απολαμβάνοντας το φαγητό τους με καλό κρασί ή ούζο."]


# Opens the file and replaces newline char. Returns raw text
def open_file(file_path):
    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = text.replace("\n", "")
    return text


# Tokenizer
def greek_text_tokenizer(raw_text):
    doc = nlp(raw_text)
    tokens = [token.text for token in doc]
    return tokens


# Compute n-grams
def compute_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams


# Function to compute n-gram counts
def compute_ngram_counts(corpus, n):
    ngrams = compute_ngrams(corpus, n)
    ngram_counts = {}
    for ngram in ngrams:
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
        else:
            ngram_counts[ngram] = 1
    return ngram_counts


# Function to calculate smoothed-one probabilities
def calculate_smoothed_one_probabilities(tokens, vocabulary_size):
    smoothed_probabilities = {}
    total_tokens = len(tokens)
    unique_tokens = set(tokens)

    for token in unique_tokens:
        count_token = tokens.count(token)
        probability = (count_token + 1) / (total_tokens + vocabulary_size)
        smoothed_probabilities[token] = probability

    return smoothed_probabilities


# Function to calculate perplexity
def calculate_perplexity(sentence, probabilities):
    tokens = sentence.split()
    n = len(tokens)
    sum_log_probabilities = 0

    for token in tokens:
        if token in probabilities:
            sum_log_probabilities += math.log(probabilities[token])
        else:
            # If token not found, use log(1/V), where n is vocabulary size
            sum_log_probabilities += math.log(1 / n)

    perplexity = math.exp(-(1 / n) * sum_log_probabilities)
    return perplexity


# Function to sample a word according to unigram probability
def sample_word(probabilities):
    rand_num = random.random()
    cumulative_prob = 0.0
    for word, prob in probabilities:
        cumulative_prob += prob
        if rand_num < cumulative_prob:
            return word
    return None


# Function to sample the next word using conditional probability
def sample_next_word(previous_word, probabilities):
    if previous_word in probabilities:
        next_word_probs = probabilities[previous_word]
        return sample_word(next_word_probs)
    else:
        return sample_word(probabilities)


# Function to generate a sentence until a period is encountered
def generate_sentence(probabilities):
    sentence = ""
    previous_word = None

    while True:
        previous_word = sample_next_word(previous_word, probabilities)
        if previous_word is None or previous_word == ".":
            break
        sentence += previous_word + " "

    return sentence.strip()


def main():

    # Tokenization
    pre_text = open_file("food corpus.txt")
    greek_tokens = greek_text_tokenizer(pre_text)

    # N-grams calculations
    unigrams = compute_ngrams(greek_tokens, 1)
    bigrams = compute_ngrams(greek_tokens, 2)

    # Display
    print("\nBelow are the Uni-grams:")
    for u in unigrams:
        print(u)

    print("\nBelow are the Bi-grams:")
    for b in bigrams:
        print(b)

    # Sentence 1 and Sentence 2 tokenizer
    sentence1_tokens = greek_text_tokenizer(sentences[0])
    sentence2_tokens = greek_text_tokenizer(sentences[1])

    # Calculate smoothed-add-one probabilities
    probabilities1 = calculate_smoothed_one_probabilities(sentence1_tokens, len(sentence1_tokens))
    probabilities2 = calculate_smoothed_one_probabilities(sentence2_tokens, len(sentence2_tokens))

    # Calculate perplexity
    perplexity1 = calculate_perplexity(sentences[0], probabilities1)
    perplexity2 = calculate_perplexity(sentences[1], probabilities2)

    # Print results
    print("Sentence 1 Probability:", probabilities1)
    print("Sentence 1 Perplexity:", perplexity1)

    print("Sentence 2 Probability:", probabilities2)
    print("Sentence 2 Perplexity:", perplexity2)

    # Sample sentences
    # Sample sentences
    sample_sentence1 = generate_sentence(list(probabilities1.items()))
    sample_sentence2 = generate_sentence(list(probabilities2.items()))

    # Print sample sentences
    print("Sentence 1:", sample_sentence1)
    print("Sentence 2:", sample_sentence2)


# main()
if __name__ == "__main__":
    main()

