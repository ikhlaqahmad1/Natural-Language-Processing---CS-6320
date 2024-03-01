"""
Homework 3 by Ikhlaq Ahmad
ixa190000
"""
# Dependencies
from gensim.models import KeyedVectors

# Load GloVe word vectors from a text file
glove_file = "glove.6B.50d.txt"

# Load pre-trained data using try/except
try:
    word_vectors = KeyedVectors.load_word2vec_format(glove_file, binary=False)
except Exception as e:
    print("Error loading GloVe word vectors:", e)
    exit()


""""######################################## Question 1 #############################################################"""


# Type 1: Semantic Disambiguation
def semantic_disambiguation(word, context_words):
    closest_word = word

    # Infinite upper bound
    min_distance = float('inf')

    for context_word in context_words:
        try:
            distance = word_vectors.distance(word, context_word)
            if distance < min_distance:
                min_distance = distance
                closest_word = context_word
        except KeyError:
            print("An exception occurred")
            pass
    return closest_word


# Examples of Semantic Disambiguation
print("Semantic Disambiguation Examples:")
print("1. {} in context of finance: {}".format('bank', semantic_disambiguation('bank', ['finance', 'river'])))
print("2. {} in context of names: {}".format('paris', semantic_disambiguation('paris', ['geography', 'names'])))
print("3. {} in context of sports: {}".format('bat', semantic_disambiguation('bat', ['animals', 'sports'])))
print("4. {} in context of sports: {}".format('car', semantic_disambiguation('car', ['vehicle', 'mechanic'])))
print("5. {} in context of sports: {}".format('bike', semantic_disambiguation('bike', ['vehicle', 'keys'])))
print()


# Type 2: Semantic Similarity - Finding words similar to a given word
def find_similar_words(word, topn=5):
    similar_words = word_vectors.most_similar(word, topn=topn)
    return [similar[0] for similar in similar_words]


# Examples of Semantic Similarity
print("Semantic Similarity Examples:")
print("1. Words similar to {}: {}".format('apple', find_similar_words('apple')))
print("2. Words similar to {}: {}".format('lion', find_similar_words('lion')))
print("3. Words similar to {}: {}".format('computer', find_similar_words('computer')))
print("4. Words similar to {}: {}".format('mars', find_similar_words('mars')))
print("5. Words similar to {}: {}".format('usa', find_similar_words('usa')))
print()

""""########################################### Question 2 ##########################################################"""


# Function to check if the causal relationship is captured by Word2Vec
def check_causal_relationship(word_1, word_2):
    try:
        similarity = word_vectors.similarity(word_1, word_2)
        return similarity
    except KeyError:
        return "One or both words not in vocabulary"


# Define examples of causal relationships
causal_examples = [
    ("fire", "smoke"),  # Fire causes smoke
    ("rain", "umbrella"),  # Rain leads to the use of an umbrella
    ("eating", "fullness")  # Eating leads to a feeling of fullness
]

print("Casual relations not captured by word2vec")
# Test causal examples
for example in causal_examples:
    word1, word2 = example
    similarity_score = check_causal_relationship(word1, word2)
    print(f"Causal relationship between '{word1}' and '{word2}': {similarity_score}")
print()

""""######################################### Question 3 ############################################################"""

# Define sets of professions associated with different genders
technology_professions = ["programmer", "engineer", "scientist"]

# Calculate average cosine similarity between each technology profession and "man"
avg_similarity_man = sum(word_vectors.similarity("man", prof) for prof in technology_professions) / len(
    technology_professions)

# Calculate average cosine similarity between each humanities profession and "woman"
avg_similarity_woman = sum(word_vectors.similarity("woman", prof) for prof in technology_professions) / len(
    technology_professions)

# Print the average similarities
print("Biases in the entity relations:")
print("Average similarity between technology professions and 'man':", avg_similarity_man)
print("Average similarity between technology professions and 'woman':", avg_similarity_woman)
