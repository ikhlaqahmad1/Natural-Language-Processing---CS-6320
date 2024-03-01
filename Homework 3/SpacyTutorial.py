"""
Homework 3 by Ikhlaq Ahmad
ixa190000
"""

import spacy

# Load English language model large
nlp = spacy.load("en_core_web_lg")


def spacy_tutorial(sentence_for_tokenization):

    doc = nlp(sentence_for_tokenization)

    # Tokenization
    tokens = [token.text for token in doc]
    print("Tokenization:")
    print(tokens)

    # Parts of speech tagging
    pos_tags = [(token.text, token.pos_) for token in doc]
    print("\nPart of Speech Tagging:")
    print(pos_tags)

    # Named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print("\nNamed Entities:")
    print(entities)

    # Dependency parsing
    dependency_tree = [(token.text, token.dep_, token.head.text) for token in doc]
    print("\nDependency Parsing:")
    print(dependency_tree)
    print()


def custom_named_entity_recognition(text_for_ner):

    # Process the text
    doc = nlp(text_for_ner)

    # Custom NER
    patterns = {
        "ORG": ["Apple", "Google", "Microsoft"],
        "PERSON": ["John Smith", "Jane Doe", "James Brown"],
        "GPE": ["New York", "California", "Mountain View", "Seattle"]
    }

    # Identify entities based on custom patterns
    for ent in doc.ents:
        for label, values in patterns.items():
            if ent.text in values:
                # Overwrite the existing label with the custom one
                ent.label_ = label

    print("Named Entities: ")
    # Print the identified entities
    for ent in doc.ents:
        print("{}: {}".format(ent.text, ent.label_))


def main():
    spacy_tutorial("Apple is looking at buying an Indian startup for $1 billion")
    custom_named_entity_recognition("Apple is located in California. "
                                    "Google's headquarters is in Mountain View. "
                                    "John Smith works for Microsoft in New York.")


# main()
if __name__ == "__main__":
    main()
