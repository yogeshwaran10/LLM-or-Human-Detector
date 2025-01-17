import spacy
import numpy as np
from collections import Counter

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Role mappings for syntactic dependencies
role_mappings = {
    "nsubj": "s",
    "dobj": "o",
    "pobj": "o"
}

def get_entity_transitions(text):
    """Generate a feature vector based on entity grid transitions."""
    # Define all possible transitions in the required order
    all_possible_transitions = [
        "o->-", "o->o", "o->s", "o->x",
        "s->-", "s->o", "s->s", "s->x",
        "x->-", "x->o", "x->s", "x->x"
    ]
    
    transitions = []
    entities = []
    sentences_counter = 0
    
    # Process text with Spacy
    doc = nlp(text)
    sentences = [sent for sent in doc.sents]
    sentences_counter += len(sentences)
    
    # Extract entities and roles from sentences
    for sent in sentences:
        dict_sentence = {}
        for token in sent:
            if token.pos_ in ["PROPN", "NOUN", "PRON"] and token.dep_ != "compound":
                if token.text not in dict_sentence:
                    token_role = role_mappings.get(token.dep_, "x")
                    dict_sentence[token.text] = token_role
        entities.append(dict_sentence)
    
    # Compute transitions
    for i in range(len(entities) - 1):
        for key, role_1 in entities[i].items():
            role_2 = entities[i + 1].get(key, "-")
            transitions.append(f"{role_1}->{role_2}")
    
    # Count transitions and normalize weights
    count_transitions = Counter(transitions)
    weighted_transitions = {k: v / (sentences_counter - 1) for k, v in count_transitions.items()}
    
    # Generate feature vector based on predefined transitions order
    feature_vector = [weighted_transitions.get(t, 0) for t in all_possible_transitions]
    
    return np.array(feature_vector), all_possible_transitions

def entity_grid_dense(transitions, all_transitions):
    """Create a dense matrix from entity grid transitions."""
    essay_data = {}
    
    # Organize transitions by essay index
    for index, transition_data in enumerate(transitions):
        for transition, weight in transition_data.items():
            if index not in essay_data:
                essay_data[index] = {transition: weight}
            else:
                essay_data[index][transition] = weight
    
    # Construct dense matrix
    dense_matrix = []
    essay_indices = sorted(essay_data.keys())
    
    for essay_index in essay_indices:
        row = [essay_data[essay_index].get(transition, 0) for transition in all_transitions]
        dense_matrix.append(row)
    print(dense_matrix)
    return np.array(dense_matrix)