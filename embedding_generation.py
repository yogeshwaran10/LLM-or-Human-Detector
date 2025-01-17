import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
import spacy
from collections import Counter
import textstat
from lexical_diversity import lex_div as ld
from entity_grid import get_entity_transitions

class EmbeddingGenerator:
    def __init__(self):
        # Initialize BERT components
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.bert_model = RobertaModel.from_pretrained('roberta-base')
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device)
        self.bert_model.eval()  # Set to evaluation mode
        
        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define features list for lexical diversity
        self.features_list = [
            "ttr", "root_ttr", "log_ttr", "maas_ttr", "msttr", 
            "mattr", "hdd", "mtld", "mtld_ma_wrap", "mtld_ma_bid"
        ]

    def encode_text(self, text: str):
        """Generate BERT embeddings for the input text."""
        encoded_inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**encoded_inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def get_readability_scores(self, text: str) -> dict:
        """Calculate readability scores."""
        readability = {
            "flesch_kincaid": textstat.flesch_kincaid_grade(text),
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "gunning_fog": textstat.gunning_fog(text),
            "smog_index": textstat.smog_index(text),
            "coleman_liau": textstat.coleman_liau_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "dale_chall": textstat.dale_chall_readability_score(text),
        }
        print(f"Readability Scores: {readability}")
        print(f"Number of Readability Features: {len(readability)}")
        return readability

    def get_style_features(self, text: str) -> tuple:
        """Extract stylistic features."""
        doc = self.nlp(text)
        pos_tokens = []
        shape_tokens = []
        LATIN = ["i.e.", "e.g.", "etc.", "c.f.", "et", "al."]

        for word in doc:
            if word.is_punct or word.is_stop or word.text in LATIN:
                pos_target = word.text
                shape_target = word.text
            else:
                pos_target = word.pos_
                shape_target = word.shape_
            pos_tokens.append(pos_target)
            shape_tokens.append(shape_target)

        print(f"POS Tokens: {pos_tokens[:20]}")  # Show the first 20 POS tags for preview
        print(f"Shape Tokens: {shape_tokens[:20]}")  # Show the first 20 shape tags for preview
        print(f"Number of Stylistic Features (POS + Shape): {len(pos_tokens) + len(shape_tokens)}")
        return " ".join(pos_tokens), " ".join(shape_tokens)

    def preprocess_for_lexical(self, text: str) -> list:
        """Preprocess text for lexical diversity features."""
        doc = self.nlp(text)
        return [f"{w.lemma_}_{w.pos_}" for w in doc 
                if not w.pos_ in ["PUNCT", "SYM", "SPACE"]]

    def get_lexical_diversity(self, text: str) -> dict:
        """Calculate lexical diversity features."""
        preprocessed = self.preprocess_for_lexical(text)
        result = {}
        for feature in self.features_list:
            try:
                result[feature] = getattr(ld, feature)(preprocessed)
            except:
                result[feature] = 0.0  # Fallback value if calculation fails
        print(f"Lexical Diversity Features: {result}")
        print(f"Number of Lexical Diversity Features: {len(result)}")
        return result

    # def get_entity_transitions(self, text: str) -> np.ndarray:
    #     """Calculate entity grid transitions and ensure output follows the predefined order."""
    #     doc = self.nlp(text)
    #     sentences = list(doc.sents)
    #     sentences_counter = len(sentences)
        
    #     if sentences_counter <= 1:
    #         # Return zero values if text is too short
    #         zero_transitions = {trans: 0.0 for trans in [
    #             "o->-", "o->o", "o->s", "o->x", 
    #             "s->-", "s->o", "s->s", "s->x", 
    #             "x->-", "x->o", "x->s", "x->x"
    #         ]}
    #         return np.array(list(zero_transitions.values()))

    #     # Rest of the entity grid processing...
    #     entities = []
    #     transitions = []
    #     role_mappings = {"nsubj": "s", "dobj": "o", "pobj": "o"}

    #     for sent in sentences:
    #         dict_sentence = {}
    #         for token in sent:
    #             if token.pos_ in ["PROPN", "NOUN", "PRON"] and token.dep_ != "compound":
    #                 if token.text not in dict_sentence:
    #                     token_role = role_mappings.get(token.dep_, "x")
    #                     dict_sentence[token.text] = token_role
    #         entities.append(dict_sentence)

    #     for i in range(len(entities) - 1):
    #         for key, role_1 in entities[i].items():
    #             role_2 = entities[i + 1].get(key, "-")
    #             transitions.append(f"{role_1}->{role_2}")

    #     count_transitions = Counter(transitions)
    #     weighted_transitions = {k: v / (sentences_counter - 1) for k, v in count_transitions.items()}

    #     # Predefined order of entity transitions
    #     all_transitions = [
    #         "o->-", "o->o", "o->s", "o->x", 
    #         "s->-", "s->o", "s->s", "s->x", 
    #         "x->-", "x->o", "x->s", "x->x"
    #     ]
        
    #     # Ensure the order of transitions matches the predefined order
    #     ordered_transitions = {t: weighted_transitions.get(t, 0.0) for t in all_transitions}
        
    #     # Create feature vector (dense representation) for model input
    #     feature_vector = [ordered_transitions[t] for t in all_transitions]

    #     print(f"Entity Transitions (Ordered): {ordered_transitions}")
    #     print(f"Feature Vector: {feature_vector}")
    #     print(f"Number of Entity Transition Features: {len(feature_vector)}")
    #     dense_matrix_tensor = torch.tensor([feature_vector], device=self.device, dtype=torch.float32)
    #     return dense_matrix_tensor.cpu().numpy().flatten()  # Move to CPU and convert to numpy  # Return numpy array directly instead of tensor

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate the complete embedding vector for the input text."""
        try:
            # Get BERT embeddings
            print(f"Generating BERT embeddings for text: {text}")
            bert_embeddings = self.encode_text(text).flatten()
            print(f"BERT Embedding Shape: {bert_embeddings.shape}")
            print(f"Number of BERT Features: {bert_embeddings.size}")

            # Get readability features
            readability = self.get_readability_scores(text)

            # Get stylistic features
            _, _ = self.get_style_features(text)  # We don't need these directly

            # Get lexical diversity features
            lexical_features = self.get_lexical_diversity(text)

            # Get entity transitions (now returns numpy array directly)
            transitions, _ =get_entity_transitions(text)
            print(transitions)
            # Combine all features
            feature_vector = np.concatenate([
                np.array(list(readability.values())),
                np.array(list(lexical_features.values())),
                transitions
            ])
            print(bert_embeddings)
            print(feature_vector)
            print(f"Feature Vector Shape: {feature_vector.shape}")
            print(f"Number of Combined Features: {feature_vector.size}")
            
            # Concatenate BERT embeddings with other features
            final_vector = np.concatenate([ feature_vector,bert_embeddings])
            print(f"Final Embedding Shape: {final_vector.shape}")
            print(f"Total Number of Features: {final_vector.size}")
            
            return final_vector
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

# Create a singleton instance
generator = EmbeddingGenerator()

def get_embedding(text: str) -> np.ndarray:
    """Main function to be called from Flask app."""
    return generator.generate_embedding(text)