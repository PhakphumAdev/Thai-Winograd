import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
class WinogradEvaluator:
    def __init__(self, model_name):
        # Initialize tokenizer and model based on the provided model name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)  # Move model to GPU if available
        self.model.eval()

    def sentence_log_probability(self, sentence):
        inputs = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs, labels=inputs)
        return -outputs.loss.item()

    def evaluate_schema_example(self, schema):
        pronoun = schema['pronoun']
        options = schema['options']
        text = schema['text']
        pronoun_loc = schema['pronoun_loc']

        results = {}
        for option in options:
            # Replace pronoun with the candidate
            candidate_text = text[:pronoun_loc] + option + text[pronoun_loc + len(pronoun):]
            log_prob = self.sentence_log_probability(candidate_text)
            results[option] = log_prob

        # Determine the best candidate based on the highest log probability
        best_candidate = max(results, key=results.get)
        return best_candidate == options[schema['label']]

    def evaluate_accuracy(self, examples):
        correct = sum(self.evaluate_schema_example(example) for example in examples)
        return correct / len(examples)
