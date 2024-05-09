from datasets import dataset
import sys
from evaluation.eval import WinogradEvaluator
winograd_th = load_dataset("pakphum/winograd_th")
winograd_en = load_dataset("winograd_wsc","wsc285")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <huggingface_model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    evaluator = WinogradEvaluator(model_name)
    accuracy_en = evaluator.evaluate_accuracy(winograd_en['test'])
    accuracy_th = evaluator.evaluate_accuracy(winograd_th['train'])
    print(f'Accuracy with {model_name}: accuracy_en={accuracy_en:.2f} accuracy_th={accuracy_th:.2f}')
