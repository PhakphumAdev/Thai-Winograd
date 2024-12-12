import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Function to load results from JSON files
def load_results_from_json(json_files):
    all_results = []
    for file in json_files:
        # Extract model and language from filename
        base_name = os.path.basename(file)
        parts = base_name.split('_')
        model = parts[2]
        language = parts[3]

        with open(file, 'r') as f:
            results = json.load(f)
            # Add model and language information to each result
            for result in results:
                result['model'] = model
                result['language'] = language
            all_results.extend(results)
    return all_results

# Function to visualize the results
def visualize_results(df):
    models = df['model'].unique()
    languages = df['language'].unique()

    for language in languages:
        plt.figure(figsize=(20, 10))
        questions = df['question_number'].unique()
        bar_width = 0.15
        bar_positions = list(range(len(questions)))
        
        for i, model in enumerate(models):
            df_lang_model = df[(df['model'] == model) & (df['language'] == language)]
            correct_values = df_lang_model.set_index('question_number')['correct'].reindex(questions).fillna(0).values
            plt.bar([p + bar_width * i for p in bar_positions], correct_values, width=bar_width, label=model)

        plt.xlabel('Question Number')
        plt.ylabel('Correct (1) / Incorrect (0)')
        plt.title(f'Comparison of Models in {language.capitalize()}')
        plt.xticks([p + bar_width * len(models) / 2 for p in bar_positions], questions)
        plt.legend()
        plt.show()

# Get the list of JSON result files
json_files = glob.glob('results/model_accuracy_*.json')

# Load the results from JSON files
all_results = load_results_from_json(json_files)

# Print out the first few results for debugging
print("Sample results:", all_results[:3])  # Debugging line

# Convert all results to a DataFrame for visualization
df_results = pd.DataFrame(all_results)

# Print out the DataFrame for debugging
print("DataFrame head:", df_results.head())  # Debugging line

# Ensure the 'correct' column is created based on model_response and gold_label
df_results['correct'] = df_results['model_response'] == df_results['gold_label']

# Visualize the results
visualize_results(df_results)