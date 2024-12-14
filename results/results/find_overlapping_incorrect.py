import pandas as pd

def find_overlapping_incorrect(csv_file_path, output_file_path):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Separate dataset_en and dataset_th
    dataset_en = data[data['Dataset'] == 'dataset_en'].reset_index(drop=True)
    dataset_th = data[data['Dataset'] == 'dataset_th'].reset_index(drop=True)

    # Find overlapping incorrect questions
    overlap_incorrect = dataset_en[(dataset_en['Correct'] == 0) & (dataset_th['Correct'] == 0)]

    # Merging dataset_en and dataset_th for comparison
    overlap_incorrect_th = dataset_th.loc[overlap_incorrect.index]
    combined_data = pd.concat([overlap_incorrect, overlap_incorrect_th[['Gold Label', 'Model Answer', 'Correct']]], axis=1)
    combined_data.columns = ['Dataset_en', 'Model Answer_en', 'Gold Label_en', 'Correct_en', 
                             'Gold Label_th', 'Model Answer_th', 'Correct_th']

    # Calculate statistics
    num_overlapping_incorrect = len(overlap_incorrect)
    total_incorrect_en = len(dataset_en[dataset_en['Correct'] == 0])
    total_incorrect_th = len(dataset_th[dataset_th['Correct'] == 0])
    
    # Calculate percentages
    percent_incorrect_en = (num_overlapping_incorrect / total_incorrect_en) * 100 if total_incorrect_en > 0 else 0
    percent_incorrect_th = (num_overlapping_incorrect / total_incorrect_th) * 100 if total_incorrect_th > 0 else 0

    # Print the results
    print(f"Number of overlapping incorrect questions: {num_overlapping_incorrect}")
    print(f"Percentage of overlapping incorrect questions in dataset_en: {percent_incorrect_en:.2f}%")
    print(f"Percentage of overlapping incorrect questions in dataset_th: {percent_incorrect_th:.2f}%")

    # Export the result to CSV
    combined_data.to_csv(output_file_path, index=False)
    print(f"Overlapping incorrect questions saved to {output_file_path}")

# Example usage
if __name__ == "__main__":
    csv_file_path = "command-r-plus-08-2024_results.csv"  # Input CSV file path
    output_file_path = "test.csv"  # Output CSV file path
    find_overlapping_incorrect(csv_file_path, output_file_path)