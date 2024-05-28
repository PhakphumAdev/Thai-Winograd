from datasets import load_dataset
from openai import OpenAI
import anthropic
import time
from tqdm import tqdm
import json
# Typhoon API key
client_typhoon = OpenAI(
    api_key='',
    base_url="https://api.opentyphoon.ai/v1",
)
# GPT API key
client_gpt = OpenAI(api_key='')
# Claude API key
client = anthropic.Anthropic(
    api_key=""
)



# Function to create Winograd schema prompt
def create_winograd_prompt(data):
    # Extract necessary information
    text = data['text']
    quote = data['quote']
    pronoun = data['pronoun']
    options = data['options']

    # Create the prompt with bold pronoun
    prompt = f"""{text.replace(pronoun, f'**{pronoun}**')}\nSnippet: {quote.replace(pronoun, f'**{pronoun}**')}\nOptions:\n{options[0]}\n{options[1]}"""

    return prompt.strip()

# Process the dataset and generate prompts
def process_dataset(dataset):
    prompts = []
    for entry in dataset['test']:
        prompt = create_winograd_prompt(entry)
        prompts.append(prompt)
    return prompts

# Typhoon
def ask_typhoon(model,systemPrompt,userPrompt):

    try:
        # Make an API call to OpenAI
        response = client_typhoon.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ],
            temperature=0,
            seed=77
          )
        return response.choices[0].message
    except Exception as e:
        return f"An error occurred: {e}"
    
# GPT
def ask_gpt(model,systemPrompt,userPrompt):

    try:
        # Make an API call to OpenAI
        response = client_gpt.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ],
            temperature=0,
            seed=77
        )
        return response.choices[0].message
    except Exception as e:
        return f"An error occurred: {e}"

# Claude
def ask_claude(model,systemPrompt,userPrompt):

    try:
        # Make an API call to OpenAI
        response = client.messages.create(
            model=model,
            max_tokens=50,
            temperature=0.0,
            system=systemPrompt,
            messages=[
                {"role": "user", "content": userPrompt}
            ],
        )
        return response.content
    except Exception as e:
        return f"An error occurred: {e}"

# Define the response extractors for different models
def gpt_response_extractor(response):
    return response.content

def claude_response_extractor(response):
    return response[0].text

# Define a function to test a task with different models and languages
def test_model_accuracy(system_prompt, model, language, prompts, labels, options, ask_function, response_extractor, sleep_time=0):
    ans = []  # List to store generated responses
    correct = 0  # Counter for correctly matched responses
    all = 0  # Counter for all attempts
    results = []  # List to store detailed results

    for i, p in tqdm(enumerate(prompts), total=len(prompts), desc=f"Processing {model} in {language}"):
        if sleep_time > 0:
            time.sleep(sleep_time)  # Sleep for the specified time
        ret = response_extractor(ask_function(model, system_prompt, p))
        ans.append(ret)  # Store the generated response

        # Determine the gold label based on the correct option
        gold_label = options[i][labels[i]]

        # Check if the generated response matches the gold label
        if ret == gold_label:
            correct += 1  # Increment correct count if there's a match

        all += 1  # Increment total attempts

        # Save the question number, gold label, and model response
        results.append({
            "question_number": i,
            "gold_label": gold_label,
            "model_response": ret
        })

    # Calculate accuracy as the number of correct answers divided by the total number of attempts
    accuracy = correct / all if all > 0 else 0  # Ensure division by zero is handled
    accuracy_percentage = accuracy * 100
    print(f"Accuracy for {model} in {language}: {accuracy_percentage:.2f}%")

    # Generate file names based on model name, language, and accuracy
    model_name = model.replace(' ', '_')
    accuracy_str = f"{accuracy_percentage:.2f}".replace('.', '_')
    json_file = f"model_accuracy_{model_name}_{language}_{accuracy_str}.json"

    # Save results to a JSON file
    with open(json_file, 'w') as jf:
        json.dump(results, jf, indent=4)

# Load a dataset
dataset_en = load_dataset('winograd_wsc','wsc285')
dataset_th = load_dataset('pakphum/winograd_th')

prompt_english=process_dataset(dataset_en)
label_english=dataset_en['test']['label']
option_english=dataset_en['test']['options']

prompt_thai=process_dataset(dataset_th)
label_thai=dataset_th['test']['label']
option_thai=dataset_th['test']['options']

system_prompt = "You will be provided with a sentence and a snippet containing a pronoun enclosed in asterisks (**). Your task is to determine the correct referent of the pronoun from the given options. Respond only with one of the provided choices, exactly as it is written. For example, if the options are 'The city councilmen' and 'The demonstrators', respond only with 'The city councilmen' or 'The demonstrators'."


# Test with GPT-4 in English
test_model_accuracy(system_prompt, "gpt-4", "english", prompt_english, label_english, option_english, ask_gpt, gpt_response_extractor)

# Test with GPT-4 in Thai
test_model_accuracy(system_prompt, "gpt-4", "thai", prompt_thai, label_thai, option_thai, ask_gpt, gpt_response_extractor)

# Test with GPT-3.5 Turbo in English
test_model_accuracy(system_prompt, "gpt-3.5-turbo-0125", "english", prompt_english, label_english, option_english, ask_gpt, gpt_response_extractor)

# Test with GPT-3.5 Turbo in Thai
test_model_accuracy(system_prompt, "gpt-3.5-turbo-0125", "thai", prompt_thai, label_thai, option_thai, ask_gpt, gpt_response_extractor)

# Test with Typhoon in English
test_model_accuracy(system_prompt, "typhoon-instruct-0219", "english", prompt_english, label_english, option_english, ask_typhoon, gpt_response_extractor)

# Test with Typhoon in Thai
test_model_accuracy(system_prompt, "typhoon-instruct-0219", "thai", prompt_thai, label_thai, option_thai, ask_typhoon, gpt_response_extractor)

# Test with Claude-3-haiku in English
test_model_accuracy(system_prompt, "claude-3-haiku-20240307", "english", prompt_english, label_english, option_english, ask_claude, claude_response_extractor, sleep_time=5)

# Test with Claude-3-haiku in Thai
test_model_accuracy(system_prompt, "claude-3-haiku-20240307", "thai", prompt_thai, label_thai, option_thai, ask_claude, claude_response_extractor, sleep_time=5)

# Test with Claude-3-sonnet in English
test_model_accuracy(system_prompt, "claude-3-sonnet-20240229", "english", prompt_english, label_english, option_english, ask_claude, claude_response_extractor, sleep_time=5)

# Test with Claude-3-sonnet in Thai
test_model_accuracy(system_prompt, "claude-3-sonnet-20240229", "thai", prompt_thai, label_thai, option_thai, ask_claude, claude_response_extractor, sleep_time=5)

# Test with Claude-3-opus in English
test_model_accuracy(system_prompt, "claude-3-opus-20240229", "english", prompt_english, label_english, option_english, ask_claude, claude_response_extractor, sleep_time=5)

# Test with Claude-3-opus in Thai
test_model_accuracy(system_prompt, "claude-3-opus-20240229", "thai", prompt_thai, label_thai, option_thai, ask_claude, claude_response_extractor, sleep_time=5)