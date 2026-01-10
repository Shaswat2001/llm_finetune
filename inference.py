# Inference
from data_loader import valid_data
from huggingface_hub import list_repo_files
from unsloth import FastLanguageModel
from model_loader import load_model_for_inference
import re
import tqdm


def solve_question(question_prompt, model, tokenizer):
    inputs = tokenizer(question_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True,
                         temperature = 0.2, min_p = 0.1)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer

def normalize(text):
    if not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'^\d+\.\s*', '', text)      # remove numbering
    text = re.sub(r'[^\w\s%/.-]', '', text)    # remove punctuation
    text = re.sub(r'\s+', ' ', text)           # normalize spaces
    return text.strip()

# import gc
# gc.collect()
# torch.cuda.empty_cache()

def calculate_acc(val_data_df, model, tokenizer):
    print(list_repo_files("srao0996/mistral-lora-finetuned-medmcqa"))
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
        val_data_df['text'][4],
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True,
                            temperature = 0.2, min_p = 0.1)
    response = tokenizer.batch_decode(outputs)

    print(response)
    print(outputs)

    all_answers = []

    val_data_prompts = list(val_data_df['text'])
    for i in tqdm.tqdm(range(0, len(val_data_prompts), 16)):
        question_prompts = val_data_prompts[i:i+16]
        ans = solve_question(question_prompts)
        ans_option = []
        for text in ans:
            ans_option.append(re.search(r'Answer: \s*(.*)', text).group(1))

        all_answers.extend(ans_option)

    print(len(all_answers))

    correct_answers = []
    for i in range(len(val_data_df)):
        if val_data_df['cop'][i] == 1:
            correct_answers.append(val_data_df['opa'][i])
        elif val_data_df['cop'][i] == 2:
            correct_answers.append(val_data_df['opb'][i])
        elif val_data_df['cop'][i] == 3:
            correct_answers.append(val_data_df['opc'][i])
        elif val_data_df['cop'][i] == 4:
            correct_answers.append(val_data_df['opd'][i])

    print(len(correct_answers))

    correct_count = 0
    for i in range(len(val_data_df)):
        print(f'{correct_answers[i]} ";" {all_answers[i]}')
        left, right = correct_answers[i], all_answers[i]

        left_n = normalize(left)
        right_n = normalize(right)

        if left_n == right_n:
            correct_count += 1

    threshold_score = 0.5
    matching_scores = [sum(normalize(g) == normalize(t) for g, t in zip(gen, truth)) / max(len(gen), len(truth)) for gen, truth in zip(all_answers, correct_answers)]
    correct_count = sum(score >= threshold_score for score in matching_scores)
    print(f'accuracy after thresholding : {correct_count/len(val_data_df)}')
    return correct_count/len(val_data_df)