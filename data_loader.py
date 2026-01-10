import json
import pandas as pd
from datasets import Dataset
from prompts import generate_prompt, generate_test_prompt
import wandb

def train_data(path='data/train.json'):
    run = wandb.init(project="Artifacts Registry", job_type="data-loading")

    data_json = []
    with open(path, 'r') as file:
        for line in file:
            data_json.append(json.loads(line))

    # Save data locally
    raw_medmcqa_df = pd.DataFrame(data_json)
    raw_medmcqa_df.to_csv('raw_medmcqa.csv', index=False)  # Save as CSV

    # Create Artifact object
    raw_dataset_artifact = wandb.Artifact(name='raw_medmcqa', type='dataset')

    # Add files to the artifact (multiple)
    raw_dataset_artifact.add_file('raw_medmcqa.csv')

    # Log the artifact
    wandb.log_artifact(raw_dataset_artifact, aliases=["raw"])

    print(f'data_json[:5] : {data_json[0]}')

    data_extracted = [
        {'question': entry['question'], 'cop': entry['cop'], 'subject_name': entry['subject_name'], 'topic_name': entry['topic_name'], 'exp': entry['exp'], 'opa': entry['opa'], 'opb': entry['opb'], 'opc': entry['opc'], 'opd': entry['opd'], 'exp': entry['exp']}
        for entry in data_json
    ]
    print(f'data_extracted[:5]: {data_extracted[0]}')

    data_df = pd.DataFrame(data_extracted)

    data_df['topic_name'] = data_df['topic_name'].fillna('Unknown')
    data_df['exp'] = data_df['exp'].fillna('')

    # Get clean data
    data_df.to_csv('clean_medmcqa.csv', index=False)

    # Create and Log New Artifact
    run.log_artifact(
        artifact_or_path="clean_medmcqa.csv",
        name="clean_medmcqa",
        type="dataset",
        aliases=["cleaned"]
    )

    wandb.finish()
    data = Dataset.from_pandas(data_df)
    return data

def instruction_tuned_dataset(data):
    # Apply to dataset
    dataset = data.map(generate_prompt, batched=True)
    print(dataset[0])
    return dataset

def valid_data(path='data/dev.json'):
    val_data_json = []
    with open(path, 'r') as file:
        for line in file:
            val_data_json.append(json.loads(line))

    val_data_extracted = [
        {'question': entry['question'], 'cop': entry['cop'], 'opa': entry['opa'], 'opb': entry['opb'], 'opc': entry['opc'], 'opd': entry['opd']}
        for entry in val_data_json
    ]

    val_data_df = pd.DataFrame(val_data_extracted)
    val_data_df['text'] = val_data_df.apply(lambda x: generate_test_prompt(x), axis=1)
    print(val_data_df['text'][0])
    return val_data_df