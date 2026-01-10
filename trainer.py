from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig, TrainingArguments
from unsloth import is_bfloat16_supported
from trl import SFTTrainer # transformer RL library - Supervised FineTuning Trainer

def define_trainer(model, tokenizer, dataset, max_seq_length):
    training_arguments = TrainingArguments(
                output_dir="mistral-lora-finetuned-medmcqa",
                per_device_train_batch_size = 8, # The batch size per GPU/TPU core
                gradient_accumulation_steps = 1, # Number of steps to perform befor each gradient accumulation
                warmup_steps = 5, # Few updates with low learning rate before actual training
                max_steps = 5000, # Specifies the total number of training steps (batches) to run.
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 50,
                num_train_epochs=1,
                optim="paged_adamw_32bit", # Optimizer
                weight_decay = 0.01,
                lr_scheduler_type = "cosine",
                seed = 3407,
                save_strategy="epoch",
                report_to = "none", # Use this for WandB etc for observability
                push_to_hub = True
        )

    trainer = SFTTrainer(
                model = model,
                tokenizer = tokenizer,
                train_dataset = dataset,
                dataset_text_field = "text",
                max_seq_length = max_seq_length,
                dataset_num_proc = 2, # Number of processors to use for processing the dataset
                packing = False, # Can make training 5x faster for short sequences.
                args = training_arguments
        )
    return trainer