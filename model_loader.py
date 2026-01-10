import torch
from unsloth import FastLanguageModel
# from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model

max_seq_length = 2048 # Choose any! Unsloth also supports RoPE (Rotary Positinal Embedding) scaling internally.
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

def load_model(model_name='unsloth/mistral-7b-instruct-v0.2-bnb-4bit'):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name =model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )

    """model_name:Specifies the name of the pre-trained model to load.


    max_seq_length:Defines the maximum sequence length (in tokens) that the model can process. max_seq_length = 2048 allows the model to process sequences up to 2048 tokens long.


    dtype:Specifies the data type for model weights and computations. None: Automatically selects the appropriate data type based on the hardware. torch.
    float16: Uses 16-bit floating point precision, reducing memory usage and potentially increasing speed on compatible GPUs. torch.bfloat16: Similar to float16 but with a wider dynamic range, beneficial for certain hardware like NVIDIA A100 GPUs.


    load_in_4bit:Determines whether to load the model using 4-bit quantization.Ideal for scenarios where memory efficiency is crucial, such as deploying models on edge devices or during experimentation.


    Now, we'll use the get_peft_model from unsloth's FastLanguageModel class to attach adapters (peft layers) on top of the models in order to perform QLoRA
    """

    # define the LoRA configuration for PEFT
    model = FastLanguageModel.get_peft_model(
                        model,
                        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                        lora_alpha = 32, # a higher alpha value assigns more weight to the LoRA activations
                        lora_dropout = 0, # Supports any, but = 0 is optimized
                        bias = "none",    # Supports any, but = "none" is optimized
                        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                        random_state = 3407,
                        use_rslora = False,
                        loftq_config = None,
                        target_modules=[
                            "q_proj", #Wq + delta(W)
                            "k_proj", #Wk + delta(k)
                            "v_proj",
                            "o_proj",
                            # matrices for the fully-connected layer
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                            "lm_head",
                        ]
                    )
    
    # r: The rank of the low-rank matrices in LoRA; higher values can capture more information but increase memory usage.


    # target_modules: List of model components (e.g., "q_proj", "k_proj") where LoRA adapters are inserted for fine-tuning.


    # lora_alpha: Scaling factor for the LoRA updates; controls the impact of the adapters on the model's outputs.


    # lora_dropout: Dropout rate applied to LoRA layers during training to prevent overfitting.


    # bias: Specifies how biases are handled in LoRA layers; options include "none", "all", or "lora_only".


    # use_gradient_checkpointing: Enables gradient checkpointing to reduce memory usage during training; "unsloth" uses Unsloth's optimized version.


    # random_state: Seed for random number generators to ensure reproducibility of training results.


    # use_rslora: Boolean indicating whether to use Rank-Stabilized LoRA (rsLoRA) for potentially more stable training.


    # ` loftq_config: Configuration for Low-Rank Quantization (LoftQ); set to None to disable this feature.

    # Trainer Setup:

    # model and tokenizer: These are the model and tokenizer objects that will be trained.

    # train_dataset: The dataset used for training.

    # dataset_text_field: Specifies the field in the dataset that contains the text data.

    # max_seq_length: Maximum sequence length for the input data.

    # dataset_num_proc: Number of processes to use for data loading.

    # packing: If True, enables sequence packing (concatenates multiple examples into a single sequence to better utilize tokens).

    # Training Arguments:

    # per_device_train_batch_size: Number of samples per batch for each device.

    # gradient_accumulation_steps: Number of steps to accumulate gradients before updating model weights.

    # warmup_steps: Number of steps for learning rate warmup.

    # max_steps: Total number of training steps.

    # learning_rate: Learning rate for the optimizer.

    # fp16 and bf16: Specifies whether to use 16-bit floating point precision or bfloat16, depending on hardware support.

    # logging_steps: Frequency of logging training progress.

    # optim: Optimizer type, here using an 8-bit version of AdamW.

    # weight_decay: Regularization parameter for weight decay.

    # lr_scheduler_type: Type of learning rate scheduler.

    # seed: Random seed for reproducibility.

    # output_dir: Directory where the training outputs will be saved.

    # report_to: Integration for observability tools like "wandb", "tensorboard", etc.

    return model, tokenizer

def load_model_for_inference(model_name='srao0996/mistral-lora-finetuned-medmcqa'):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    return model, tokenizer