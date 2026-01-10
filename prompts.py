# official Mistral 7B instruct prompt
prompt_template = """
Question:
{}
[INST] Solve this post graduate medical entrance exam MCQ and provide the correct option. [/INST]
Answer: {} </s>"""

def generate_prompt(examples):
    texts = []
    # extract columns
    cops = examples['cop']
    opas = examples['opa']
    opbs = examples['opb']
    opcs = examples['opc']
    opds = examples['opd']
    questions = examples['question']

    for cop_idx, opa, opb, opc, opd, question_text in zip(cops, opas, opbs, opcs, opds, questions):
        # Build the question string with options
        question_with_options = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}'.format(
            question_text, opa, opb, opc, opd
        )
        # Determine the correct answer
        if cop_idx == 1:
            answer = opa
        elif cop_idx == 2:
            answer = opb
        elif cop_idx == 3:
            answer = opc
        elif cop_idx == 4:
            answer = opd
        else:
            answer = ""  # fallback if cop_idx is invalid

        # Format prompt
        text = prompt_template.format(question_with_options, answer)
        texts.append(text)

    return {'text': texts}

def generate_test_prompt(x):
    question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\n'.format(x['question'], x['opa'], x['opb'], x['opc'], x['opd'])
    prompt = f"""
    Question:
    {question}
    [INST] Solve this post graduate medical entrance exam MCQ and answer correctly. [/INST]
    Answer: """
    return prompt