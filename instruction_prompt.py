from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

huggingface_dataset_name = "knkarthick/dialogsum"

print("Loading dataset...")
dataset = load_dataset(huggingface_dataset_name)

example_indices = [40, 200]

dash_line = '-'.join('' for x in range(100))

print("Loading model...")
model_name='google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    
    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print('Example ', index)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}')
    print(dash_line)
