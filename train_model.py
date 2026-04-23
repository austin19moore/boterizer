import os
import sys
import json
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from unsloth import FastModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import pandas as pd
from datasets import load_from_disk
import numpy as np


# General Config
CSV_PATH = "PATH.csv"
MODEL_PATH = "./MODEL_NAME_HERE"
FINETUNED_OUTPUT_DIR = "output_finetuned"
GGUF_OUTPUT_DIR = "output_gguf"
MERGED_OUTPUT_DIR = './output_merged'
QUANTIZATION = "q4_k_m"
USERNAME = "NAME_HERE"

# Data Config
MESSAGE_CONTEXT_LENGTH = 3
SEQ_LENGTH = 256
SYSTEM_PROMPT = f"""Your name is {USERNAME}, reply freely to the provided messages in their style. Messages are formatted as 'Author: Content'."""
qwen_model = False
BOTS = {'MEE6', 'Dyno', 'Carl-bot', 'Clyde'}
SKIP_PATTERNS = ['[image]', '[sticker]', '[attachment]', 'http://', 'https://']
SESSION_GAP_MINUTES = 30
MIN_RESPONSE_CHARS = 30

# Training Config, adjust BATCH_SIZE and GRADIENT_ACC_STEPS to fit in VRAM
BATCH_SIZE = 6
GRADIENT_ACC_STEPS = 8
EPOCHS = 1
r = 32
EVAL_STEPS = 200
SAVE_STEPS = 200


def load_and_prepare_data(csv_path):
    """
    Load Discord CSV and format for training.
    - Rows where Author == USERNAME are treated as assistant responses
    - Previous messages become conversation context
    - Consecutive messages from the same author are collapsed
    - Sessions are split on gaps > SESSION_GAP_MINUTES
    """
    if os.path.exists('train_data.jsonl'):
        print("Training data already exists. Skipping loading.")
        return

    # Load and sort by date
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df[['Author', 'Date', 'Content']]
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date').reset_index(drop=True)

    # Attempt to clean data
    df['Content'] = df['Content'].fillna('').astype(str).str.strip()
    df['Author'] = df['Author'].fillna('unknown').astype(str).str.strip()
    df = df[~df['Author'].isin(BOTS)]
    df['Content'] = df['Content'].str.replace(r'^@\w+\s*', '', regex=True).str.strip()
    df = df[df['Content'].str.len() >= 3]
    df = df[~df['Content'].apply(lambda x: any(p in x for p in SKIP_PATTERNS))]

    # Collapse consecutive messages from the same autho
    group = (df['Author'] != df['Author'].shift()).cumsum()
    df = df.groupby(group, sort=False).agg(
        Author=('Author', 'first'),
        Date=('Date', 'first'),
        Content=('Content', ' '.join)
    ).reset_index(drop=True)

    # Check for session gaps (ie > SESSION_GAP_MINUTES minutes between messages)
    def spans_session_gap(rows):
        timestamps = [r['Date'] for r in rows]
        for i in range(1, len(timestamps)):
            if (timestamps[i] - timestamps[i - 1]).total_seconds() > SESSION_GAP_MINUTES * 60:
                return True
        return False

    # Process rows
    conv_count = 0
    with open('train_data.jsonl', 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            if row['Author'] != USERNAME:
                continue
            if len(row['Content']) < MIN_RESPONSE_CHARS:
                continue
            if idx < MESSAGE_CONTEXT_LENGTH:
                continue

            prev_rows = df.iloc[idx - MESSAGE_CONTEXT_LENGTH:idx].to_dict('records')

            if spans_session_gap(prev_rows):
                continue

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            for msg in prev_rows:
                if not msg['Content']:
                    continue
                role = 'assistant' if msg['Author'] == USERNAME else 'user'
                messages.append({'role': role, 'content': f"{msg['Author']}: {msg['Content']}"})

            if not any(m['role'] == 'user' for m in messages):
                continue

            messages.append({'role': 'assistant', 'content': row['Content']})
            f.write(json.dumps({'messages': messages}, ensure_ascii=False) + '\n')
            conv_count += 1

    if conv_count == 0:
        print("Error: No valid conversation pairs found in CSV!")
        sys.exit(1)

    print(f"Created {conv_count} conversation training samples")
    return


def train_model():
    print(f"Loading model: {MODEL_PATH}...")

    model, processor = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=SEQ_LENGTH,
        load_in_4bit=True,
    )

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Uncomment to find chat template thinking tags
    # for i, line in enumerate(tokenizer.chat_template.split('\n')):
    #     if 'last_query_index' in line or 'think' in line.lower():
    #         print(i, repr(line))
    
    # Remove qwen thinking tags before formatting
    if qwen_model:
        lines = tokenizer.chat_template.split('\n')
        lines[99] = '        {%- if false %}'
        lines[100] = "            {{- '<|im_start|>' + message.role + '\\n' + content }}"
        tokenizer.chat_template = '\n'.join(lines)

    def formatting_func(batch):
        return {
            'text': [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                for convo in batch['messages']
            ]
        }

    print("Loading training dataset...")
    full_dataset = load_dataset('json', data_files='train_data.jsonl', split='train')
    dataset_dict = full_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    # Cache dataset
    CACHE = 'cached_dataset'
    if os.path.exists(CACHE):
        train_dataset = load_from_disk(f'{CACHE}/train')
        eval_dataset = load_from_disk(f'{CACHE}/eval')
    else:
        train_dataset = train_dataset.map(formatting_func, batched=True, num_proc=4)
        eval_dataset = eval_dataset.map(formatting_func, batched=True, num_proc=4)
        train_dataset = train_dataset.filter(lambda x: len(tokenizer.encode(x['text'])) <= SEQ_LENGTH)
        eval_dataset = eval_dataset.filter(lambda x: len(tokenizer.encode(x['text'])) <= SEQ_LENGTH)
        train_dataset.save_to_disk(f'{CACHE}/train')
        eval_dataset.save_to_disk(f'{CACHE}/eval')

    # See token length distribution, use to adjust SEQ_LENGTH
    lengths = [len(tokenizer.encode(t)) for t in train_dataset["text"]]
    print(f"Token lengths: {np.percentile(lengths, [50, 90, 95, 99, 100])}")

    print(f"Filtered dataset size: {len(train_dataset)}")
    print(f"Example from dataset: {train_dataset[0]['text']}")

    model = FastModel.get_peft_model(
        model,
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.1,
        finetune_vision_layers=False,
        use_gradient_checkpointing='unsloth',
    )

    total_steps = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACC_STEPS) * EPOCHS
    print(f"Total training steps: {total_steps}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field='text',
        packing=False,
        max_seq_length=SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACC_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=2e-4,
            lr_scheduler_type='cosine',
            weight_decay=0.05,
            warmup_ratio=0.1,
            logging_steps=50,
            eval_strategy='steps',
            eval_steps=EVAL_STEPS,
            save_strategy='steps',
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            save_on_each_node=False,
            greater_is_better=False,
            metric_for_best_model='eval_loss',
            output_dir=FINETUNED_OUTPUT_DIR,
            optim='adamw_8bit',
            bf16=True,
            tf32=True,
            remove_unused_columns=False,
            report_to='none',
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        ),
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=False)

    best_checkpoint = trainer.state.best_model_checkpoint
    print(f"Best checkpoint: {best_checkpoint}")

    print("Using base model for merge...")
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=SEQ_LENGTH,
        load_in_4bit=False,
    )

    if best_checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, best_checkpoint)
    else:
        print("Warning: No best checkpoint found, using final")

    print("Merging lora weights...")
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_OUTPUT_DIR)
    processor.save_pretrained(MERGED_OUTPUT_DIR)

    # Reload merged for GGUF conversion
    print("Loading merged model for GGUF conversion...")
    model, processor = FastModel.from_pretrained(
        model_name=MERGED_OUTPUT_DIR,
        max_seq_length=SEQ_LENGTH,
        load_in_4bit=False,
    )

    print(f"Converting to GGUF ({QUANTIZATION})...")
    model.save_pretrained_gguf(
        GGUF_OUTPUT_DIR,
        processor,
        quantization_method=[QUANTIZATION, 'f16'],
    )


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file '{CSV_PATH}' not found!")
        sys.exit(1)

    try:
        load_and_prepare_data(CSV_PATH)
        train_model()
        print("TRAINING COMPLETE!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
