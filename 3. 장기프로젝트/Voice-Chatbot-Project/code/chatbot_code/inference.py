import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, AutoModel

import time
import argparse
# streamlit
from TTS.utils.synthesizer import Synthesizer
import streamlit as st
import librosa
from pydub import AudioSegment
import IPython
import numpy as np
import io
from io import BytesIO
from scipy.io.wavfile import write
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a text"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="담대한",
        help="Prompting text",
    )

    args = parser.parse_args()
    # Sanity checks
    if args.model_name_or_path is None:
        raise ValueError("Need model name or path.")

    return args


words = ['침디', '침착맨', '안녕하세요', '안녕', '\n']
def remove_text_after_last_period(text, words):
    pattern = '|'.join(words)
    text = re.sub(pattern, '', text)
    return last_remove(text)
    
def last_remove(text):
    match = re.search(r'^(.*\.)', text)  # Greedy approach: find the longest match
    if match:
        return match.group(1)
    else:
        return text

def chat():
    args = parse_args()

    model_path = 'chatbot_code/output/final_model'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    while True:
        print('loading finished')
        input_ids = tokenizer.encode(args.prompt_text)
        # Check generation time
        start = time.time() 
        gen_ids = model.generate(torch.tensor([input_ids]),
                                max_length=70,
                                repetition_penalty=2.0,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                bos_token_id=tokenizer.bos_token_id,
                                # num_beams=5,
                                do_sample=False,
                                top_k=10, 
                                top_p=0.95,
                                use_cache=True)
        generated_text = tokenizer.decode(gen_ids[0,:].tolist())
        end = time.time()
        print(f'침착맨 : {generated_text}')
        print(f"{end - start:.5f} sec")
        return remove_text_after_last_period(generated_text)
        #return generated_text

def answer(model, tokenizer, text):
    input_ids = tokenizer.encode(text)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen_ids = model.generate(torch.tensor([input_ids]).to(device),
                            max_length=60,
                            repetition_penalty=2.0,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            #num_beams=4,
                            do_sample=True,
                            top_k=10, 
                            top_p=0.9,
                            temperature = 0.8,
                            use_cache=True)
    generated_text = tokenizer.decode(gen_ids[0,:].tolist())
    print(f'침착맨 : {generated_text}')
    # Remove the input text from the output
    if generated_text.startswith(text):  # if the output starts with the input
        result_text = generated_text[len(text):]  # remove the input from the output
    else:
        result_text = generated_text  # if not, just return the generated text
    result_text = result_text.strip()
    result_text = remove_text_after_last_period(result_text, words)
    
    return result_text   # Return the result text, removing leading/trailing whitespace

if __name__ == "__main__":
    chat()