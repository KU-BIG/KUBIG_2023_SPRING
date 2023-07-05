# streamlit
from TTS.utils.synthesizer import Synthesizer
import streamlit as st
import IPython
import numpy as np
from io import BytesIO
from scipy.io.wavfile import write
from chatbot_code.inference import answer
from chatbot_code.bot import qachatbot
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import ast
import pandas as pd
import torch


@st.cache_resource
def load_chat_data():
    Chatbot_Data = pd.read_csv("chatbot_code/chatbot_last.csv")
    emb = torch.tensor(np.array([ast.literal_eval(vec) for vec in Chatbot_Data['EmbVector']]), dtype=torch.float32)
    return Chatbot_Data, emb

# 오디오 모델 불러오기 함수
@st.cache_resource
def load_synthesizer():
    synthesizer = Synthesizer(
    "../../content/drive/My Drive/Colab Notebooks/data/glowtts-v2/glowtts-v2-June-25-2023_02+36PM-3aa165ae/checkpoint_113000.pth.tar",
    "../../content/drive/My Drive/Colab Notebooks/data/glowtts-v2/glowtts-v2-June-25-2023_02+36PM-3aa165ae/config.json",
    None,
    "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/hifigan-v2-June-26-2023_07+23AM-3aa165ae/checkpoint_465000.pth.tar",
    "../../content/drive/My Drive/Colab Notebooks/data/hifigan-v2/hifigan-v2-June-26-2023_07+23AM-3aa165ae/config.json",
    None,
    None,
    False,
    )
    return synthesizer

# 챗봇 모델 불러오기 함수
@st.cache_resource
def load_chatbot_model(pre=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pre:
        #model_path = 'chatbot_code/output/final_model'
        model_path = 'kakaobrain/kogpt',
        tokenizer = AutoTokenizer.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )
        model = AutoModelForCausalLM.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype='auto', low_cpu_mem_usage=True
    ).to(device)
        return tokenizer, model
    
    if not pre:
        model_path = 'chatbot_code/output/last/final_model'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        return tokenizer, model



synthesizer = load_synthesizer()
symbols = synthesizer.tts_config.characters.characters
Chatbot_Data, emb = load_chat_data()
tokenizer, model = load_chatbot_model(pre = False)

#페이지 구성
st.title('쿠빅 침착맨 초대석')
st.subheader("침착맨연KU소")
st.image("ku_chim.jpg", width = 330)
st.subheader("반갑습니다 여러분의 귀염둥이 침착맨입니다.")
st.subheader("실시간 침착맨 음성봇")


#대화 실행
text = 0
with st.form(key="입력 form"):
    text = st.text_input("실시간 침착맨 음성봇")
    submitted = st.form_submit_button("대화 보내기")
    if submitted:
        
        with st.spinner("침착맨이 할 말을 생성중입니다."):
            question,answers = qachatbot(Chatbot_Data, emb, text)
            question,answer1 = qachatbot(Chatbot_Data, emb, answers)
            question,answer2 = qachatbot(Chatbot_Data, emb, answer1)
            question,answer3 = qachatbot(Chatbot_Data, emb, answer2)
            question,answer4 = qachatbot(Chatbot_Data, emb, answer3)
            question,answer5 = qachatbot(Chatbot_Data, emb, answer4)
            question,answer6 = qachatbot(Chatbot_Data, emb, answer5)
            
            answers = list(set([answers, answer1, answer2, answer3, answer4, answer5, answer6]))
            
            answers = ' '.join(answers)
            #prompt = f'{text}라는 질문이 왔습니다. 이때 대답은?'
            #prompt = f'{text}라는 고민이 라디오 사연으로 왔을 때 답장해줘'
            
            #answers = answers + '그리고 있잖아요.' + answer(model, tokenizer,synthesizer, prompt)
            st.write(answers)
            wav = synthesizer.tts(answers, None, None)
            IPython.display.display(IPython.display.Audio(wav, rate=22050))
            wav_norm = np.int16(wav/np.max(np.abs(wav)) * 32767)
            # wav_norm을 wav 바이트로 변환하고 BytesIO 객체를 생성합니다.
            virtual_file = BytesIO()
            write(virtual_file, 22050, wav_norm)

            # virtual_file을 처음부터 다시 읽습니다.
            virtual_file.seek(0)
            st.audio(virtual_file.read(), format = 'audio/wav')
    
st.subheader("침착맨 라디오")
with st.form(key="form"):
    text = st.text_input("침착맨 라디오")
    submitted = st.form_submit_button("사연 보내기")
    if submitted:
        with st.spinner("침착맨이 사연을 고민중입니다."):
            question,answers = qachatbot(Chatbot_Data, emb, text)
            question,answer1 = qachatbot(Chatbot_Data, emb, answers)
            question,answer2 = qachatbot(Chatbot_Data, emb, answer1)
            question,answer3 = qachatbot(Chatbot_Data, emb, answer2)
            answers = answers + answer1+ answer2+ answer3
            #prompt = f'{text}라는 질문이 왔습니다. 이때 대답은?'
            prompt = f'{text}라는 고민이 있습니다. 해결해주세요.'
            
            answers = answers + "또, " + answer(model, tokenizer,prompt)
            st.write(answers)
            wav = synthesizer.tts(answers, None, None)
            IPython.display.display(IPython.display.Audio(wav, rate=22050))
            wav_norm = np.int16(wav/np.max(np.abs(wav)) * 32767)
            # wav_norm을 wav 바이트로 변환하고 BytesIO 객체를 생성합니다.
            virtual_file = BytesIO()
            write(virtual_file, 22050, wav_norm)

            # virtual_file을 처음부터 다시 읽습니다.
            virtual_file.seek(0)
            st.audio(virtual_file.read(), format = 'audio/wav')
    

#실행
#streamlit run common_code/run.py --server.port 30007
#nohup streamlit run run_qa_copy.py --server.port 30007 &