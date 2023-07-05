import urllib.request
import pandas as pd
from pororo import Pororo
import torch
import numpy as np
from sentence_transformers import util
import ast

sTe = Pororo(task="sentence_embedding", lang="ko")

'''
#데이터 다운로드
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
    filename="ChatBotData.csv",
)



Chatbot_Data = pd.read_csv("ChatBotData.csv")
a = sTe(Chatbot_Data['Q'][0])
print(a)
breakpoint()

Chatbot_Data['EmbVector'] = Chatbot_Data['Q'].apply(lambda x: sTe(x))
Chatbot_Data['EmbVector'] = Chatbot_Data['EmbVector'].astype(str)
'''
#Chatbot_Data.to_csv('chat_last.csv', index = False)


#Chatbot_Data = pd.read_csv("chatbot_last.csv")
#breakpoint()
#emb = torch.tensor(np.array(Chatbot_Data['EmbVector'].apply(ast.literal_eval)), dtype=torch.float32)
#breakpoint()

#emb = torch.tensor(np.array(Chatbot_Data['EmbVector'].apply(ast.literal_eval)), dtype=torch.float32)
def qachatbot(Chatbot_Data, emb,characteristic):
    
    #emb = torch.tensor(np.array(Chatbot_Data['EmbVector'].tolist()), dtype=torch.float32)
    characteristic = sTe(characteristic)
    # 질문을 Tensor로 바꿉니다.
    characteristic = torch.tensor(characteristic, dtype=torch.float32)
    # 코사인 유사도
    cos_sim = util.pytorch_cos_sim(characteristic, emb)

    #유사도가 가장 비슷한 질문 인덱스를 구합니다.
    best_sim_idx = int(np.argmax(cos_sim))

    # 질문의 유사도와 가장 비슷한  답변 제공
    question = Chatbot_Data['Q'][best_sim_idx]
    answer = Chatbot_Data['A'][best_sim_idx]

    return question, answer

if __name__ == "__main__":
    #question, answer = qachatbot('안녕하세요반가워요')
    #print(question)
    #print(answer)
    pass
    
