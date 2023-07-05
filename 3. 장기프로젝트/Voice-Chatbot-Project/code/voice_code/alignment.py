import pandas as pd
import os
import re

def get_sorted_file_list(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]  # 파일 이름과 확장자 분리
            match = re.search(r'-(.+)', file_name)
            match = match.group(1)[3:]

            if match:
                number = int(match)
                file_list.append((number, file_name))
    sorted_file_list = [file_path + '.wav' for _, file_path in sorted(file_list)]
    
    return sorted_file_list

def alignment(aud_list, text_list, file_path):
    '''
    Input
    aud_list : audio가 압축해제 되어있는 디렉토리 이름(상대경로)
    text_list : 각 audio에 해당하는 text가 저장되어 있는 데이터(data frame)
    file_path : txt file 위치
    
    Output
    (audio name), 
    '''
    
    for num in range(len(aud_list)):
        aud_path = aud_list[num]
        text_path = text_list[num]
        df = pd.read_excel(text_path, header = None, names = ['text'])
        
        aud_name = get_sorted_file_list(aud_path)
        
        with open(file_path, 'a') as file:
            for i in range(len(aud_name)):
                text = df.iloc[i,0]
                writing = f'{aud_name[i]}|{text}|{len(text)}'
                file.write(writing + '\n')

    
    


    
if __name__ == "__main__":

    aud_path = '../../data/aud_file'
    text_path = '../../data/text_file'
    file_path = '../../data/alignment.txt'
    
    aud_names = ['aud_1', 'aud_2', 'aud_3', 'aud_5', 'aud_6', 'aud_13', 'aud_14', 'aud_43', 'aud_44']
    text_names = ['wang-01.xlsx', 'wang-02.xlsx', 'wang-03.xlsx', 'wang-05.xlsx', 'wang-06.xlsx', 'wang-13.xlsx', 'wang-14.xlsx', 'wang-43.xlsx', 'wang-44.xlsx']
    
    aud_list = []
    text_list = []
    for num in range(len(aud_names)):
        aud_list.append(os.path.join(aud_path, aud_names[num]))
        text_list.append(os.path.join(text_path, text_names[num]))

    alignment(aud_list, text_list, file_path)