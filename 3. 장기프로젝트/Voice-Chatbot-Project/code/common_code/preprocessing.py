import pandas as pd
import os

data_dir = 'data'


def text_pp(text):
    text.strip()
    text = text.replace('\n', '')
    return text


def list_pp(data):
    data = list(filter(None, result))
    for i, text in enumerate(data):
        data[i] = text_pp(text)
    return data


def text_path(num):
    #자리수 맞추는 함수
    num = str(num).zfill(2)
    return 'text_'+ num + '.xlsx'




if __name__ == "__main__":
    for i in range(1,47):
        file_name = text_path(i)
        load_path = os.path.join(data_dir, file_name)
        df = pd.read_excel(load_path)
        
        result = []
        
        for data in df.iloc[:, -1]:
            try:
                result = result + data.split('.')
            except:
                pass
        
        result = list_pp(result)

        index = [str(i+1) for i in range(len(result))]
        init_time = ['0' for i in range(len(result))]
        data = pd.DataFrame({'text_index' : index, 'text' : result, 'start_time' : init_time, 'end_time' : init_time})
        
        result_name = 'split' + '_' + file_name
        result_path = os.path.join(data_dir, 'split')

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        save_path = os.path.join(data_dir, 'split', result_name)

        data.to_excel(save_path)