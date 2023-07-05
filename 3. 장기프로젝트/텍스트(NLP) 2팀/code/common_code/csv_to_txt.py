import pandas as pd

# CSV 파일 경로와 이름 지정
excel_load = 'data/split/split_text_'

# 텍스트 파일 경로와 이름 지정
txt_load = 'data/text/txt_'

# CSV 파일 열기
for i in range(1,47):
    # Excel 파일 불러오기
    num = str(i).zfill(2)
    excel_name = excel_load + num + '.xlsx'
    txt_name = txt_load + num + '.txt'
    excel_file = pd.read_excel(excel_name)

    # 특정 열(column) 선택
    column_data = excel_file['text']

    # 텍스트 파일에 데이터 저장
    with open(txt_name, 'w') as file:
        for data in column_data:
            file.write(str(data).strip() + '\n')