# app.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from time import sleep


# 페이지 기본 설정
st.set_page_config(
    page_icon="🐶",
    page_title="임청수의 스트림릿",
    layout="wide",
)

# # # 로딩바 구현하기
# # with st.spinner(text="페이지 로딩중..."):
# #     sleep(2)

# # 페이지 헤더, 서브헤더 제목 설정
# st.header("임청수 페이지에 오신걸 환영합니다👋")
# st.subheader("스트림릿 기능 맛보기")

# # 페이지 컬럼 분할(예: 부트스트랩 컬럼, 그리드)
# cols = st.columns((1, 1, 2))
# cols[0].metric("10/11", "15 °C", "2")
# cols[0].metric("10/12", "17 °C", "2 °F")
# cols[0].metric("10/13", "15 °C", "2")
# cols[1].metric("10/14", "17 °C", "2 °F")
# cols[1].metric("10/15", "14 °C", "-3 °F")
# cols[1].metric("10/16", "13 °C", "-1 °F")

# # 라인 그래프 데이터 생성(with. Pandas)
# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])

# # 컬럼 나머지 부분에 라인차트 생성
# cols[2].line_chart(chart_data)

import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

stt_button = Button(label="Speak", width=100)

stt_button.js_on_event("button_click", CustomJS(code="""
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
 
    recognition.onresult = function (e) {
        var value = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
            }
        }
        if ( value != "") {
            document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
        }
    }
    recognition.start();
    """))

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)


if result:
    if "GET_TEXT" in result:
        st.write(result.get("GET_TEXT"))
        input_text = result.get("GET_TEXT")
        #print(input_text)
