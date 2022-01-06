from pre import preprocess
from lstm import predict as LSTM_predict
from SVM import predict as SVM_predict
from DT import predict as DT_predict
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

def main():
    col1, col2 = st.columns([6, 4])

    col1.title('Phân loại tin tức giả\n')
    col1.text("Demo đồ án NMKHDL phân loại các tin tức giả sử dụng các model học máy.")
    col1.text("MSSV các thành viên: 19120318, 18120027, 18120520, 19120387, 19120470")
    col1.text("")
    col1.subheader("Nhập tin tức cần kiểm tra")
    p = col1.text_area("Input:")
    
    col2.image("fake-news.png")
    col2.subheader("Chọn model đi nào")
    modelOptions = ("LSTM", "SVM", "Decision Tree")
    model = col2.selectbox("Select:", modelOptions)

    result = col2.button("Kiểm tra")

    if result:
        if not p.strip():
            col2.error("Vui lòng nhập tin tức cần kiểm tra !")
        else:
            text_df = preprocess(p)
            if model == modelOptions[0]:
                pred = LSTM_predict(text_df)[0][0]
            elif model == modelOptions[1]:
                pred = SVM_predict(text_df)[0]
            else:
                pred = DT_predict(text_df)[0]

            if pred > 0.5:
                col2.warning("Đây là tin tức giả.")
            else:
                col2.info("Đây là tin tức thật.")
    

        
if __name__ == '__main__':
    main()



