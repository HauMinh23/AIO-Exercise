# Dự Án Streamlit

## Mục Lục
- [Giới Thiệu](#giới-thiệu)
- [Yêu Cầu](#yêu-cầu)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Chức Năng](#chức-năng)
  - [Chatbot với Huggingface](#1-chatbot-với-huggingface)
  - [Tính Khoảng Cách Levenshtein](#2-tính-khoảng-cách-levenshtein)
  - [Hiển Thị Giao Diện Người Dùng Đơn Giản](#3-hiển-thị-giao-diện-người-dùng-đơn-giản)
  - [Phát Hiện Đối Tượng Trong Hình Ảnh](#4-phát-hiện-đối-tượng-trong-hình-ảnh)
- [Đóng Góp](#đóng-góp)
- [Giấy Phép](#giấy-phép)

## Giới Thiệu

Dự án này bao gồm nhiều ứng dụng nhỏ sử dụng Streamlit, từ chatbot đến xử lý hình ảnh. Đây là một dự án demo nhằm giới thiệu các khả năng của Streamlit trong việc phát triển các ứng dụng web tương tác.

## Yêu Cầu

- Python 3.7 trở lên
- Các thư viện Python cần thiết:
  - streamlit
  - transformers
  - bitsandbytes
  - accelerate
  - huggingface_hub
  - opencv-python
  - numpy
  - Pillow

## Cài Đặt

1. **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install streamlit transformers bitsandbytes accelerate huggingface_hub opencv-python numpy Pillow
    ```

## Sử Dụng

1. **Chạy ứng dụng Streamlit:**
    ```bash
    streamlit run app.py
    ```

## Chức Năng

### 1. Chatbot với Huggingface

Ứng dụng chatbot sử dụng tài khoản Huggingface để thực hiện các cuộc hội thoại thông qua mô hình ngôn ngữ lớn.

```python
import streamlit as st
from hugchat import hugchat
from hugchat.login import Login

st.title('Chatbox')

with st.sidebar:
    st.title('Huggingface Account')
    hf_email = st.text_input('E-mail')
    hf_pass = st.text_input('Password', type='password')

if 'messages' not in st.session_state.keys():
    st.session_state.messages = [
        {'role': 'assistant', 'content': 'How may I help you'}]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

def generate_response(promt_input, email, passwd):
    sign = Login(email, passwd)
    cookies = sign.login()
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(promt_input)

if promt := st.chat_input(disabled=not (hf_email and hf_pass)):
    st.session_state.messages.append({'role': 'user', 'content': promt})
    with st.chat_message('user'):
        st.write(promt)

if st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner("Thinking ..."):
            response = generate_response(promt, hf_email, hf_pass)
            st.write(response)
        message = {'role': 'assistant', 'content': response}
        st.session_state.messages.append(message)

### 2. Tính Khoảng Cách Levenshtein

Ứng dụng này cho phép người dùng nhập một từ và tìm từ gần đúng nhất trong từ điển dựa trên khoảng cách Levenshtein.
import streamlit as st

def levenshtein_distance(token1, token2):
    if not token1 hoặc not token2:
        return len(token1) + len(token2)
    distances = [[0] * (len(token2) + 1) for _ in range(len(token1) + 1)]
    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1
    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            distances[t1][t2] = min(
                distances[t1 - 1][t2] + 1,
                distances[t1][t2 - 1] + 1,
                distances[t1 - 1][t2 - 1] + (token1[t1 - 1] != token2[t2 - 1])
            )
    return distances[-1][-1]

def load_vocab(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    words = sorted(set([line.strip().lower() for line in lines]))
    return words

vocabs = load_vocab(file_path='./data/vocab.txt')

def main():
    st.title("Word Correction using Levenshtein Distance")
    word = st.text_input('Word:')
    if st.button("Compute"):
        leven_distances = dict()
        for vocab in vocabs:
            leven_distances[vocab] = levenshtein_distance(word, vocab)
        sorted_distences = dict(sorted(leven_distances.items(), key=lambda item: item[1]))
        correct_word = list(sorted_distences.keys())[0]
        st.write('Correct word: ', correct_word)
        col1, col2 = st.columns(2)
        col1.write('Vocabulary:')
        col1.write(vocabs)
        col2.write('Distances:')
        col2.write(sorted_distences)

if __name__ == "__main__":
    main()

### 3. Hiển Thị Giao Diện Người Dùng Đơn Giản

Ứng dụng này hiển thị các thành phần giao diện cơ bản của Streamlit như tiêu đề, các loại markdown, và các loại input khác nhau.

import streamlit as st

st.title("MY PROJECT")
st.header("This is a header")
st.subheader("This is a subheader")
st.caption("This is a caption")
st.text("I love AI VIET NAM")

st.divider()

st.markdown("# Heading 1")
st.markdown("[AI VIET NAM](https://aivietnam.edu.vn/)")
st.markdown("""
        1. Machine Learning
        2. Deep Learning""")
st.markdown(r"$\sqrt{2x+2}$")

st.divider()

st.write('I love AI VIET NAM')
st.write('## Heading 2')
st.write(r'$ \sqrt{2x+2} $')

def get_user_name():
    return 'Thai'

with st.echo():
    st.write('This code will be printed')
    def get_email():
        return 'thai@gmail.com'
    user_name = get_user_name()
    email = get_email()
    st.write(user_name, email)

st.divider()

def get_name():
    st.write("Thai")

agree = st.checkbox("I agree", on_change=get_name)
if agree:
    st.write("Great!")

st.radio(
    "Your favorite color:",
    ['Yellow', 'Bleu'],
    captions=['Vàng', 'Xanh']
)

option = st.selectbox(
    "Your contact:",
    ("Email", "Home phone", "Mobile phone"))

st.write("Selected:", option)

options = st.multiselect(
    "Your favorite colors:",
    ["Green", "Yellow", "Red", "Blue"],
    ["Yellow", "Red"])

st.write("You selected:", options)

color = st.select_slider(
    "Your favorite color:",
    options=["red", "orange", "violet"])
st.write("My favorite color is", color)

st.divider()

if st.button("Say hello"):
    st.write("Hello")
else:
    st.write("Goodbye")

st.link_button(
    "Go to Google",
    "https://www.google.com.vn/")

st.divider()
title = st.text_input(
    "Movie title:", "Life of Brian"
)
st.write("The current movie title is", title)

messages = st.container(height=200)
if prompt := st.chat_input("Say something"):
    messages.chat_message("user").write(prompt)
    messages.chat_message("assistant").write(f"Echo: {prompt}")

st.divider()

uploaded_files = st.file_uploader(
    "Choose files", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)

st.divider()

number = st.number_input("Insert a number")
st.write("The current number is ", number)

values = st.slider(
    "Select a range of values",
    0.0, 100.0, (25.0, 75.0))
st.write("Values:", values)

st.divider()
with st.form("my_form"):
    col1, col2 = st.columns(2)
    f_name = col1.text_input('First Name')
    l_name = col2.text_input('Last Name')
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("First Name: ", f_name, " - Last Name:", l_name)

### 4. Phát Hiện Đối Tượng Trong Hình Ảnh

Ứng dụng này tải lên hình ảnh và sử dụng mô hình MobileNetSSD để phát hiện đối tượng trong hình ảnh.

import cv2
import numpy as np
from PIL import Image
import streamlit as st

MODEL = "D:\\AIO2024\\Github\\project_module1\\streamlit\\MobileNetSSD_deploy.caffemodel"
PROTOTXT = "D:\\AIO2024\\Github\\project_module1\\streamlit\\MobileNetSSD_deploy.prototxt.txt"

def process_image(image):
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    image = np.array(image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return image

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

st.title("Object Detection using MobileNetSSD")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting objects...")
    processed_image = process_image(image)
    st.image(processed_image, caption='Processed Image.', use_column_width=True)
## Đóng Góp
Chúng tôi hoan nghênh mọi sự đóng góp. Vui lòng tạo một pull request hoặc liên hệ với chúng tôi qua email.
## Giấy Phép
Dự án này được cấp phép theo MIT License.

Để sử dụng file này, bạn chỉ cần lưu nội dung trên vào một file có tên `README.md` trong thư mục gốc của dự án Streamlit của bạn. Nội dung này sẽ cung cấp hướng dẫn chi tiết về cách cài đặt và sử dụng các chức năng chính của dự án.
