Chatbot:

Chatbot là một ứng dụng Streamlit tương tác với Chatbot của Hugging Face. Người dùng có thể nhập thông tin đăng nhập tài khoản Hugging Face vào thanh bên để bắt đầu.

Cách sử dụng:

Nhập email và mật khẩu tài khoản Hugging Face của bạn vào thanh bên.
Bắt đầu trò chuyện với chatbot trong giao diện chính.

Cách hoạt động:

Ứng dụng khởi tạo với thông điệp trợ lý ban đầu: “Làm ơn cho tôi biết bạn cần giúp đỡ về điều gì?”
Tin nhắn của người dùng được lưu trữ trong trạng thái phiên.
Hàm generate_response đăng nhập vào Hugging Face bằng thông tin đăng nhập được cung cấp, tạo một phiên bản chatbot và trò chuyện với chatbot.
Nếu người dùng nhập tin nhắn mới, nó sẽ được thêm vào trạng thái phiên.
Nếu tin nhắn cuối cùng không phải từ trợ lý, bạn sẽ tạo phản hồi bằng cách sử dụng hàm generate_response.

---
Word Correction using Levenshtein Distance:

Ứng dụng này sử dụng thuật toán khoảng cách Levenshtein để sửa lỗi chính tả trong từ vựng. Bạn nhập một từ và ứng dụng sẽ gợi ý từ gần đúng nhất.

Cách sử dụng:

Nhập từ cần kiểm tra vào ô “Word”.
Nhấn nút “Compute” để tính toán.

Cách hoạt động:

Ứng dụng tính khoảng cách Levenshtein giữa từ bạn nhập và từ trong từ điển.
Từ gần đúng nhất sẽ được hiển thị.

Yêu cầu:

Python 3.6+
Streamlit

---

My Project:

Đây là một ứng dụng Streamlit với các tính năng sau:

Hiển thị tiêu đề và phụ đề.
Hiển thị các đoạn văn bản và công thức toán học.
Hiển thị danh sách và liên kết.
Cho phép người dùng nhập dữ liệu và tương tác với các phần tử.

Cách sử dụng:

Cài đặt các thư viện cần thiết:
pip install streamlit

Chạy ứng dụng:
streamlit run app.py

Chức năng:

Hiển thị văn bản,
Tiêu đề: “This is a header”,
Phụ đề: “This is a subheader”,
Chú thích: “This is a caption”,
Văn bản: “I love AI VIET NAM”,
Hiển thị danh sách và liên kết
Tiêu đề cấp 1: “Heading 1”
Liên kết đến trang web AI VIET NAM: AI VIET NAM
Danh sách:
Machine Learning
Deep Learning
Hiển thị công thức toán học
Công thức căn bậc hai: 2x+2​
Tương tác với người dùng
Checkbox: “I agree”
Radio button: “Your favorite color”
Dropdown: “Your contact”
Multiselect: “Your favorite colors”
Slider: “Your favorite color”
Button: “Say hello”
Liên kết nút: “Go to Google”
Trường văn bản: “Movie title”
Trò chuyện: “Say something”
Tải lên tệp: “Choose files”
Trường số: “Insert a number”
Dải giá trị: “Select a range of values”
Biểu mẫu: “First Name” và “Last Name”

---

Nhận diện đối tượng trong ảnh:

Ứng dụng Streamlit này thực hiện việc nhận diện đối tượng trên các hình ảnh được tải lên bằng mô hình MobileNet SSD (Single Shot MultiBox Detector). Nó sẽ phát hiện các đối tượng trong ảnh và vẽ khung giới hạn xung quanh chúng.

Cách sử dụng:

Cài đặt các thư viện cần thiết:

pip install streamlit

Chạy ứng dụng:

streamlit run app.py

Tính năng:

Tải lên một hình ảnh.
Xem hình ảnh đã tải lên.
Xử lý ảnh để nhận diện đối tượng và hiển thị khung giới hạn.

Yêu cầu:

Python 3.6+
Streamlit
