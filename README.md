# Dự Án Phát Hiện Mũ Bảo Hiểm với YOLOv10

## Mục Lục
- [Giới Thiệu](#giới-thiệu)
- [Yêu Cầu](#yêu-cầu)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Đóng Góp](#đóng-góp)
- [Giấy Phép](#giấy-phép)
- [Lời Cảm Ơn](#lời-cảm-ơn)

## Giới Thiệu

Dự án này sử dụng YOLOv10 để phát hiện mũ bảo hiểm trong hình ảnh và video. YOLOv10 là một trong những mô hình học sâu mạnh mẽ nhất cho các nhiệm vụ phát hiện đối tượng thời gian thực. Dự án này sẽ giúp phát hiện mũ bảo hiểm để đảm bảo tuân thủ an toàn trong các môi trường như công trường xây dựng, nhà máy và giám sát giao thông.

## Yêu Cầu

- Python 3.7 trở lên
- Các thư viện Python cần thiết: được liệt kê trong `requirements.txt`
- Google Colab hoặc môi trường tương tự để chạy mã

## Cài Đặt

1. **Clone kho lưu trữ:**
    ```bash
    git clone https://github.com/THU-MIG/yolov10.git
    ```

2. **Chuyển đến thư mục yolov10:**
    ```bash
    cd yolov10
    ```

3. **Cài đặt các thư viện yêu cầu:**
    ```bash
    pip install -q -r requirements.txt
    pip install -e .
    ```

4. **Tải mô hình YOLOv10 đã được huấn luyện:**
    ```bash
    wget https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10n.pt
    gdown 1twdtZEfcw4ghSZIiPDypJurZnNXzMO7R
    ```

5. **Tạo thư mục và giải nén bộ dữ liệu mũ bảo hiểm:**
    ```bash
    mkdir safety_helmet_dataset
    unzip -q '/content/yolov10/Safety_Helmet_Dataset.zip' -d '/content/yolov10/safety_helmet_dataset'
    ```

## Sử Dụng

1. **Huấn luyện mô hình:**
    ```python
    from ultralytics import YOLOv10

    MODEL_PATH = 'yolov10n.pt'
    model = YOLOv10(MODEL_PATH)

    YAML_PATH = '/content/yolov10/safety_helmet_dataset/data.yaml'
    EPOCHS = 30
    IMG_SIZE = 320
    BATCH_SIZE = 32

    model.train(data=YAML_PATH, epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH_SIZE)
    ```

2. **Dự đoán trên bộ dữ liệu kiểm tra:**
    ```python
    TRAINED_MODEL_PATH = '/content/yolov10/runs/detect/train/weights/best.pt'
    model = YOLOv10(TRAINED_MODEL_PATH)

    results = model.predict(data=YAML_PATH, imgsz=IMG_SIZE, split='test')
    ```

3. **Dự đoán trên hình ảnh cụ thể:**
    ```python
    from google.colab.patches import cv2_imshow

    IMAGE_URL = 'https://nonbaohiemthangloi.com.vn/uploads/news/11_2018/non-bao-hiem-thang-loi-021.jpeg'
    CONF_THRESHOLD = 0.5
    results = model.predict(source=IMAGE_URL, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
    annotated_img = results[0].plot()

    cv2_imshow(annotated_img)
    ```

## Đóng Góp

Chúng tôi hoan nghênh sự đóng góp từ cộng đồng. Để đóng góp cho dự án này, hãy làm theo các bước sau:

1. Fork kho lưu trữ.
2. Tạo một nhánh mới (`git checkout -b feature-branch`).
3. Thực hiện các thay đổi của bạn.
4. Commit các thay đổi (`git commit -m 'Thêm tính năng mới'`).
5. Đẩy lên nhánh (`git push origin feature-branch`).
6. Mở một pull request.

## Giấy Phép

Dự án này được cấp phép theo Giấy Phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## Lời Cảm Ơn

- Cảm ơn đặc biệt đến cộng đồng mã nguồn mở đã cung cấp các công cụ và tài nguyên giúp dự án này trở nên khả thi.
- Gửi lời cảm ơn đến các tập dữ liệu và mô hình đã được huấn luyện trước đã sử dụng.

-------------
# Dự Án Mô Hình Ngôn Ngữ Lớn (LLM)

## Mục Lục
- [Giới Thiệu](#giới-thiệu)
- [Yêu Cầu](#yêu-cầu)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Đóng Góp](#đóng-góp)
- [Giấy Phép](#giấy-phép)
- [Lời Cảm Ơn](#lời-cảm-ơn)

## Giới Thiệu

Dự án này sử dụng mô hình ngôn ngữ lớn (LLM) để thực hiện các nhiệm vụ sinh văn bản và phân loại cảm xúc. Chúng tôi sử dụng mô hình Vicuna-7B-v1.5 từ Lmsys kết hợp với cấu hình BitsAndBytes để tối ưu hóa việc sử dụng bộ nhớ và tăng cường hiệu suất tính toán.

## Yêu Cầu

- Python 3.7 trở lên
- Các thư viện Python cần thiết: `transformers`, `bitsandbytes`, `accelerate`
- Hugging Face token để truy cập các mô hình

## Cài Đặt

1. **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -q transformers==4.41.2
    pip install -q bitsandbytes==0.43.1
    pip install -q accelerate==0.31.0
    ```

2. **Cài đặt thư viện Hugging Face Hub:**
    ```bash
    pip install huggingface_hub
    ```

3. **Đăng nhập vào Hugging Face Hub:**
    ```python
    from huggingface_hub import login
    login(token="YOUR_TOKEN")
    ```

## Sử Dụng

1. **Cấu hình mô hình và tải mô hình:**
    ```python
    import torch
    from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

    MODEL_NAME = 'lmsys/vicuna-7b-v1.5'
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True,
        use_auth_token="YOUR_TOKEN"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token="YOUR_TOKEN")

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )
    ```

2. **Sinh văn bản:**
    ```python
    prompt = "Hello, who are you?"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(**model_inputs)[0]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(answer.split('\n\n')[1])
    ```

3. **Phân loại cảm xúc:**
    ```python
    prompt = """
    ### A chat between a human and an assistant.

    ### Human:
    Your task is to classify the sentiment of input text into one of two categories: neutral or negative. Here is an example:

    Input: What do you do?
    Output: student

    Now, let's practice:
    Input: What are you doing?
    Output:
    ### Assistant:
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(**model_inputs)[0]
    answer = tokenizer.decode(generated_ids,
                              temperature=1.2,
                              do_sample=True,
                              skip_special_tokens=True)

    print(answer.split('Assistant:\n')[1])
    ```

## Đóng Góp

Chúng tôi hoan nghênh sự đóng góp từ cộng đồng. Để đóng góp cho dự án này, hãy làm theo các bước sau:

1. Fork kho lưu trữ.
2. Tạo một nhánh mới (`git checkout -b feature-branch`).
3. Thực hiện các thay đổi của bạn.
4. Commit các thay đổi (`git commit -m 'Thêm tính năng mới'`).
5. Đẩy lên nhánh (`git push origin feature-branch`).
6. Mở một pull request.

## Giấy Phép

Dự án này được cấp phép theo Giấy Phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## Lời Cảm Ơn

- Cảm ơn đặc biệt đến cộng đồng mã nguồn mở đã cung cấp các công cụ và tài nguyên giúp dự án này trở nên khả thi.
- Gửi lời cảm ơn đến Hugging Face và các nhà phát triển mô hình đã cung cấp các mô hình và tài liệu hữu ích.
