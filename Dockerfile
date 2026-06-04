# 1. Lấy nền tảng là Image chính chủ của Ultralytics (Đã cài sẵn Python, PyTorch, CUDA, OpenCV)
FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]