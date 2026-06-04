# Hệ Thống Nhận Diện Vũ Khí & Theo Dõi Đa Camera (GNN Tracking System)

Hệ thống giám sát an ninh thông minh xử lý luồng video từ nhiều camera cùng lúc. Bằng việc kết hợp nhận diện vũ khí (YOLO), phân tích hành vi cơ thể (YOLO-Pose) và Mạng nơ-ron đồ thị (GNN - Graph Neural Network), hệ thống có khả năng theo dõi đối tượng tình nghi liên tục xuyên suốt các góc máy khác nhau.

---

## Tính Năng Chi Tiết

Hệ thống không chỉ dừng lại ở việc "thấy vũ khí là báo động", mà áp dụng một chuỗi logic phức tạp để tránh báo động giả và bám sát đối tượng:

1. **Phân tích hành vi cầm nắm (Pose-Weapon Intersection):**
   * Sử dụng `YOLO-Pose` để nội suy khung xương, xác định chính xác vị trí 2 cổ tay của đối tượng.
   * Tính toán khoảng cách không gian giữa cổ tay và Bounding Box của vũ khí (`weapon-detect`). Hệ thống chỉ ghi nhận "có vũ khí" khi phát hiện đối tượng thực sự ĐANG CẦM vũ khí đó.
2. **Cơ chế xác thực thời gian thực (Time-Window Validation):**
   * Tích hợp bộ đếm khung hình: Đối tượng phải cầm vũ khí ít nhất `3 frames` trong vòng `5 giây` thì hệ thống mới chính thức gắn cờ báo động (Armed). Giúp loại bỏ hoàn toàn nhiễu do AI nhận diện nhầm trong tích tắc.
3. **Đồng bộ Sổ đen Toàn cục (Global GNN Blacklist):**
   * Mỗi camera duy trì một bộ nhớ tạm (Local Armed Bank).
   * Cứ mỗi 20 frames, các camera sẽ gửi đặc trưng nhận dạng (Features) của kẻ tình nghi lên `GNN Worker`. 
   * GNN Worker đóng vai trò là "Bộ não trung tâm", tính toán độ tương đồng đồ thị để hợp nhất ID. Nếu Camera A thấy kẻ gian giấu súng đi và bước sang Camera B, Camera B lập tức nhận diện được ID này và giữ nguyên viền đỏ cảnh báo (Armed) dù không hề thấy vũ khí.
4. **Tối ưu hóa đa luồng (Multi-threading & FPS Limit):**
   * Các luồng RTSP/Webcam được quản lý độc lập bằng Threading, tránh hiện tượng nghẽn cổ chai (bottleneck) khi một camera bị mất kết nối.

---

## Yêu Cầu Hệ Thống

Dự án được tối ưu để chạy trong môi trường **Docker** kết hợp với **GPU NVIDIA**. 

* **Hệ điều hành:** Windows 10/11 (đã cài đặt WSL2 - Ubuntu) hoặc Linux thuần.
* **Phần cứng:** Card đồ họa NVIDIA (Khuyến nghị VRAM từ 6GB trở lên).
* **Phần mềm cần thiết:**
  * NVIDIA Driver mới nhất (Chỉ cần cài trên Windows, **KHÔNG** cài CUDA trong WSL).
  * [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) (Bật tính năng WSL Integration trong Settings).

---

## Hướng Dẫn Cài Đặt Và Chạy Trực Tiếp (Local/WSL)

1. **Clone mã nguồn:**
   ```bash
   git clone <link-github-cua-ban>
   cd <ten-thu-muc-du-an>

2. **Tùy chỉnh config.json:** 
  Mở file cấu hình và cập nhật luồng camera (IP/RTSP hoặc Webcam local)

### Chạy bằng Docker
 **Bước 1.Tải xhost để hiển thị giao diện**
  ```
  sudo apt update
  sudo apt install x11-xserver-utils
  ```
**Bước 2. Build Image**
  ```
  docker build -t <name> .
  ```
  Thay thế name bằng tên mà bạn muốn
  
**Bước 3. Khởi chạy Container**
  ```
  docker run -it --rm \
  --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/wslg:/mnt/wslg \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e PULSE_SERVER=$PULSE_SERVER \
  -v $(pwd):/app \
  <name> python main.py
  ```
