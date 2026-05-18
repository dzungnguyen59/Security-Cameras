import cv2
import threading
import time
import numpy as np
from pathlib import Path
from boxmot import DeepOcSort

class Camera:
    def __init__(self, src, cam_id):
        self.src = src # Nguồn video
        self.cam_id = cam_id # ID camera
        self.cap = None 
        self.frame = None # Lưu frame mới nhất đã qua xử lý 
        self.is_connected = False # Trạng thái kết nối
        self.stopped = False # Trạng thái dừng luồng

        self.DISPLAY_W = 640
        self.DISPLAY_H = 480
    
    def start(self):
        # Bắt đầu luồng đọc khung hình
        # deamon=True để luồng tự động dừng khi chương trình chính kết thúc
        threading.Thread(target=self._reader, args=(), daemon=True).start()
        return self
    
    def resize_with_aspect_ratio(self, frame):
        """Thu nhỏ ảnh giữ tỉ lệ và chèn vào khung đen (Padding)"""
        h, w = frame.shape[:2]
        
        # Tính toán tỉ lệ scale để ảnh nằm vừa trong khung DISPLAY_W x DISPLAY_H
        scale = min(self.DISPLAY_W / w, self.DISPLAY_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize ảnh gốc theo tỉ lệ mới
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Tạo một "canvas" đen đúng kích thước tiêu chuẩn
        canvas = np.zeros((self.DISPLAY_H, self.DISPLAY_W, 3), dtype=np.uint8)
        
        # Tính toán vị trí để đặt ảnh vào giữa canvas
        x_offset = (self.DISPLAY_W - new_w) // 2
        y_offset = (self.DISPLAY_H - new_h) // 2
        
        # Dán ảnh đã resize vào giữa canvas đen
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

    def _reader(self):
        """Luồng này chỉ làm nhiệm vụ duy nhất: Đọc và dọn dẹp Buffer"""

        print(f"[{self.cam_id}] Đã khởi chạy luồng đọc dữ liệu.", flush=True)
        error_count = 0
        MAX_ERRORS = 10  # Nếu lỗi liên tiếp 10 lần (~5 giây), ép reset kết nối

        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.is_connected = False
                    print(f"[{self.cam_id}] Đang cố gắng kết nối tới mạng...", flush=True)

                    # Đảm bảo giải phóng hoàn toàn bộ nhớ của kết nối cũ bị lỗi
                    if self.cap is not None:
                        self.cap.release()

                    self.cap = cv2.VideoCapture(self.src)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Ép buffer về 1

                    if not self.cap.isOpened():
                        print(f"[{self.cam_id}] Kết nối thất bại. Sẽ thử lại sau 5 giây...", flush=True)
                        time.sleep(5)
                        continue
                    else:
                        print(f"[{self.cam_id}] Kết nối THÀNH CÔNG!", flush=True)
                        error_count = 0  # Reset bộ đếm lỗi khi kết nối lại thành công

            # grab() giúp bỏ qua việc decode các frame cũ, nhảy thẳng tới frame mới nhất
            # Chúng ta lặp grab() cho đến khi buffer trống
                ret = self.cap.grab()
                if not ret:
                    error_count += 1
                    self.is_connected = False
                    print(f"[{self.cam_id}] Mất tín hiệu hình ảnh (Lỗi {error_count}/{MAX_ERRORS})", flush=True)
                    time.sleep(1)

                    # Nếu OpenCV bị "kẹt" luồng mạng, nó không tự nhận biết là mất mạng.
                    # Ta phải dùng thủ thuật: Lỗi quá 10 lần -> Ép xóa kết nối để vòng lặp sau tự tạo lại từ đầu.
                    if error_count >= MAX_ERRORS:
                        print(f"[{self.cam_id}] Quá thời gian chờ tín hiệu. Đang ép Reset Camera...", flush=True)
                        self.cap.release()
                        self.cap = None

                    continue

                # Chỉ retrieve (giải mã) khi cần thiết (ví dụ: mỗi khi có frame mới)
                # Ở đây ta retrieve luôn để biến self.frame luôn sẵn sàng
                ret, raw_frame = self.cap.retrieve()
                if ret:
                    # Resize ngay tại luồng đọc để luồng Main nhẹ hơn
                    self.frame = self.resize_with_aspect_ratio(raw_frame)
                    self.is_connected = True
                    error_count = 0  # Lấy được ảnh bình thường -> Trả bộ đếm lỗi về 0
                else:
                    error_count += 1
                
                # Không dùng time.sleep() ở đây hoặc sleep cực ngắn (0.001) 
                # để đảm bảo tốc độ đọc luôn nhanh hơn tốc độ camera gửi về
                time.sleep(0.01)

            except Exception as e:
                # 4. BẮT TẤT CẢ CÁC LỖI BẤT NGỜ (TRÁNH CHẾT LUỒNG)
                print(f"[{self.cam_id}] Gặp lỗi ngoại lệ nghiêm trọng: {e}", flush=True)
                self.is_connected = False
                if getattr(self, 'cap', None) is not None:
                    self.cap.release()
                    self.cap = None
                time.sleep(5) # Nghỉ ngơi hệ thống một chút trước khi thử kết nối lại

    def stop(self):
        self.stopped = True
        time.sleep(0.5) # Đợi luồng update dừng hẳn
        # Kiểm tra xem self.cap có tồn tại (không phải None) thì mới giải phóng
        if self.cap is not None:
            self.cap.release()
            print(f"Camera {self.cam_id} đã được giải phóng.")

class ManagedCamera(Camera): # Kế thừa từ class Camera cũ của bạn
    def __init__(self, src, cam_id):
        super().__init__(src, cam_id)
        # CẤP ĐỘ 1: Local Tracker cho từng camera
        self.tracker = DeepOcSort(
            reid_weights=Path('weights/osnet_x0_25_msmt17.pt'), 
            device='cuda:0',
            half=False,                                         
            max_age=30
        )
        self.local_tracks = []
        

    def process_tracking(self, yolo_results, frame):
        dets = yolo_results[0].boxes.data.cpu().numpy()
        dets = dets[dets[:, 5] == 0]  # Chỉ lấy class 0 (person)
        self.local_tracks = self.tracker.update(dets, frame)
        
        track_info = []
        for t in self.local_tracks:
            tid = int(t[4])
            feat = None
            
            # Truy cập trực tiếp vào active_tracks (đã được xác nhận có tồn tại)
            for track in self.tracker.active_tracks:
                if track.id == tid:
                    # Tùy bản BoxMOT, feature có thể nằm ở các biến khác nhau
                    if hasattr(track, 'features') and len(track.features) > 0:
                        feat = track.features[-1]
                    elif hasattr(track, 'curr_feat'):
                        feat = track.curr_feat
                    elif hasattr(track, 'emb'):
                        feat = track.emb
                    break
            
            track_info.append({'id': tid, 'feature': feat})
            
        return self.local_tracks, track_info
