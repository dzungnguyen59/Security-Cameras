import cv2
import time
import json
from ultralytics import YOLO

# Import các module nội bộ
from modules.camera import ManagedCamera
from modules.gnn_worker import GlobalGNNWorker
from modules.utils import make_grid 

class TrackingSystem:
    def __init__(self, config_path="config.json"):
        """Khởi tạo toàn bộ hệ thống từ file cấu hình"""
        print("Đang khởi tạo hệ thống...")
        self.load_config(config_path)
        
        # 1. Khởi tạo AI (Yêu cầu model yolo-pose để bắt cổ tay)
        self.yolo = YOLO(self.config["models"]["yolo"])
        self.yolo_weapon = YOLO(self.config["models"]["weapon"])
        
        # 2. Khởi tạo luồng GNN (Worker) - Đây là nơi lưu trữ "Sổ đen" Global
        self.gnn_worker = GlobalGNNWorker()
        self.gnn_worker.start()
        
        # 3. Khởi tạo danh sách Camera
        self.cameras = []
        for cfg in self.config["cameras"]:
            c = ManagedCamera(src=cfg['src'], cam_id=cfg['id']).start()
            self.cameras.append(c)
            
        print(f"Đã mở {len(self.cameras)} camera. Vui lòng đợi...")
        time.sleep(2)

        # 4. Các thông số cài đặt
        self.fps_limit = self.config["settings"]["fps_limit"]
        self.grid_cols = self.config["settings"]["grid_columns"]
        self.time_per_frame = 1.0 / self.fps_limit

        # CẤU HÌNH NHẬN DIỆN VŨ KHÍ
        self.weapon_history = {} # {local_uid_key: [timestamps]}
        self.TIME_WINDOW = 5.0   
        self.MIN_FRAMES = 3      # Đếm đủ 3 frame trong 5s để xác nhận
        self.local_armed_bank = {}  # Sổ đen tạm thời của Camera

    def load_config(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file {path}!")
            exit()

    def run(self):
        prev_time = time.time()
        fps_display = 0
        frame_count = 0

        try:
            while True:
                start_time = time.time()
                frame_count += 1
                active_cams = [c for c in self.cameras if c.is_connected and c.frame is not None]

                if not active_cams:
                    time.sleep(0.1)
                    continue

                imgs = [cam.frame for cam in active_cams]
                results = self.yolo.predict(imgs, conf=0.3, device="0", verbose=False)
                results_weapon = self.yolo_weapon.predict(imgs, conf=0.4, device="0", verbose=False) 

                frames_to_display = []
                data_for_gnn = []

                for i, cam in enumerate(active_cams):
                    try:
                        tracks, track_info = cam.process_tracking([results[i]], cam.frame)
                    except Exception as e:
                        print(f"Lỗi tại process_tracking ({cam.cam_id}): {e}")
                        continue
                    
                    weapon_boxes = results_weapon[i].boxes.data.cpu().numpy()
                    weapon_names = self.yolo_weapon.names
                    current_time = time.time()
                    f = cam.frame.copy()

                    # BƯỚC 1: XÁC ĐỊNH VŨ KHÍ BẰNG YOLO-POSE & VẼ POSE
                    weapon_assignments = {} 
                    wrists = []
                    
                    # Khai báo các cặp điểm nối xương (Skeleton) của chuẩn COCO
                    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), 
                                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), 
                                (1, 3), (2, 4), (3, 5), (4, 6)]

                    if hasattr(results[i], 'keypoints') and results[i].keypoints is not None:
                        raw_kpts = results[i].keypoints.data.cpu().numpy()
                        for kpt in raw_kpts:
                            # --- 1. VẼ CÁC ĐƯỜNG NỐI XƯƠNG LÊN MÀN HÌNH ---
                            for link in skeleton:
                                pt1, pt2 = link
                                if len(kpt) > max(pt1, pt2):
                                    x1, y1, conf1 = kpt[pt1]
                                    x2, y2, conf2 = kpt[pt2]
                                    # Chỉ vẽ nối nếu cả 2 khớp xương đều rõ ràng
                                    if conf1 > 0.2 and conf2 > 0.2:
                                        cv2.line(f, (int(x1), int(y1)), (int(x2), int(y2)), (255, 105, 180), 2) # Xương màu hồng

                            # --- 2. VẼ CÁC ĐIỂM KHỚP LÊN MÀN HÌNH ---
                            for j, pt in enumerate(kpt):
                                if len(pt) >= 3 and pt[2] > 0.2:
                                    # Highlight riêng cổ tay (điểm 9, 10) to hơn và có màu vàng để dễ nhìn
                                    is_wrist = j in [9, 10]
                                    color = (0, 255, 255) if is_wrist else (0, 255, 0)
                                    radius = 6 if is_wrist else 4
                                    cv2.circle(f, (int(pt[0]), int(pt[1])), radius, color, -1)

                            # --- 3. LOGIC XÉT VŨ KHÍ ---
                            if len(kpt) >= 11: 
                                lw, rw = kpt[9], kpt[10] # Điểm 9, 10 là cổ tay
                                if lw[2] > 0.2: wrists.append((lw[0], lw[1]))
                                if rw[2] > 0.5: wrists.append((rw[0], rw[1]))

                    for w_box in weapon_boxes:
                        wx1, wy1, wx2, wy2 = map(int, w_box[:4])
                        w_type = weapon_names[int(w_box[5])]
                        
                        cv2.rectangle(f, (wx1, wy1), (wx2, wy2), (0, 165, 255), 2)
                        cv2.putText(f, w_type.upper(), (wx1, wy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                        holding_wrist = None
                        min_dist = 40 
                        for kx, ky in wrists:
                            dx = max(wx1 - kx, 0, kx - wx2)
                            dy = max(wy1 - ky, 0, ky - wy2)
                            dist = (dx**2 + dy**2)**0.5
                            if dist < min_dist:
                                min_dist, holding_wrist = dist, (kx, ky)
                        
                        if holding_wrist:
                            hx, hy = holding_wrist
                            for t_coord in tracks:
                                tx1, ty1, tx2, ty2, track_id = t_coord[:5]
                                if tx1 <= hx <= tx2 and ty1 <= hy <= ty2:
                                    weapon_assignments[int(track_id)] = w_type
                                    break

                    # BƯỚC 2: LOGIC GÁN TRẠNG THÁI VÀO GLOBAL ID
                    cam_gnn_data = {'cam_id': cam.cam_id, 'tracks': []}
                    
                    for t_coord, t_info in zip(tracks, track_info):
                        x1, y1, x2, y2, track_id = t_coord[:5]
                        local_track_id = int(track_id)
                        global_id = self.gnn_worker.global_id_map.get((cam.cam_id, local_track_id), f"L-{local_track_id}")
                        
                        uid_key = f"{cam.cam_id}_{local_track_id}"
                        if uid_key not in self.weapon_history: 
                            self.weapon_history[uid_key] = []
                        
                        # Dọn dẹp frame cũ
                        self.weapon_history[uid_key] = [t for t in self.weapon_history[uid_key] if current_time - t <= self.TIME_WINDOW]
                        
                        # Cộng điểm nếu tay đang chạm vũ khí
                        cur_w = weapon_assignments.get(local_track_id)
                        if cur_w: 
                            self.weapon_history[uid_key].append(current_time)
                        
                        # --- GHI VÀO SỔ ĐEN LOCAL VĨNH VIỄN ---
                        if len(self.weapon_history[uid_key]) >= self.MIN_FRAMES:
                            if cur_w: # Lưu tên vũ khí vào trí nhớ của Camera
                                self.local_armed_bank[uid_key] = cur_w

                        # Truyền SỔ ĐEN LOCAL lên GNN 
                        cam_gnn_data['tracks'].append({
                            'id': t_info['id'], 
                            'feature': t_info['feature'],
                            'weapon_type': self.local_armed_bank.get(uid_key) 
                        })

                        # --- XÁC ĐỊNH TRÍ NHỚ ĐỂ VẼ BBOX ---
                        final_weapon = None
                        
                        if isinstance(global_id, int):
                            final_weapon = self.gnn_worker.is_armed(global_id)
                        
                        if final_weapon is None:
                            final_weapon = self.local_armed_bank.get(uid_key)

                        # --- ĐỔI MÀU BBOX ---
                        is_armed = final_weapon is not None
                        color = (0, 0, 255) if is_armed else (0, 255, 0)
                        label = f"ID: {global_id}"
                        if is_armed:
                            w_name = str(final_weapon).upper() if isinstance(final_weapon, str) else "WEAPON"
                            label += f" [ARMED: {w_name}]"

                        cv2.rectangle(f, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(f, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    frames_to_display.append(f)
                    data_for_gnn.append(cam_gnn_data)
                
                # Đồng bộ với GNN mỗi 20 frame
                if frame_count % 20 == 0:
                    self.gnn_worker.update_features(data_for_gnn)

                grid_view = make_grid(frames_to_display, cols=self.grid_cols)
                if grid_view is not None:
                    curr_t = time.time()
                    fps_display = (fps_display * 0.9) + ((1.0 / (curr_t - prev_time)) * 0.1) if (curr_t - prev_time) > 0 else 0
                    prev_time = curr_t
                    cv2.rectangle(grid_view, (0, 0), (180, 40), (0, 0, 0), -1)
                    cv2.putText(grid_view, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Robbery Detection System", grid_view)

                if cv2.waitKey(1) & 0xFF == ord('q'): break

                elapsed = time.time() - start_time
                sleep_amount = max(1, int((self.time_per_frame - elapsed) * 1000))
                if cv2.waitKey(sleep_amount) & 0xFF == ord('q'): 
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        for cam in self.cameras: cam.stop()
        cv2.destroyAllWindows()