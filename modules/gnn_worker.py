import itertools
import threading
import queue
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Luồng xử lý GNN để gán ID toàn cục cho các đối tượng được theo dõi
class GlobalGNNWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.task_queue = queue.Queue(maxsize=1)
        self.feature_bank = {} 
        self.next_global_id = 1 
        self.lock = threading.Lock() 
        self.global_id_map = {} 
        
        # Bộ nhớ lưu trạng thái có vũ khí của Global ID (Lưu dạng chuỗi String, vd: 'knife')
        self.armed_bank = {} 

        # --- CẤU HÌNH CHO TÍNH NĂNG GỘP ID ---
        self.last_merge_time = time.time()
        self.MERGE_INTERVAL = 5.0    # Cứ 5 giây dọn dẹp một lần
        self.MERGE_THRESHOLD = 0.8

    def is_armed(self, global_id):
        # Trả về TÊN VŨ KHÍ nếu ID này có trong sổ đen, nếu không trả về None
        with self.lock:
            return self.armed_bank.get(global_id, None)
        
    def _merge_duplicate_ids(self):
        """Hàm dọn dẹp: Tìm các ID giống nhau và gộp lại làm 1"""
        with self.lock:
            # Lấy danh sách các ID hiện có
            active_ids = list(self.feature_bank.keys())
            if len(active_ids) < 2:
                return # Nếu có ít hơn 2 người thì không cần gộp

            merge_actions = []
            
            # itertools.combinations tạo ra tất cả các cặp có thể có (ví dụ: 1-2, 1-3, 2-3)
            for id1, id2 in itertools.combinations(active_ids, 2):
                feat1 = self.feature_bank[id1].reshape(1, -1)
                feat2 = self.feature_bank[id2].reshape(1, -1)
                score = float(cosine_similarity(feat1, feat2)[0][0])
                
                # Nếu cực kỳ giống nhau -> Chắc chắn là do hệ thống nhảy ID
                if score > self.MERGE_THRESHOLD:
                    merge_actions.append((id1, id2, score))

            for id1, id2, score in merge_actions:
                # Đề phòng ID đã bị xóa ở vòng lặp trước đó
                if id1 not in self.feature_bank or id2 not in self.feature_bank:
                    continue
                
                # Ưu tiên giữ lại ID xuất hiện trước (ID nhỏ hơn)
                keep_id = min(id1, id2)
                remove_id = max(id1, id2)

                print(f"[GNN Worker] Đã phát hiện phân mảnh. Gộp ID {remove_id} vào ID {keep_id} (Độ giống: {score:.2f})", flush=True)

                # 1. Cập nhật Feature: Lấy trung bình cộng của cả 2 để ra nét đặc trưng hoàn hảo nhất
                self.feature_bank[keep_id] = (self.feature_bank[keep_id] + self.feature_bank[remove_id]) / 2.0
                del self.feature_bank[remove_id] # Xóa ID thừa đi

                # 2. Định tuyến lại bản đồ: Camera nào đang nhắm vào ID bị xóa sẽ được điều hướng về ID gốc
                for cam_key, global_id in self.global_id_map.items():
                    if global_id == remove_id:
                        self.global_id_map[cam_key] = keep_id

                # 3. Kế thừa Sổ đen vũ khí: Nếu người này từng cầm súng ở ID bị xóa, thì ID gốc cũng phải có súng!
                if remove_id in self.armed_bank:
                    if keep_id not in self.armed_bank:
                        self.armed_bank[keep_id] = self.armed_bank[remove_id]
                    del self.armed_bank[remove_id]

    def run(self):
        print("--- LUỒNG GNN WORKER ĐÃ KHỞI CHẠY ---", flush=True)
        while True:
            try:
                data = self.task_queue.get(timeout=1)
                temp_map = {} 

                for cam_data in data:
                    c_id = cam_data['cam_id'] 
                    tracks_list = cam_data.get('tracks', []) 
                    
                    used_global_ids = set() 
                    
                    for t in tracks_list:
                        local_id = int(t['id']) 
                        feat = t['feature']
                        
                        # SỬA 1: Nhận trực tiếp tên vũ khí từ core.py (trả về None nếu không có)
                        weapon_type_local = t.get('weapon_type', None) 
                        
                        if feat is None:
                            continue
                        
                        feat_np = np.array(feat).flatten()
                        feat_norm = feat_np.reshape(1, -1)
                        
                        matched_id = None 
                        max_score = 0.8 

                        # So khớp
                        for g_id, known_feat in self.feature_bank.items():
                            if g_id in used_global_ids:
                                continue

                            known_feat_norm = np.array(known_feat).reshape(1, -1)
                            score = float(cosine_similarity(feat_norm, known_feat_norm)[0][0])
                            
                            if score > max_score:
                                max_score = score
                                matched_id = g_id

                        if matched_id is not None:
                            g_id_key = int(matched_id)
                            # Cập nhật feature
                            known_feat = self.feature_bank.get(g_id_key)
                            if known_feat is not None:
                                self.feature_bank[g_id_key] = 0.8 * np.array(known_feat).flatten() + 0.2 * feat_np
                            
                            temp_map[(c_id, local_id)] = g_id_key
                            used_global_ids.add(g_id_key)
                            
                            # SỬA 2: Nếu frame này người đó cầm vũ khí, lưu TÊN VŨ KHÍ vào sổ đen
                            if weapon_type_local is not None:
                                with self.lock:
                                    self.armed_bank[g_id_key] = weapon_type_local

                        else:
                            new_id_val = int(self.next_global_id)
                            self.feature_bank[new_id_val] = feat_np
                            temp_map[(c_id, local_id)] = new_id_val
                            self.next_global_id += 1
                            used_global_ids.add(new_id_val)
                            
                            # SỬA 3: Lưu TÊN VŨ KHÍ cho ID mới tạo
                            if weapon_type_local is not None:
                                with self.lock:
                                    self.armed_bank[new_id_val] = weapon_type_local
                            
                            print(f"DEBUG: Tạo ID mới cho Local {local_id} -> Global {new_id_val}", flush=True)
                
                with self.lock:
                    self.global_id_map.update(temp_map)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Lỗi Worker: {e}", flush=True)

            # --- THỰC THI GỘP ID (Chạy ngầm mỗi MERGE_INTERVAL giây) ---
            current_time = time.time()
            if current_time - self.last_merge_time > self.MERGE_INTERVAL:
                self._merge_duplicate_ids()
                self.last_merge_time = current_time

    def update_features(self, step_data):
        """
        Nhận dữ liệu từ vòng lặp chính và đẩy vào hàng đợi xử lý.
        step_data: list chứa embedding và id của các camera.
        """
        try:
            # Nếu hàng đợi đầy, giải phóng cái cũ để lấy cái mới nhất (tránh lag)
            if self.task_queue.full():
                self.task_queue.get_nowait()
            
            self.task_queue.put(step_data, block=False)
        except queue.Full:
            pass