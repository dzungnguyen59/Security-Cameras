import itertools
import threading
import queue
import numpy as np
import time
import faiss

# Luồng xử lý GNN để gán ID toàn cục cho các đối tượng được theo dõi (Powered by FAISS)
class GlobalGNNWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.task_queue = queue.Queue(maxsize=1)
        
        # --- CẤU HÌNH FAISS (Vector Database) ---
        self.FEATURE_DIM = 512  # Kích thước vector của OSNet (sửa lại nếu model của anh dùng size khác)
        # Khởi tạo Index tính Inner Product (Tích vô hướng)
        base_index = faiss.IndexFlatIP(self.FEATURE_DIM)
        # Bọc IndexIDMap để tự quản lý Global ID (1, 2, 3...) thay vì để FAISS tự đánh số
        self.index = faiss.IndexIDMap(base_index)
        
        self.feature_bank = {} # Vẫn giữ dict để lưu bản sao vector phục vụ việc gộp ID
        self.next_global_id = 1 
        self.lock = threading.Lock() 
        self.global_id_map = {} 
        
        # Bộ nhớ lưu trạng thái có vũ khí của Global ID (vd: 'knife')
        self.armed_bank = {} 

        # --- CẤU HÌNH CHO TÍNH NĂNG GỘP ID ---
        self.last_merge_time = time.time()
        self.MERGE_INTERVAL = 5.0    # Cứ 5 giây dọn dẹp một lần
        self.MERGE_THRESHOLD = 0.85  # Nên để ngưỡng gộp cao hơn ngưỡng nhận diện một chút

    def is_armed(self, global_id):
        # Trả về TÊN VŨ KHÍ nếu ID này có trong sổ đen, nếu không trả về None
        with self.lock:
            return self.armed_bank.get(global_id, None)
        
    def _merge_duplicate_ids(self):
        """Hàm dọn dẹp: Tìm các ID giống nhau và gộp lại làm 1"""
        with self.lock:
            active_ids = list(self.feature_bank.keys())
            if len(active_ids) < 2:
                return # Nếu có ít hơn 2 người thì không cần gộp

            merge_actions = []
            
            for id1, id2 in itertools.combinations(active_ids, 2):
                feat1 = self.feature_bank[id1]
                feat2 = self.feature_bank[id2]
                
                # TỐI ƯU TOÁN HỌC: Vì vector đã chuẩn hóa L2, Cosine Similarity chính là Tích vô hướng (Dot Product)
                score = float(np.dot(feat1.flatten(), feat2.flatten()))
                
                if score > self.MERGE_THRESHOLD:
                    merge_actions.append((id1, id2, score))

            for id1, id2, score in merge_actions:
                if id1 not in self.feature_bank or id2 not in self.feature_bank:
                    continue
                
                keep_id = min(id1, id2)
                remove_id = max(id1, id2)

                print(f"[GNN Worker] Đã phát hiện phân mảnh. Gộp ID {remove_id} vào ID {keep_id} (Độ giống: {score:.2f})", flush=True)

                # 1. Cập nhật Feature: Lấy trung bình cộng của cả 2 để ra nét đặc trưng hoàn hảo nhất
                updated_feat = (self.feature_bank[keep_id] + self.feature_bank[remove_id]) / 2.0
                faiss.normalize_L2(updated_feat) # BẮT BUỘC chuẩn hóa lại sau khi cộng
                
                self.feature_bank[keep_id] = updated_feat
                del self.feature_bank[remove_id] 
                
                # CẬP NHẬT VÀO FAISS: Xóa cả 2 ID cũ, nhét ID giữ lại với vector mới vào
                self.index.remove_ids(np.array([keep_id, remove_id], dtype=np.int64))
                self.index.add_with_ids(updated_feat, np.array([keep_id], dtype=np.int64))

                # 2. Định tuyến lại bản đồ hiển thị
                for cam_key, global_id in list(self.global_id_map.items()):
                    if global_id == remove_id:
                        self.global_id_map[cam_key] = keep_id

                # 3. Kế thừa Sổ đen vũ khí
                if remove_id in self.armed_bank:
                    if keep_id not in self.armed_bank:
                        self.armed_bank[keep_id] = self.armed_bank[remove_id]
                    del self.armed_bank[remove_id]

    def run(self):
        print("--- LUỒNG GNN WORKER ĐÃ KHỞI CHẠY (FAISS TÍCH HỢP) ---", flush=True)
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
                        weapon_type_local = t.get('weapon_type', None) 
                        
                        if feat is None:
                            continue
                        
                        # CHUẨN BỊ DỮ LIỆU CHO FAISS (Ép kiểu float32 và shape 2D)
                        feat_np = np.array(feat, dtype=np.float32).reshape(1, -1)
                        # Chuẩn hóa chiều dài vector về 1
                        faiss.normalize_L2(feat_np)
                        
                        matched_id = None 
                        max_score = 0.8 

                        # SO KHỚP VỚI FAISS 
                        if self.index.ntotal > 0:
                            scores, neighbors = self.index.search(feat_np, k=1)
                            best_score = float(scores[0][0])
                            best_id = int(neighbors[0][0])
                            
                            if best_score > max_score and best_id not in used_global_ids and best_id != -1:
                                matched_id = best_id

                        if matched_id is not None:
                            g_id_key = int(matched_id)
                            
                            # Cập nhật mượt mà feature (0.8 cũ + 0.2 mới)
                            known_feat = self.feature_bank[g_id_key]
                            updated_feat = 0.8 * known_feat + 0.2 * feat_np
                            faiss.normalize_L2(updated_feat) # Chuẩn hóa lại
                            
                            self.feature_bank[g_id_key] = updated_feat
                            
                            # Cập nhật Vector vào bộ nhớ FAISS
                            self.index.remove_ids(np.array([g_id_key], dtype=np.int64))
                            self.index.add_with_ids(updated_feat, np.array([g_id_key], dtype=np.int64))
                            
                            temp_map[(c_id, local_id)] = g_id_key
                            used_global_ids.add(g_id_key)
                            
                            if weapon_type_local is not None:
                                with self.lock:
                                    self.armed_bank[g_id_key] = weapon_type_local

                        else:
                            # TẠO ID MỚI
                            new_id_val = int(self.next_global_id)
                            self.feature_bank[new_id_val] = feat_np.copy()
                            
                            # Lưu vào FAISS (Bắt buộc dùng np.int64 cho ID)
                            self.index.add_with_ids(feat_np, np.array([new_id_val], dtype=np.int64))
                            
                            temp_map[(c_id, local_id)] = new_id_val
                            self.next_global_id += 1
                            used_global_ids.add(new_id_val)
                            
                            if weapon_type_local is not None:
                                with self.lock:
                                    self.armed_bank[new_id_val] = weapon_type_local
                            
                            print(f"DEBUG: TẠO ID {new_id_val} TỪ CAMERA {c_id} (FAISS Indexed)", flush=True)
                
                with self.lock:
                    self.global_id_map.update(temp_map)
                    
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Lỗi Worker: {e}", flush=True)

            # --- THỰC THI GỘP ID ---
            current_time = time.time()
            if current_time - self.last_merge_time > self.MERGE_INTERVAL:
                self._merge_duplicate_ids()
                self.last_merge_time = current_time

    def update_features(self, step_data):
        try:
            if self.task_queue.full():
                self.task_queue.get_nowait()
            self.task_queue.put(step_data, block=False)
        except queue.Full:
            pass