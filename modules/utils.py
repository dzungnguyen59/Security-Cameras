import numpy as np

def make_grid(frames, cols):
    """Hàm sắp xếp danh sách các frame thành lưới"""
    if not frames: # Nếu danh sách rỗng
        return None
    rows = []
    # Chia danh sách frames thành các nhóm nhỏ dựa trên số cột (cols)
    for i in range(0, len(frames), cols):
        row_frames = frames[i:i+cols]
        # Nếu hàng cuối không đủ cột, thêm khung đen cho đủ
        while len(row_frames) < cols:
            black_placeholder = np.zeros_like(frames[0])
            row_frames.append(black_placeholder)
        
        # Ghép các frame trong hàng theo chiều ngang
        rows.append(np.hstack(row_frames))
    if not rows: # Kiểm tra lần cuối trước khi vstack
        return None
    return np.vstack(rows)

def get_intersection_over_weapon(box_person, box_weapon):
    """Tính xem bao nhiêu phần trăm diện tích vũ khí nằm trong box người (IoW)"""
    xp1, yp1, xp2, yp2 = box_person[:4]
    xw1, yw1, xw2, yw2 = box_weapon[:4]
    
    # Tọa độ vùng giao nhau
    x_left = max(xp1, xw1)
    y_top = max(yp1, yw1)
    x_right = min(xp2, xw2)
    y_bottom = min(yp2, yw2)
    
    # Nếu không giao nhau
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    weapon_area = (xw2 - xw1) * (yw2 - yw1)
    
    if weapon_area == 0:
        return 0.0
        
    return intersection_area / weapon_area
