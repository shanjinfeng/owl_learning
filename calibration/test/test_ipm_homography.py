import cv2
import numpy as np
import os
from marker_detect import ArucoMarkerDetector

# ================= 物理参数配置 =================
IMAGE_WIDTH = 2048    # 大恒相机分辨率 宽
IMAGE_HEIGHT = 1536   # 大恒相机分辨率 高

def get_world_coords(u, v, H_inv):
    """
    根据逆单应性矩阵 H_inv，将图像像素坐标 (u, v) 转换为地面物理坐标 (X, Y)
    (输出单位由生成 H 矩阵时的 square_size 决定，目前为 毫米 mm)
    """
    # 齐次坐标计算
    w = H_inv[2, 0] * u + H_inv[2, 1] * v + H_inv[2, 2]
    if abs(w) < 1e-9:
        return 0.0, 0.0
    
    x = (H_inv[0, 0] * u + H_inv[0, 1] * v + H_inv[0, 2]) / w
    y = (H_inv[1, 0] * u + H_inv[1, 1] * v + H_inv[1, 2]) / w
    return x, y

def main():
    # 1. 加载单应性矩阵并求逆
    if not os.path.exists("/home/sjf/owl/calibration/calibration/H.npy"):
        print("[ERROR] 找不到 H.npy，请先运行标定脚本！")
        return
        
    H = np.load("/home/sjf/owl/calibration/calibration/H.npy")
    H_inv = np.linalg.inv(H)
    print("[INFO] 成功加载并求逆单应性矩阵 H")
    
    # ========================================================
    # 2. 建立绝对防错的相对物理坐标系（消除标定板摆放角度的干扰）
    # ========================================================
    # 原点：图像底部中点
    p_bot = get_world_coords(IMAGE_WIDTH / 2.0, IMAGE_HEIGHT, H_inv)
    # 前方参考点：图像顶部中点
    p_top = get_world_coords(IMAGE_WIDTH / 2.0, 0, H_inv)
    # 右方参考点：图像右下角
    p_right_edge = get_world_coords(IMAGE_WIDTH, IMAGE_HEIGHT, H_inv)
    
    # 算出现实世界中“向前”的单位向量
    vec_fwd_raw = np.array(p_top) - np.array(p_bot)
    vec_fwd = vec_fwd_raw / np.linalg.norm(vec_fwd_raw)
    
    # 算出现实世界中“向右”的单位向量（强制与“向前”垂直）
    vec_right_raw = np.array(p_right_edge) - np.array(p_bot)
    vec_right_ortho = vec_right_raw - np.dot(vec_right_raw, vec_fwd) * vec_fwd
    vec_right = vec_right_ortho / np.linalg.norm(vec_right_ortho)

    print(f"[INFO] 物理原点(画面底部中点)坐标: X={p_bot[0]:.1f}mm, Y={p_bot[1]:.1f}mm")

    # 3. 初始化二维码检测器
    detector = ArucoMarkerDetector()
    
    # 4. 读取测试图像
    test_image_path = "/home/sjf/owl/calibration/test_png/2026-05-18_15_00_36_897.png"  # ★ 请确保这里是你刚拍的图片的名字
    if not os.path.exists(test_image_path):
        # 尝试读取 png
        test_image_path = test_image_path.replace(".jpg", ".png")
        if not os.path.exists(test_image_path):
            print(f"[ERROR] 找不到图片 {test_image_path}，请修改代码中的图片路径！")
            return
            
    img = cv2.imread(test_image_path)
    
    # 5. 检测二维码
    boxes = detector.predict(img)
    print(f"\n[INFO] 检测到 {len(boxes)} 个二维码")
    vis_img = detector.draw_boxes(img, boxes)
    
    # 6. 逆透视坐标解算
    for i, box in enumerate(boxes):
        # 提取中心点像素
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        marker_id = int(box[4]) if len(box) >= 5 else -1
        
        # 映射到物理世界坐标 (单位：毫米)
        p_marker = get_world_coords(cx, cy, H_inv)
        
        # 计算相对于底部中点的物理向量 (单位：毫米)
        vec_target = np.array(p_marker) - np.array(p_bot)
        
        # 将向量投影到“向前”和“向右”方向（内积）
        forward_dist_mm = np.dot(vec_target, vec_fwd)
        right_dist_mm = np.dot(vec_target, vec_right)
        
        # ★ 将毫米(mm)转换为厘米(cm) ★
        forward_cm = forward_dist_mm / 10.0
        right_cm = right_dist_mm / 10.0
        
        print(f"--- Marker ID: {marker_id} ---")
        print(f"像素位置 : u={cx:.1f}, v={cy:.1f}")
        print(f"物理位置 : 向前 = {forward_cm:.2f} cm, 向右 = {right_cm:.2f} cm")
        
        # 可视化标注
        text = f"Fwd:{forward_cm:.1f}cm Right:{right_cm:.1f}cm"
        cv2.putText(vis_img, text, (int(box[0]), int(box[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 7. 显示结果
    cv2.namedWindow("IPM Coordinates", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("IPM Coordinates", 1024, 768)
    cv2.imshow("IPM Coordinates", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()