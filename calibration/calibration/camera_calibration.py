import cv2
import numpy as np
import glob
import os

# =========================================================
# 1. 配置
# =========================================================
IMAGE_DIR = "/home/sjf/owl/calibration/calibration_png"

# 11×8 内角点（你指定）
chessboard_size = (11, 8)

# ✔ 关键修复：统一单位 → 米（避免H爆炸）
square_size = 20.1  # 26mm = 0.026m

# =========================================================
# 2. 构造棋盘3D点（世界坐标）
# =========================================================
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# =========================================================
# 3. 读取图像
# =========================================================
image_paths = []
image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.ppm"))
image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.png"))
image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
image_paths += glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
image_paths = sorted(image_paths)

print("\n==============================")
print(f"[INFO] IMAGE_DIR = {IMAGE_DIR}")
print(f"[INFO] Found images = {len(image_paths)}")
print("==============================\n")

if len(image_paths) == 0:
    raise RuntimeError("No images found")

# =========================================================
# 4. 标定数据
# =========================================================
objpoints = []
imgpoints = []

valid = 0
gray_shape = None

# =========================================================
# 5. 棋盘检测（无GUI版本）
# =========================================================
for path in image_paths:
    print("[INFO] Processing:", path)

    img = cv2.imread(path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners)

        valid += 1
    else:
        print("[WARN] chessboard not found")

print("\n==============================")
print(f"[INFO] Valid images = {valid}")
print("==============================\n")

if valid < 5:
    raise RuntimeError("Too few valid images for calibration")

# =========================================================
# 6. 相机标定
# =========================================================
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray_shape,
    None,
    None
)

print("\n========== CAMERA MATRIX ==========")
print(K)

print("\n========== DISTORTION ==========")
print(dist.ravel())

# =========================================================
# 7. 取第一张图做 BEV（solvePnP）
# =========================================================
img = cv2.imread(image_paths[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size)

if not ret:
    raise RuntimeError("First image chessboard not found")

corners = cv2.cornerSubPix(
    gray, corners, (11, 11), (-1, -1),
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)

# =========================================================
# 8. solvePnP（关键修复点）
# =========================================================
success, rvec, tvec = cv2.solvePnP(
    objp,
    corners,
    K,
    dist
)

if not success:
    raise RuntimeError("solvePnP failed")

R, _ = cv2.Rodrigues(rvec)

# =========================================================
# 9. 正确 Homography（修复爆炸问题）
# =========================================================
H = K @ np.hstack((R[:, :2], tvec))

# ✔ 关键：归一化（防止数值爆炸）
H = H / H[2, 2]

print("\n========== HOMOGRAPHY ==========")
print(H)

# =========================================================
# 10. BEV生成（稳定版本）
# =========================================================
bev_size = (800, 800)

bev = cv2.warpPerspective(img, H, bev_size)

# =========================================================
# 11. 保存结果（自动退出关键）
# =========================================================
np.save("K.npy", K)
np.save("dist.npy", dist)
np.save("H.npy", H)

cv2.imwrite("bev_result.jpg", bev)

print("\n[INFO] Saved: K.npy / dist.npy / H.npy / bev_result.jpg")

# =========================================================
# 12. 自动结束（解决你说的“不会退出”问题）
# =========================================================
print("\n==============================")
print("✅ DONE - Program finished automatically")
print("==============================\n")

