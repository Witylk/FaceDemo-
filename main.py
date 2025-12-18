# -*- coding: utf-8 -*-
import cv2
import face_recognition
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== 配置区域 ====================
DATASET_DIR = "face_dataset"
MODEL_FILE = "face_model.pkl"
RESULT_ROOT_DIR = "recognition_results"
ADMIN_PASSWORD = "SZTU"
THRESHOLD = 0.45
PADDING_RATIO = 0.25


# =================================================

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 字体路径列表
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
        "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
        "/System/Library/Fonts/PingFang.ttc"  # Mac
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, textSize, encoding="utf-8")
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()
    draw.text(position, text, textColor, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def ensure_dirs():
    if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
    if not os.path.exists(RESULT_ROOT_DIR): os.makedirs(RESULT_ROOT_DIR)


def load_known_faces():
    """加载已保存的模型"""
    if not os.path.exists(MODEL_FILE):
        return {"encodings": [], "names": []}
    try:
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
        return data
    except:
        return {"encodings": [], "names": []}


def train_model():
    """
    核心修改：这个函数现在可以独立运行。
    它会遍历 face_dataset 文件夹下的所有子文件夹，读取图片并重新生成模型。
    """
    print("\n" + "=" * 40)
    print("[*] 正在扫描 face_dataset 文件夹并重新训练...")

    known_encodings = []
    known_names = []

    # 检查数据集目录是否存在
    if not os.path.exists(DATASET_DIR):
        print(f"[!] 错误：找不到 {DATASET_DIR} 文件夹。")
        return None

    # 遍历每个人名的文件夹
    person_dirs = os.listdir(DATASET_DIR)
    if not person_dirs:
        print("[!] 警告：dataset 文件夹是空的！")
        return None

    total_images = 0
    for name in person_dirs:
        person_dir = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(person_dir): continue

        print(f"[-] 正在处理用户: {name}")
        images = os.listdir(person_dir)

        has_face = False
        for filename in images:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(person_dir, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    # 增加容错：如果图片里有多张脸，取第一张；如果没有脸，跳过
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                        total_images += 1
                        has_face = True
                    else:
                        print(f"    [x] 跳过图片(未检测到人脸): {filename}")
                except Exception as e:
                    print(f"    [!] 图片读取错误 {filename}: {e}")

        if not has_face:
            print(f"    [!] 警告：用户 {name} 目录下没有有效的包含人脸的图片！")

    if not known_encodings:
        print("[!] 训练失败：没有提取到任何有效的人脸特征。")
        return None

    data = {"encodings": known_encodings, "names": known_names}
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[√] 训练完成！")
    print(f"    - 共包含用户数: {len(set(known_names))}")
    print(f"    - 总人脸样本数: {total_images}")
    print(f"    - 模型已保存至: {MODEL_FILE}")
    print("=" * 40 + "\n")
    return data


def register_face():
    pwd = input("\n请输入管理员密码: ")
    if pwd != ADMIN_PASSWORD:
        print("[!] 密码错误！")
        return
    name = input("请输入录入人员的姓名 (例如: Obama): ")
    if not name: return

    person_dir = os.path.join(DATASET_DIR, name)
    if not os.path.exists(person_dir): os.makedirs(person_dir)

    print(f"\n[-] 正在打开摄像头为 [{name}] 采集数据...")
    print("[-] 按 's' 保存，按 'q' 退出")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Linux/Mac可能需要 CAP_V4L2，Windows通常不用
    if not cap.isOpened():
        # 如果上面失败，尝试默认索引
        cap = cv2.VideoCapture(0)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        display_frame = cv2AddChineseText(display_frame, f"正在录入: {name}", (20, 20), (0, 255, 255), 30)
        display_frame = cv2AddChineseText(display_frame, f"已采集: {count} 张", (20, 60), (0, 255, 255), 25)

        cv2.imshow("Register", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            count += 1
            filename = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"    [√] 保存成功: {filename}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 录入结束后，自动触发训练
    if count > 0:
        train_model()


# ========================================================
#   图像处理逻辑 (保持不变)
# ========================================================
def process_single_image(img_path, data, output_folder, show_window=True):
    # ... (代码逻辑保持原样，省略重复部分以节省篇幅，功能与之前一致) ...
    try:
        image = face_recognition.load_image_file(img_path)
        cv_img = cv2.imread(img_path)
        clean_img = cv_img.copy()
        if cv_img is None: return False

        locs = face_recognition.face_locations(image, number_of_times_to_upsample=1)
        encs = face_recognition.face_encodings(image, locs)

        found_target = False
        found_names = []

        for (t, r, b, l), enc in zip(locs, encs):
            match = face_recognition.compare_faces(data["encodings"], enc, tolerance=THRESHOLD)
            name = "Unknown"
            dists = face_recognition.face_distance(data["encodings"], enc)

            if len(dists) > 0 and match[np.argmin(dists)]:
                name = data["names"][np.argmin(dists)]
                found_target = True
                found_names.append(name)

                # 自动学习保存
                target_dir = os.path.join(DATASET_DIR, name)
                if not os.path.exists(target_dir): os.makedirs(target_dir)

                h_img, w_img = clean_img.shape[:2]
                face_h = b - t
                face_w = r - l
                new_t = max(0, int(t - face_h * PADDING_RATIO))
                new_b = min(h_img, int(b + face_h * PADDING_RATIO))
                new_l = max(0, int(l - face_w * PADDING_RATIO))
                new_r = min(w_img, int(r + face_w * PADDING_RATIO))
                face_crop = clean_img[new_t:new_b, new_l:new_r]

                if face_crop.size > 0:
                    new_filename = f"{name}_auto_{np.random.randint(10000, 99999)}.jpg"
                    cv2.imwrite(os.path.join(target_dir, new_filename), face_crop)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(cv_img, (l, t), (r, b), color, 2)

            display_name = name if name == "Unknown" else f"{name}(已识别)"
            if name == "Unknown":
                cv2.putText(cv_img, display_name, (l, t - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            else:
                cv_img = cv2AddChineseText(cv_img, display_name, (l, t - 30), color, 30)

        if found_target:
            filename = "checked_" + os.path.basename(img_path)
            save_p = os.path.join(output_folder, filename)
            cv2.imwrite(save_p, cv_img)
            print(f"    [√] 发现目标: {','.join(set(found_names))} -> {save_p}")
            if show_window:
                cv2.imshow("Result", cv_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return True
        else:
            print(f"    [x] 未发现目标: {os.path.basename(img_path)}")
            return False
    except Exception as e:
        print(f"    [!] Error: {e}")
        return False


def recognize_mode():
    data = load_known_faces()
    # 增加智能检测：如果模型为空，询问是否训练
    if not data["encodings"]:
        print("[!] 检测到模型为空或不存在！")
        choice = input("是否立即扫描 dataset 文件夹进行训练? (y/n): ")
        if choice.lower() == 'y':
            data = train_model()
            if not data: return  # 训练失败
        else:
            return

    input_path = input("\n请输入图片路径 或 文件夹路径: ").strip("'\"")
    if not os.path.exists(input_path):
        print("[!] 路径不存在")
        return

    folder_name = input("请给该组输出图片命名 (回车默认): ")
    if not folder_name: folder_name = "default_output"
    current_output_dir = os.path.join(RESULT_ROOT_DIR, folder_name)
    if not os.path.exists(current_output_dir): os.makedirs(current_output_dir)

    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[-] 开始批量处理 {len(files)} 张图片...")
        cnt = 0
        for f in files:
            if process_single_image(os.path.join(input_path, f), data, current_output_dir, show_window=False):
                cnt += 1
        print(f"\n[-] 完成。筛选出 {cnt} 张目标。查看: {current_output_dir}")
    else:
        process_single_image(input_path, data, current_output_dir, show_window=True)


def recognize_video_cam():
    data = load_known_faces()
    if not data["encodings"]:
        print("[!] 模型为空，请先【重新训练模型】或【录入人脸】")
        return
    print("\n[-] 摄像头启动... 按 'q' 退出")
    # Windows 推荐去掉 cv2.CAP_V4L2，直接用 0
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        locs = face_recognition.face_locations(rgb_small_frame)
        encs = face_recognition.face_encodings(rgb_small_frame, locs)

        face_names = []
        for enc in encs:
            match = face_recognition.compare_faces(data["encodings"], enc, tolerance=THRESHOLD)
            name = "Unknown"
            dists = face_recognition.face_distance(data["encodings"], enc)
            if len(dists) > 0 and match[np.argmin(dists)]:
                name = data["names"][np.argmin(dists)]
            face_names.append(name)

        for (t, r, b, l), name in zip(locs, face_names):
            t *= 4;
            r *= 4;
            b *= 4;
            l *= 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            if name == "Unknown":
                cv2.putText(frame, name, (l, t - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            else:
                frame = cv2AddChineseText(frame, name, (l, t - 30), color, 30)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()


def process_video_file():
    data = load_known_faces()
    if not data["encodings"]:
        print("[!] 模型为空，请先【重新训练模型】")
        return
    video_path = input("\n请输入视频路径: ").strip("'\"")
    if not os.path.exists(video_path): return
    output_name = input("输出视频命名 (无需后缀): ")
    if not output_name: output_name = "processed_video"
    save_path = os.path.join(RESULT_ROOT_DIR, output_name + ".mp4")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"[-] 开始处理视频: {save_path}")
    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cnt += 1
        if cnt % 10 == 0: print(f"\r进度: {cnt}/{total} ({(cnt / total) * 100:.1f}%)", end="")

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = np.ascontiguousarray(small[:, :, ::-1])
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        names = []
        for enc in encs:
            match = face_recognition.compare_faces(data["encodings"], enc, tolerance=THRESHOLD)
            name = "Unknown"
            dists = face_recognition.face_distance(data["encodings"], enc)
            if len(dists) > 0 and match[np.argmin(dists)]:
                name = data["names"][np.argmin(dists)]
            names.append(name)

        for (t, r, b, l), name in zip(locs, names):
            t *= 2;
            r *= 2;
            b *= 2;
            l *= 2
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            if name == "Unknown":
                cv2.putText(frame, name, (l, t - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            else:
                frame = cv2AddChineseText(frame, name, (l, t - 30), color, 30)
        out.write(frame)
    cap.release()
    out.release()
    print("\n[√] 处理完成")


def main():
    ensure_dirs()
    # 启动时检查模型状态
    data = load_known_faces()
    model_status = f"已加载 {len(set(data['names']))} 人" if data["names"] else "未加载模型"

    while True:
        print("\n" + "=" * 35)
        print(f"   人脸识别系统 V2.0  [{model_status}]")
        print("=" * 35)
        print(" 1. 识别模式 (图片/视频/监控)")
        print(" 2. 录入人脸 (通过摄像头)")
        print(" 3. ★ 重新训练模型 (从文件夹读取) ★")
        print(" 0. 退出")
        print("=" * 35)

        c = input("请输入选项: ")

        if c == '1':
            print("\n  >> 1. 图片/文件夹筛选")
            print("  >> 2. 摄像头实时")
            print("  >> 3. 视频文件处理")
            sc = input("  选项: ")
            if sc == '1':
                recognize_mode()
            elif sc == '2':
                recognize_video_cam()
            elif sc == '3':
                process_video_file()

        elif c == '2':
            register_face()
            # 录入完更新状态显示
            data = load_known_faces()
            model_status = f"已加载 {len(set(data['names']))} 人"

        elif c == '3':
            # 手动触发训练
            new_data = train_model()
            if new_data:
                model_status = f"已加载 {len(set(new_data['names']))} 人"

        elif c == '0':
            break


if __name__ == "__main__":
    main()