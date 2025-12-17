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
RESULT_ROOT_DIR = "recognition_results"  # 结果的总目录
ADMIN_PASSWORD = "SZTU"
THRESHOLD = 0.42  # 阈值
PADDING_RATIO = 0.25  # [新] 截图扩边比例 (0.25表示上下左右各向外扩25%)，解决截图太小的问题


# =================================================

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
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
    if not os.path.exists(MODEL_FILE): return {"encodings": [], "names": []}
    try:
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
        return data
    except:
        return {"encodings": [], "names": []}


def train_model():
    print("\n[*] 正在重新训练模型，请稍候...")
    known_encodings = []
    known_names = []
    for name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(person_dir): continue
        print(f"[-] 正在处理用户: {name}")
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(person_dir, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                except:
                    pass
    data = {"encodings": known_encodings, "names": known_names}
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"[√] 训练完成！共录入 {len(known_encodings)} 张人脸数据。\n")
    return data


def register_face():
    pwd = input("\n请输入管理员密码: ")
    if pwd != ADMIN_PASSWORD:
        print("[!] 密码错误！")
        return
    name = input("请输入录入人员的姓名 (例如: YuShiming): ")
    if not name: return

    person_dir = os.path.join(DATASET_DIR, name)
    if not os.path.exists(person_dir): os.makedirs(person_dir)

    print(f"\n[-] 正在打开摄像头为 [{name}] 采集数据...")
    print("[-] 按 's' 保存，按 'q' 退出")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

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
            # 命名格式：人名(拍照)_序号.jpg
            filename = os.path.join(person_dir, f"{name}(拍照)_{count}.jpg")
            # 录入时直接保存原图
            cv2.imwrite(filename, frame)
            print(f"    [√] 保存成功: {filename}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if count > 0: train_model()


# ========================================================
#   核心修正：单张图片处理 (去重 + 智能扩边)
# ========================================================
def process_single_image(img_path, data, output_folder, show_window=True):
    try:
        image = face_recognition.load_image_file(img_path)
        cv_img = cv2.imread(img_path)

        # 备份一份干净的原始图片，用于裁剪回流
        clean_img = cv_img.copy()

        if cv_img is None: return False

        # 上采样设为1
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

                # --- [自动学习逻辑] ---
                # 只有识别出具体名字时，才执行保存
                target_dir = os.path.join(DATASET_DIR, name)
                if not os.path.exists(target_dir): os.makedirs(target_dir)

                # [关键修正1] 智能扩边 (Padding) - 解决截图太小的问题
                h_img, w_img = clean_img.shape[:2]
                face_h = b - t
                face_w = r - l

                # 计算新的坐标（向外扩张 PADDING_RATIO）
                new_t = max(0, int(t - face_h * PADDING_RATIO))
                new_b = min(h_img, int(b + face_h * PADDING_RATIO))
                new_l = max(0, int(l - face_w * PADDING_RATIO))
                new_r = min(w_img, int(r + face_w * PADDING_RATIO))

                # 裁剪扩边后的人脸
                face_crop = clean_img[new_t:new_b, new_l:new_r]

                if face_crop.size > 0:
                    # [关键修正2] 确保这里只有一次 imwrite
                    # 命名格式：人名(已识别)_随机数.jpg
                    new_filename = f"{name}(已识别)_{np.random.randint(10000, 99999)}.jpg"
                    save_path_dataset = os.path.join(target_dir, new_filename)

                    cv2.imwrite(save_path_dataset, face_crop)
                    print(f"    [★] 自动学习: 人脸区域(已扩边)已加入训练集 -> {new_filename}")
                # -----------------------

            # 绘制结果 (只在 cv_img 上画，不影响 clean_img)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(cv_img, (l, t), (r, b), color, 2)

            # 显示名字（加个括号显示状态，仅显示用）
            display_name = name
            if name != "Unknown":
                display_name = f"{name}(已识别)"

            if name == "Unknown":
                cv2.putText(cv_img, display_name, (l, t - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            else:
                cv_img = cv2AddChineseText(cv_img, display_name, (l, t - 30), color, 30)

        # 结果输出 (只有发现目标才保存一张大图)
        if found_target:
            filename = "checked_" + os.path.basename(img_path)
            save_p = os.path.join(output_folder, filename)

            cv2.imwrite(save_p, cv_img)
            print(f"    [√] 发现目标: {','.join(set(found_names))} -> 结果图已存至: {save_p}")

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
    if not data["encodings"]:
        print("[!] 请先录入！")
        return

    input_path = input("\n请输入图片路径 或 文件夹路径: ").strip("'\"")
    if not os.path.exists(input_path):
        print("[!] 路径不存在")
        return

    folder_name = input("请给该组输出图片命名 (例如: 第一次测试结果): ")
    if not folder_name:
        folder_name = "default_output"

    current_output_dir = os.path.join(RESULT_ROOT_DIR, folder_name)
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)

    print(f"[-] 输出目录已建立: {current_output_dir}")

    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[-] 开始批量处理 {len(files)} 张图片...")
        cnt = 0
        for f in files:
            if process_single_image(os.path.join(input_path, f), data, current_output_dir, show_window=False):
                cnt += 1
        print(f"\n[-] 批量处理完成，共筛选出 {cnt} 张目标图片。")
        print(f"[-] 请查看文件夹: {current_output_dir}")
    else:
        process_single_image(input_path, data, current_output_dir, show_window=True)


def recognize_video_cam():
    data = load_known_faces()
    if not data["encodings"]:
        print("[!] 请先录入！")
        return
    print("\n[-] 摄像头启动... 按 'q' 退出")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
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
        print("[!] 请先录入！")
        return
    video_path = input("\n请输入视频路径: ").strip("'\"")
    if not os.path.exists(video_path): return

    output_name = input("请给输出视频命名 (无需后缀): ")
    if not output_name: output_name = "processed_video"

    save_path = os.path.join(RESULT_ROOT_DIR, output_name + ".mp4")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"[-] 开始处理视频，保存至: {save_path}")
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
    while True:
        print("\n" + "=" * 30)
        print("   人脸识别系统 - 终极定制版")
        print("=" * 30)
        print(" 1. 识别模式")
        print(" 2. 录入人脸 (管理员)")
        print(" 0. 退出")
        print("=" * 30)
        c = input("选项: ")
        if c == '1':
            print("\n 1. 图片/文件夹筛选")
            print(" 2. 摄像头实时")
            print(" 3. 视频文件处理")
            sc = input("选项: ")
            if sc == '1':
                recognize_mode()
            elif sc == '2':
                recognize_video_cam()
            elif sc == '3':
                process_video_file()
        elif c == '2':
            register_face()
        elif c == '0':
            break


if __name__ == "__main__":
    main()