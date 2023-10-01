import tkinter as tk
import cv2
import torch
from PIL import ImageTk, Image

# Tạo một cửa sổ tkinter
window = tk.Tk()
window.title("Machine Learning")
window.geometry("1500x1000")
window.configure(bg='#FFFFFF')

# Tạo một frame để chứa logo_label và name_label
top_frame = tk.Frame(window, bg='#FFFFFF')
top_frame.pack(pady=20)

# Tạo một đối tượng Label để hiển thị logo
logo_image = ImageTk.PhotoImage(Image.open("Data/img.png").resize((100, 100)))
logo_label = tk.Label(top_frame, image=logo_image, bg='#FFFFFF')
logo_label.pack(side=tk.LEFT)

# Tạo một đối tượng Label để hiển thị dòng name_label
name_label = tk.Label(top_frame, text="Phát hiện và đếm số lượng người", font=("Helvetica", 28), bg='#FFFFFF')
name_label.pack(side=tk.LEFT)

# Tạo một đối tượng Label để hiển thị khung hình
image_label = tk.Label(window)
image_label.pack()

cap = cv2.VideoCapture("Data/vid1.mp4")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
count = 0

# Tạo một đối tượng Label để hiển thị số lượng người
count_label = tk.Label(window, text="Số người phát hiện: 0", font=("Helvetica", 16), pady=10)
count_label.pack()

def process_frame():
    global count
    ret, frame = cap.read()
    if not ret:
        return
    count += 1
    if count % 3 != 0:
        window.after(1, process_frame)
        return
    c = 1

    frame = cv2.resize(frame, (1000, 500))
    results = model(frame)

    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
        print(d)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        c += 1

    # Cập nhật số lượng người trên Label
    count_label.config(text=f"Số người phát hiện: {c - 1}", font=("Helvetica", 20))

    # Chuyển đổi từ định dạng OpenCV sang PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Chuyển đổi PIL Image thành đối tượng ImageTk
    image_tk = ImageTk.PhotoImage(image)

    # Cập nhật hình ảnh trên Label
    image_label.configure(image=image_tk)
    image_label.image = image_tk

    window.after(1, process_frame)

# Hàm xử lý sự kiện khi nhấn nút kết thúc
def exit_program():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

# Tạo một nút kết thúc
exit_button = tk.Button(window, text="Kết thúc",font=("Helvetica", 16), command=exit_program,bg='#FF0000')
exit_button.pack(side=tk.BOTTOM)

# Bắt đầu xử lý khung hình
process_frame()

# Chạy vòng lặp chính của tkinter
window.mainloop()