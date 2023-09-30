import tkinter as tk

import cv2
import torch
from PIL import ImageTk, Image

# Tạo một cửa sổ tkinter
window = tk.Tk()
window.title("Manhzin")
window.geometry("1000x1000")
window.configure(bg='#30D5C8')
name_label = tk.Label(window, text=f"Phát hiện và đếm số lượng người ", font=("Helvetica", 28), bg='#CCFFCC')
name_label.pack()



# Tạo một đối tượng Label để hiển thị khung hình
image_label = tk.Label(window)
image_label.pack()

cap = cv2.VideoCapture(0)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
count = 0

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

    frame = cv2.resize(frame, (1020, 500))
    results = model(frame)

    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])
        print(d)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, f'P{c}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        c += 1

    cv2.putText(frame, f'So nguoi phat hien : {c - 1}', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 255), 2)

    # Chuyển đổi từ định dạng OpenCV sang PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Chuyển đổi PIL Image thành đối tượng ImageTk
    image_tk = ImageTk.PhotoImage(image)

    # Cập nhật hình ảnh trên Label
    image_label.configure(image=image_tk)
    image_label.image = image_tk

    window.after(1, process_frame)




# Bắt đầu xử lý khung hình
process_frame()

# Chạy vòng lặp chính của tkinter
window.mainloop()

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()