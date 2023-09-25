import tkinter as tk

# 创建主窗口
root = tk.Tk()
root.title("Login")

# 设置窗口大小和位置
window_width = 300
window_height = 150
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# 创建标签和输入框
username_label = tk.Label(root, text="Account:")
username_label.pack()
username_entry = tk.Entry(root)
username_entry.pack()

password_label = tk.Label(root, text="Password:")
password_label.pack()
password_entry = tk.Entry(root, show="*")  # 输入密码时显示*，以保护密码
password_entry.pack()

# 登录验证函数
def login():
    # 获取输入的用户名和密码
    username = username_entry.get()
    password = password_entry.get()

    # 在这里添加您的验证逻辑，例如，简单示例中验证用户名和密码是否匹配
    if username == "admin" and password == "password":
        result_label.config(text="Login successfully")
        open_control_window()  # 登录成功时打开控制窗口
    else:
        result_label.config(text="Wrong information")

# 创建登录按钮
login_button = tk.Button(root, text="Login", command=login)
login_button.pack()

# 显示登录结果的标签
result_label = tk.Label(root, text="")
result_label.pack()

# 打开控制窗口函数
def open_control_window():
    control_window = tk.Toplevel(root)
    control_window.title("Human Control")

    # 设置控制窗口的大小为200x200
    control_window.geometry("200x200")

    # 添加速度控制的进度条
    speed_label = tk.Label(control_window, text="Velocity:")
    speed_label.pack()
    speed_scale = tk.Scale(control_window, from_=0, to=50, orient="horizontal")
    speed_scale.pack()

    m_s = speed_scale.get()
    print(m_s)

    # 添加上下左右按钮，并绑定相应的函数
    def move(direction):
        m_s = speed_scale.get()
        print("Velocity : " + str(m_s))
        print("Movement : " + str(direction))

    up_button = tk.Button(control_window, text="Forward", command=lambda: move("Forward"))
    down_button = tk.Button(control_window, text="Back", command=lambda: move("Back"))
    left_button = tk.Button(control_window, text="Left", command=lambda: move("Left"))
    right_button = tk.Button(control_window, text="Right", command=lambda: move("Right"))

    up_button.pack()
    down_button.pack()
    left_button.pack()
    right_button.pack()


# 运行主循环
root.mainloop()


