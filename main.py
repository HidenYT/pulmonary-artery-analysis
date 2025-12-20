import tkinter as tk
from core.config import ConfigService
from ui.main_window import MainWindow
from ui.settings_window import SettingsWindow


config_service = ConfigService()


root = tk.Tk()
root.title("Анализ КТ")

menu = tk.Menu(root)
root.config(menu=menu)

settings_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Настройки", menu=settings_menu)
settings_menu.add_command(
    label="Открыть",
    command=lambda: SettingsWindow(root, config_service)
)

MainWindow(root, config_service).pack(padx=20, pady=20)

root.mainloop()
