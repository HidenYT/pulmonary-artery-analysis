import tkinter as tk
import torch
from classification.loader import load_resnet50
from core.config import ConfigService
from segmentation.loader import load_segnet
from ui.main_window import MainWindow
from ui.settings_window import SettingsWindow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = load_resnet50("weights/model_resnet_1stack", device)
segmentator = load_segnet("weights/segnet_aug_dice_weights_50_epoch", device)


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

MainWindow(root, config_service, classifier=classifier, segmentator=segmentator, device=device).pack(padx=20, pady=20)

root.mainloop()
