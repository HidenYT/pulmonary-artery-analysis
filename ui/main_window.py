import tkinter as tk
from tkinter import filedialog, messagebox
import os

from core.config import ConfigService
from core.processor import run_processing
from report_generation.report import generate_pdf

class MainWindow(tk.Frame):
    def __init__(self, master, config_service: ConfigService):
        super().__init__(master)
        self._image_path = None
        self._mask_path = None
        self._report_path = None
        self._config_service = config_service

        tk.Button(self, text="Выбрать КТ снимок", command=self._select_image).pack(pady=5)
        tk.Button(self, text="Выбрать маску", command=self._select_mask).pack(pady=5)
        tk.Button(self, text="Сохранить в", command=self._select_report_save_path).pack(pady=5)
        tk.Button(self, text="Запуск", command=self.run).pack(pady=20)

    def _select_image(self):
        self._image_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg")]
        )

    def _select_mask(self):
        self._mask_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg")]
        )

    def _select_report_save_path(self):
        self._report_path = filedialog.asksaveasfilename(
            filetypes=[("PDF", "*.pdf")],
            defaultextension=".pdf"
        )

    def run(self):
        if not self._validate_all_paths_filled():
            return
        
        config = self._config_service.load()
        result = run_processing(self._image_path, self._mask_path, config)

        generate_pdf(result, self._report_path, config)

        os.startfile(self._report_path)
    
    def _validate_all_paths_filled(self) -> bool:
        error_paths = []
        if not self._image_path:
            error_paths.append("изображение")
        if not self._mask_path:
            error_paths.append("маска")
        if not self._report_path:
            error_paths.append("отчёт")
        if not error_paths:
            return True
        if len(error_paths) == 1:
            error_paths_msg = error_paths[-1]
        else:
            error_paths_msg = f'{", ".join(error_paths[:-1])} и {error_paths[-1]}'
        messagebox.showerror("Ошибка", f"Выберите {error_paths_msg}")
        return False
