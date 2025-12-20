import tkinter as tk
from core.config import Config, ConfigService

class SettingsWindow(tk.Toplevel):
    def __init__(self, master, config_service: ConfigService):
        super().__init__(master)
        self._config_service = config_service
        self.entries = {}
        config = config_service.load().model_dump(exclude_computed_fields=True)

        for key, value in config.items():
            tk.Label(self, text=key).pack()
            entry = tk.Entry(self)
            entry.insert(0, str(value))
            entry.pack()
            self.entries[key] = entry

        tk.Button(self, text="Сохранить", command=self.save).pack(pady=10)

    def save(self):
        entries_vals = {k: str(e.get()) for k, e in self.entries.items()}
        cfg = Config(**entries_vals)
        self._config_service.save(cfg)
        self.destroy()
