import tkinter as tk
from core.config import Config, ConfigService

class SettingsWindow(tk.Toplevel):
    def __init__(self, master, config_service: ConfigService):
        super().__init__(master)
        self._config_service = config_service
        self.entries = {}
        config_model = config_service.load()
        config_dict = config_model.model_dump(exclude_computed_fields=True)

        for key, value in config_dict.items():
            field_name = key
            description = Config.model_fields[key].description
            if description is not None:
                field_name = description
            tk.Label(self, text=field_name).pack()
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
