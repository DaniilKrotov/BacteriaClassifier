import json
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2

# Импортируем функции из ваших файлов
from petri_dish_detector import process_image
from predict_bacteria_grid import (
    load_artifacts,
    grid_extract_features
)
from trainning import (
    collect_dataset,
    train_model,
    save_artifacts,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import numpy as np


APP_TITLE = "Petri Bacteria GUI"



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1120x760")
        self.minsize(980, 680)

        self.train_thread = None
        self.predict_thread = None

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(12, 12, 12, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Классификация бактерий по чашке Петри",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            header,
            text="Обучение модели по датасету и предсказание по изображению через ваши функции.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.tabs = ttk.Notebook(body)
        self.tabs.grid(row=0, column=0, sticky="nsew")

        self.train_tab = ttk.Frame(self.tabs, padding=12)
        self.predict_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.train_tab, text="Обучение")
        self.tabs.add(self.predict_tab, text="Предсказание")

        self._build_train_tab()
        self._build_predict_tab()

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _browse_file(self, var: tk.StringVar, filetypes=None):
        path = filedialog.askopenfilename(filetypes=filetypes or [("Все файлы", "*.*")])
        if path:
            var.set(path)

    def _build_train_tab(self):
        root = self.train_tab
        for i in range(3):
            root.columnconfigure(i, weight=1 if i == 1 else 0)

        self.train_dataset_var = tk.StringVar()
        self.train_out_var = tk.StringVar(value=str(Path.cwd() / "trained_bacteria_model_gui"))
        self.model_type_var = tk.StringVar(value="rf")
        self.grid_rows_var = tk.IntVar(value=6)
        self.grid_cols_var = tk.IntVar(value=6)
        self.size_var = tk.IntVar(value=1024)
        self.fill_black_var = tk.BooleanVar(value=True)
        self.debug_var = tk.BooleanVar(value=False)

        row = 0
        ttk.Label(root, text="Папка датасета").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(root, textvariable=self.train_dataset_var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Выбрать", command=lambda: self._browse_dir(self.train_dataset_var)).grid(row=row, column=2, sticky="ew")

        row += 1
        ttk.Label(root, text="Папка для сохранения модели").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(root, textvariable=self.train_out_var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Выбрать", command=lambda: self._browse_dir(self.train_out_var)).grid(row=row, column=2, sticky="ew")

        params = ttk.LabelFrame(root, text="Параметры обучения", padding=10)
        params.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(12, 6))
        for i in range(4):
            params.columnconfigure(i, weight=1)

        ttk.Label(params, text="Тип модели").grid(row=0, column=0, sticky="w")
        ttk.Combobox(params, textvariable=self.model_type_var, values=["rf", "svm"], state="readonly").grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(2, 8))

        ttk.Label(params, text="Строк сетки").grid(row=0, column=1, sticky="w")
        ttk.Spinbox(params, from_=2, to=20, textvariable=self.grid_rows_var).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(2, 8))

        ttk.Label(params, text="Столбцов сетки").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(params, from_=2, to=20, textvariable=self.grid_cols_var).grid(row=1, column=2, sticky="ew", padx=(0, 8), pady=(2, 8))

        ttk.Label(params, text="Размер нормализ. изображения").grid(row=2, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.size_var).grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(2, 8))


        ttk.Checkbutton(params, text="Закрашивать вне чашки в черный", variable=self.fill_black_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 4))
        ttk.Checkbutton(params, text="Debug режим детектора", variable=self.debug_var).grid(row=4, column=2, columnspan=2, sticky="w", pady=(4, 4))

        actions = ttk.Frame(root)
        actions.grid(row=row + 2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self.train_btn = ttk.Button(actions, text="Обучить и сохранить модель", command=self.start_train)
        self.train_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ttk.Button(actions, text="Открыть папку модели", command=self.open_model_dir_from_train).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _build_predict_tab(self):
        root = self.predict_tab
        for i in range(3):
            root.columnconfigure(i, weight=1 if i == 1 else 0)

        self.predict_image_var = tk.StringVar()
        self.predict_model_dir_var = tk.StringVar()

        row = 0
        ttk.Label(root, text="Изображение").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(root, textvariable=self.predict_image_var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(
            root,
            text="Выбрать",
            command=lambda: self._browse_file(
                self.predict_image_var,
                [("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Все файлы", "*.*")],
            ),
        ).grid(row=row, column=2, sticky="ew")

        row += 1
        ttk.Label(root, text="Папка обученной модели").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(root, textvariable=self.predict_model_dir_var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Выбрать", command=lambda: self._browse_dir(self.predict_model_dir_var)).grid(row=row, column=2, sticky="ew")


        self.predict_btn = ttk.Button(root, text="Сделать предсказание", command=self.start_predict)
        self.predict_btn.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(16, 0))

        result_frame = ttk.LabelFrame(root, text="Результат", padding=10)
        result_frame.grid(row=row + 2, column=0, columnspan=3, sticky="nsew", pady=(16, 0))
        root.rowconfigure(row + 2, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.result_text = tk.Text(result_frame, wrap="word", state="disabled", height=18)
        self.result_text.grid(row=0, column=0, sticky="nsew")

    def log(self, message: str):
        pass

    def set_result(self, message: str):
        def _set():
            self.result_text.configure(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", message)
            self.result_text.configure(state="disabled")
        self.result_text.after(0, _set)

    def format_prediction_result(self, result: dict) -> str:
        best_class = result.get("class", "—")
        best_conf = result.get("confidence_percent")
        top_predictions = result.get("top_predictions", []) or []

        if top_predictions:
            best_class = str(top_predictions[0].get("class", best_class))
            best_conf = top_predictions[0].get("confidence_percent", best_conf)

        lines = ["Самый вероятный класс:", f"  {best_class}"]

        if best_conf is not None:
            lines.append(f"Уверенность: {float(best_conf):.2f}%")
        else:
            lines.append("Уверенность: недоступна")

        if top_predictions:
            lines.append("")
            lines.append("Топ вероятностей:")
            for i, item in enumerate(top_predictions, start=1):
                cls = item.get("class", "—")
                conf = item.get("confidence_percent")
                if conf is None:
                    lines.append(f"{i}. {cls}")
                else:
                    lines.append(f"{i}. {cls} — {float(conf):.2f}%")

        if result.get("proba"):
            lines.append("")
            lines.append("Все классы:")
            sorted_items = sorted(result["proba"].items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_items:
                lines.append(f"- {cls}: {float(prob) * 100.0:.2f}%")

        return "\n".join(lines)

    def open_model_dir_from_train(self):
        model_dir = self.train_out_var.get().strip()
        if model_dir:
            self.predict_model_dir_var.set(model_dir)
            self.tabs.select(self.predict_tab)

    def start_train(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("Обучение", "Обучение уже выполняется.")
            return

        dataset_dir = self.train_dataset_var.get().strip()
        out_dir = self.train_out_var.get().strip()

        if not dataset_dir:
            messagebox.showerror("Ошибка", "Выберите папку датасета.")
            return
        if not out_dir:
            messagebox.showerror("Ошибка", "Выберите папку для сохранения модели.")
            return

        self.train_btn.configure(state="disabled")
        self.log("\n=== Запуск обучения ===\n")

        self.train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.train_thread.start()

    def _train_worker(self):
        try:
            dataset_dir = self.train_dataset_var.get().strip()
            out_dir = self.train_out_var.get().strip()
            model_type = self.model_type_var.get().strip()
            grid_rows = int(self.grid_rows_var.get())
            grid_cols = int(self.grid_cols_var.get())
            test_size = 0.2
            seed = 42
            cfg = {
                "output_size": int(self.size_var.get()),
                "padding": 0.0,
                "fill_black": bool(self.fill_black_var.get()),
                "debug": bool(self.debug_var.get()),
            }

            self.log(f"Датасет: {dataset_dir}\n")
            self.log(f"Сохранение модели: {out_dir}\n")
            self.log(f"Параметры: model={model_type}, grid={grid_rows}x{grid_cols}, test_size={test_size}, seed={seed}\n")

            X, y, paths, class_names = collect_dataset(
                dataset_dir=dataset_dir,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                cfg=cfg,
            )

            self.log(f"Собрано образцов: {len(X)}\n")
            self.log(f"Размер вектора признаков: {X.shape[1]}\n")

            class_counts = Counter(y.tolist())
            n_samples = len(y)
            n_classes = len(class_counts)

            if n_classes < 2:
                raise ValueError("Для обучения нужно минимум 2 класса с образцами.")

            self.log("Количество примеров по классам:\n")
            for cls, cnt in sorted(class_counts.items()):
                self.log(f"  {cls}: {cnt}\n")

            min_count = min(class_counts.values())
            stratify = y if min_count >= 2 else None
            if stratify is None:
                self.log("[WARN] Stratified split отключен: есть классы с < 2 примерами.\n")

            if stratify is not None:
                n_test = int(np.ceil(n_samples * test_size))
                if n_test < n_classes:
                    min_test_float = n_classes / n_samples
                    safe_test_size = max(test_size, min_test_float)
                    safe_test_size = min(0.5, max(safe_test_size + 1e-6, test_size))
                    self.log(
                        f"[WARN] test_size слишком мал для stratify: {test_size} -> {safe_test_size:.3f}\n"
                    )
                    test_size = safe_test_size

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=seed,
                stratify=stratify,
            )

            self.log(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

            model = train_model(X_train, y_train, model_type=model_type, random_state=seed)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, digits=4)
            cm = confusion_matrix(y_test, y_pred)

            self.log("\n=== RESULT ===\n")
            self.log(f"Accuracy: {acc:.4f}\n")
            self.log("Classification report:\n")
            self.log(report + "\n")
            self.log("Confusion matrix:\n")
            self.log(str(cm) + "\n")

            meta = {
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "feature_version": "grid_patch_v1",
                "petri_cfg": cfg,
                "classes_seen": sorted(list(set(y.tolist()))),
                "model_type": model_type,
                "feature_dim": int(X.shape[1]),
            }
            save_artifacts(out_dir, model, meta)

            self.log("\n[OK] Модель успешно обучена и сохранена.\n")
            self.predict_model_dir_var.set(out_dir)
            self.tabs.after(0, lambda: self.tabs.select(self.predict_tab))
            self.after(0, lambda: messagebox.showinfo("Готово", "Обучение завершено. Модель сохранена."))

        except Exception as e:
            err = f"\n[ERR] Ошибка обучения: {e}\n{traceback.format_exc()}\n"
            self.log(err)
            self.after(0, lambda: messagebox.showerror("Ошибка обучения", str(e)))
        finally:
            self.after(0, lambda: self.train_btn.configure(state="normal"))

    def start_predict(self):
        if self.predict_thread and self.predict_thread.is_alive():
            messagebox.showwarning("Предсказание", "Предсказание уже выполняется.")
            return

        image_path = self.predict_image_var.get().strip()
        model_dir = self.predict_model_dir_var.get().strip()

        if not image_path:
            messagebox.showerror("Ошибка", "Выберите изображение.")
            return
        if not model_dir:
            messagebox.showerror("Ошибка", "Выберите папку обученной модели.")
            return

        self.predict_btn.configure(state="disabled")
        self.log("\n=== Запуск предсказания ===\n")
        self.predict_thread = threading.Thread(target=self._predict_worker, daemon=True)
        self.predict_thread.start()

    def _predict_worker(self):
        try:
            image_path = self.predict_image_var.get().strip()
            model_dir = self.predict_model_dir_var.get().strip()
            self.log(f"Изображение: {image_path}\n")
            self.log(f"Модель: {model_dir}\n")

            model, meta = load_artifacts(model_dir)
            cfg = meta["petri_cfg"]
            grid_rows = int(meta["grid_rows"])
            grid_cols = int(meta["grid_cols"])

            norm, dbg, geom = process_image(image_path, cfg)
            if norm is None:
                raise RuntimeError("Чашка Петри не найдена на изображении")

            feats = grid_extract_features(norm, grid_rows=grid_rows, grid_cols=grid_cols).reshape(1, -1)
            pred = model.predict(feats)[0]

            result = {"class": str(pred)}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feats)[0]
                classes = model.classes_
                proba_dict = {str(c): float(p) for c, p in zip(classes, proba)}
                result["proba"] = proba_dict
                result["confidence_percent"] = round(proba_dict[str(pred)] * 100.0, 2)
                top_n = min(3, len(classes))
                top_idx = np.argsort(proba)[::-1][:top_n]
                result["top_predictions"] = [
                    {
                        "class": str(classes[i]),
                        "probability": float(proba[i]),
                        "confidence_percent": round(float(proba[i]) * 100.0, 2),
                    }
                    for i in top_idx
                ]
            else:
                result["confidence_percent"] = None
                result["top_predictions"] = []


            pretty = json.dumps(result, ensure_ascii=False, indent=2)
            formatted_result = self.format_prediction_result(result)
            self.set_result(formatted_result)
            self.log("Результат предсказания:\n")
            self.log(formatted_result + "\n\n")
            self.log("JSON детали:\n")
            self.log(pretty + "\n")

        except Exception as e:
            err = f"\n[ERR] Ошибка предсказания: {e}\n{traceback.format_exc()}\n"
            self.log(err)
            self.set_result(f"Ошибка: {e}")
            self.after(0, lambda: messagebox.showerror("Ошибка предсказания", str(e)))
        finally:
            self.after(0, lambda: self.predict_btn.configure(state="normal"))


if __name__ == "__main__":
    app = App()
    app.mainloop()
