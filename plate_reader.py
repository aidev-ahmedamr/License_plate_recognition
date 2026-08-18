"""
plate_reader.py — Desktop App
================================
Tkinter GUI over pipeline.py. Image mode runs the pipeline once per frame;
video/camera mode runs it through the multi-frame tracker so each physical
plate gets one voted-on reading instead of flickering per-frame guesses.

Run:
    pip install -r requirements.txt
    python plate_reader.py
"""

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import pipeline

SCAN_INTERVAL_SECONDS = 0.3
PREVIEW_MAX_W, PREVIEW_MAX_H = 760, 430


class PlateReaderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Plate Reader — License Plate Recognition")
        self.root.configure(bg="#14161a")
        self.root.geometry("1000x700")
        self.root.minsize(880, 620)

        self._cap = None
        self._running = False
        self._stop_flag = threading.Event()
        self._reported_ids = set()

        self._build_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- style / layout (same look as before) ----------------
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg, surface, yellow, off_white = "#14161a", "#21252c", "#f2c230", "#edebe3"
        style.configure("TFrame", background=bg)
        style.configure("Surface.TFrame", background=surface)
        style.configure("TLabel", background=bg, foreground=off_white, font=("Segoe UI", 10))
        style.configure("Heading.TLabel", background=bg, foreground=off_white, font=("Segoe UI Semibold", 15))
        style.configure("Status.TLabel", background=bg, foreground="#9a978d", font=("Consolas", 9))
        style.configure("Accent.TButton", background=yellow, foreground=bg, font=("Segoe UI Semibold", 10), padding=8)
        style.map("Accent.TButton", background=[("active", "#e0ac1f")])
        style.configure("Secondary.TButton", background=surface, foreground=off_white, font=("Segoe UI", 10), padding=8)
        style.map("Secondary.TButton", background=[("active", "#2a2f37")])
        style.configure("Results.Treeview", background=surface, fieldbackground=surface,
                         foreground=off_white, rowheight=26, font=("Consolas", 10), borderwidth=0)
        style.configure("Results.Treeview.Heading", background="#2a2f37", foreground=yellow, font=("Segoe UI Semibold", 9))
        style.map("Results.Treeview", background=[("selected", "#33383f")])

    def _build_layout(self):
        top = ttk.Frame(self.root, padding=(20, 16))
        top.pack(fill="x")
        ttk.Label(top, text="PLATE READER", style="Heading.TLabel").pack(side="left")
        self.status_label = ttk.Label(top, text="Idle", style="Status.TLabel")
        self.status_label.pack(side="right")

        controls = ttk.Frame(self.root, padding=(20, 0))
        controls.pack(fill="x")
        ttk.Button(controls, text="Open Image", style="Accent.TButton", command=self.open_image).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Open Video", style="Secondary.TButton", command=self.open_video).pack(side="left", padx=8)
        self.camera_btn = ttk.Button(controls, text="Start Camera", style="Secondary.TButton", command=self.toggle_camera)
        self.camera_btn.pack(side="left", padx=8)
        ttk.Button(controls, text="Clear Results", style="Secondary.TButton", command=self.clear_results).pack(side="left", padx=8)

        body = ttk.Frame(self.root, padding=20)
        body.pack(fill="both", expand=True)

        preview_frame = ttk.Frame(body, style="Surface.TFrame")
        preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 16))
        self.preview_label = tk.Label(preview_frame, bg="#000000", text="No media loaded", fg="#6c8a9a", font=("Segoe UI", 11))
        self.preview_label.pack(fill="both", expand=True, padx=1, pady=1)

        results_frame = ttk.Frame(body, width=340)
        results_frame.pack(side="right", fill="y")
        results_frame.pack_propagate(False)
        ttk.Label(results_frame, text="Detected plates", style="TLabel").pack(anchor="w", pady=(0, 8))

        self.tree = ttk.Treeview(results_frame, columns=("plate", "conf", "valid"), show="headings",
                                  style="Results.Treeview", height=20)
        self.tree.heading("plate", text="PLATE")
        self.tree.heading("conf", text="CONF %")
        self.tree.heading("valid", text="FORMAT")
        self.tree.column("plate", width=170)
        self.tree.column("conf", width=70, anchor="center")
        self.tree.column("valid", width=70, anchor="center")
        self.tree.pack(fill="both", expand=True)

    # ---------------- helpers ----------------
    def set_status(self, text):
        self.status_label.config(text=text)
        self.root.update_idletasks()

    def clear_results(self):
        self._reported_ids.clear()
        for row in self.tree.get_children():
            self.tree.delete(row)

    def add_result_row(self, text, conf, valid, row_key=None):
        """row_key lets us update a row in place (used by tracker output);
        omit it for one-off image-mode results."""
        for row in self.tree.get_children():
            if self.tree.item(row)["tags"] and self.tree.item(row)["tags"][0] == str(row_key):
                self.tree.delete(row)
        self.tree.insert("", 0, values=(text, f"{round(conf * 100)}", "✓" if valid else "—"),
                          tags=(str(row_key),) if row_key is not None else ())

    def show_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        w, h = img.size
        scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=photo, text="")
        self.preview_label.image = photo

    # ---------------- image mode (single-pass, no tracking needed) ----------------
    def open_image(self):
        self._stop_active_loop()
        path = filedialog.askopenfilename(title="Choose an image", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Could not read that image file.")
            return
        self.clear_results()
        self.set_status("Scanning image…")
        threading.Thread(target=self._process_image_thread, args=(frame,), daemon=True).start()

    def _process_image_thread(self, frame):
        try:
            results, annotated = pipeline.process_frame(frame)
        except Exception as exc:
            error_message = str(exc)
            self.root.after(0, lambda msg=error_message: messagebox.showerror("Pipeline error", msg))
            self.root.after(0, lambda: self.set_status("Error"))
            return

        def finish():
            self.show_frame(annotated)
            for r in results:
                self.add_result_row(r["plate_text"], r["ocr_confidence"], r["format_valid"])
            self.set_status(f"Done — {len(results)} plate(s) found")

        self.root.after(0, finish)

    # ---------------- video mode (tracked) ----------------
    def open_video(self):
        self._stop_active_loop()
        path = filedialog.askopenfilename(title="Choose a video", filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv *.webm")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open that video file.")
            return
        self.clear_results()
        self._cap = cap
        self._running = True
        self._stop_flag.clear()
        self.set_status("Scanning video…")
        threading.Thread(target=self._tracked_loop, args=(cap, False), daemon=True).start()

    # ---------------- camera mode (tracked, live) ----------------
    def toggle_camera(self):
        if self._running and self._cap is not None:
            self._stop_active_loop()
            self.camera_btn.config(text="Start Camera")
            self.set_status("Camera stopped")
            return
        self._stop_active_loop()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the camera.")
            return
        self.clear_results()
        self._cap = cap
        self._running = True
        self._stop_flag.clear()
        self.camera_btn.config(text="Stop Camera")
        self.set_status("Camera live — tracking plates…")
        threading.Thread(target=self._tracked_loop, args=(cap, True), daemon=True).start()

    def _tracked_loop(self, cap, is_camera: bool):
        tracker = pipeline.PlateTracker()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        stride = max(1, int(fps * SCAN_INTERVAL_SECONDS)) if not is_camera else None
        last_scan = 0.0
        frame_idx = 0

        while not self._stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            do_scan = (frame_idx % stride == 0) if stride else (time.time() - last_scan >= SCAN_INTERVAL_SECONDS)

            if do_scan:
                last_scan = time.time()
                try:
                    confirmed_tracks, annotated = pipeline.process_frame_tracked(frame, tracker)
                except Exception:
                    confirmed_tracks, annotated = [], frame
                for t in confirmed_tracks:
                    text, agreement = t.best_text
                    self.root.after(
                        0, lambda tid=t.track_id, tx=text, ag=agreement: self.add_result_row(tx, ag, True, row_key=tid)
                    )
            else:
                annotated = frame

            self.root.after(0, lambda a=annotated: self.show_frame(a))
            frame_idx += 1

        cap.release()
        self._running = False
        if not is_camera:
            self.root.after(0, lambda: self.set_status("Video finished"))

    # ---------------- lifecycle ----------------
    def _stop_active_loop(self):
        if self._running:
            self._stop_flag.set()
            self._running = False
            time.sleep(0.05)

    def _on_close(self):
        self._stop_active_loop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PlateReaderApp(root)
    root.mainloop()
