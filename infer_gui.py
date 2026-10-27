import json
import os
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "gui_settings.json")
NO_LANG = "(none)"


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def read_languages(config_path):
    """Read langs.txt ("name,id" per line) from the model folder named by the config."""
    try:
        import yaml

        config_path = resolve(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        save_dir = cfg["output"]["save_dir"]
        config_dir = os.path.dirname(os.path.abspath(config_path))
        lang_file = None
        for cand in (resolve(save_dir), os.path.join(config_dir, save_dir), config_dir):
            if os.path.isfile(os.path.join(cand, "langs.txt")):
                lang_file = os.path.join(cand, "langs.txt")
                break
        if lang_file is None:
            return []
        langs = []
        with open(lang_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    name, lid = line.rsplit(",", 1)
                    if lid.strip().lstrip("-").isdigit():
                        langs.append((name.strip(), lid.strip()))
        return langs
    except Exception:
        return []


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WFL-ASR Inference")
        self.proc = None
        self.stopped = False

        self.vars = {
            "input": tk.StringVar(),
            "ckpt": tk.StringVar(value="oomfmodelwfl/latest-v1.ckpt"),
            "cfg": tk.StringVar(value="oomfmodelwfl/config.yaml"),
            "lang": tk.StringVar(),
            "decoder": tk.StringVar(value="viterbi"),
            "bias": tk.StringVar(value="5"),
            "no_offset": tk.BooleanVar(value=False),
            "sil_ph": tk.StringVar(value="SP"),
            "sil_thr": tk.StringVar(value="0.005"),
            "sil_dur": tk.StringVar(value="0.5"),
            "use_lab": tk.BooleanVar(value=False),
            "strict_fa": tk.BooleanVar(value=True),
            "alias": tk.StringVar(),
        }
        self.lang_display = tk.StringVar(value=NO_LANG)
        self.lang_map = {}
        self._lang_job = None
        self.load_settings()
        self.setup_style()
        self.build_ui()
        self.vars["cfg"].trace_add("write", lambda *_: self.schedule_lang_refresh())
        self.refresh_languages()
        self.dark_titlebar()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- setup ----------
    def setup_style(self):
        BG, FIELD, BTN, BTN_HOVER = "#1e1f22", "#2b2d31", "#383a40", "#43464d"
        FG, MUTED, BORDER, ACCENT = "#e3e5e8", "#949ba4", "#3f4147", "#4f7fe0"
        self.BG = BG
        self.configure(bg=BG)

        # dropdown list of the comboboxes (plain Tk widget, styled via the option database)
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.option_add("*TCombobox*Listbox.borderWidth", 0)

        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, fieldbackground=FIELD, troughcolor=BG,
                     bordercolor=BORDER, lightcolor=BG, darkcolor=BG, focuscolor=BG,
                     insertcolor=FG, selectbackground=ACCENT, selectforeground="#ffffff")
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Muted.TLabel", foreground=MUTED)
        self.bold = tkfont.nametofont("TkDefaultFont").copy()
        self.bold.configure(weight="bold")
        st.configure("Heading.TLabel", font=self.bold)
        st.configure("TSeparator", background=BORDER)

        st.configure("TCheckbutton", background=BG, foreground=FG, indicatorbackground=FIELD,
                     indicatorforeground="#ffffff", bordercolor=BORDER)
        st.map("TCheckbutton", background=[("active", BG)],
               indicatorbackground=[("selected", ACCENT), ("!selected", FIELD)])

        st.configure("TEntry", fieldbackground=FIELD, foreground=FG, bordercolor=BORDER,
                     lightcolor=FIELD, darkcolor=FIELD, padding=4)
        st.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)],
               darkcolor=[("focus", ACCENT)])

        st.configure("TCombobox", fieldbackground=FIELD, background=BTN, foreground=FG, arrowcolor=FG,
                     bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD, padding=4,
                     selectbackground=FIELD, selectforeground=FG)
        st.map("TCombobox",
               fieldbackground=[("readonly", FIELD), ("disabled", "#25262a")],
               foreground=[("disabled", MUTED)],
               background=[("active", BTN_HOVER), ("disabled", "#25262a")],
               arrowcolor=[("disabled", MUTED)],
               selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", FG)])

        st.configure("TButton", background=BTN, foreground=FG, bordercolor=BORDER, lightcolor=BTN,
                     darkcolor=BTN, padding=(10, 4), relief="flat")
        st.map("TButton", background=[("active", BTN_HOVER), ("disabled", "#2a2b2f")],
               foreground=[("disabled", "#6b6e75")], bordercolor=[("disabled", "#2f3034")])
        st.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT,
                     lightcolor=ACCENT, darkcolor=ACCENT)
        st.map("Accent.TButton", background=[("active", "#6590ea"), ("disabled", "#2f3f66")],
               foreground=[("disabled", "#8d98b5")], bordercolor=[("disabled", "#2f3f66")],
               lightcolor=[("active", "#6590ea"), ("disabled", "#2f3f66")],
               darkcolor=[("active", "#6590ea"), ("disabled", "#2f3f66")])

        st.configure("Vertical.TScrollbar", background=BTN, troughcolor=BG, bordercolor=BG,
                     lightcolor=BTN, darkcolor=BTN, arrowcolor=FG, gripcount=0)
        st.map("Vertical.TScrollbar", background=[("active", BTN_HOVER)])

    def dark_titlebar(self):
        """Best effort: dark title bar on Windows 10/11."""
        if os.name != "nt":
            return
        try:
            import ctypes

            self.update()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            value = ctypes.c_int(1)
            for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (new / old value)
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                    break
            # nudge the window so Windows repaints the title bar
            self.withdraw()
            self.deiconify()
        except Exception:
            pass

    def section(self, parent, title):
        head = ttk.Frame(parent)
        head.pack(fill="x", pady=(14, 6))
        ttk.Label(head, text=title, style="Heading.TLabel").pack(side="left")
        ttk.Separator(head).pack(side="left", fill="x", expand=True, padx=(10, 0))
        body = ttk.Frame(parent)
        body.pack(fill="x")
        return body

    def build_ui(self):
        root = ttk.Frame(self, padding=(16, 2, 16, 12))
        root.pack(fill="both", expand=True)

        files = self.section(root, "Files")
        files.columnconfigure(1, weight=1)
        self.file_row(files, 0, "Input", "input", None)
        ttk.Button(files, text="Folder", width=10, command=self._pick_folder).grid(row=0, column=2, padx=(0, 4), pady=3)
        ttk.Button(files, text="WAV file", width=10, command=self._pick_wav).grid(row=0, column=3, pady=3)
        self.file_row(files, 1, "Checkpoint", "ckpt",
                      lambda: self.browse_file("ckpt", [("Checkpoint", "*.ckpt *.pt"), ("All files", "*.*")]))
        self.file_row(files, 2, "Config", "cfg",
                      lambda: self.browse_file("cfg", [("YAML", "*.yaml *.yml"), ("All files", "*.*")]))

        opt = self.section(root, "Settings")
        self.lang_box = ttk.Combobox(opt, textvariable=self.lang_display, width=14, state="readonly")
        self.lang_box.bind("<<ComboboxSelected>>", self.on_lang_selected)
        self.opt_row(opt, 0, 0, "Language", self.lang_box)
        self.opt_row(opt, 0, 2, "Decoder", ttk.Combobox(opt, textvariable=self.vars["decoder"],
                     values=["viterbi", "constrained"], width=12, state="readonly"))
        self.opt_row(opt, 0, 4, "Viterbi bias", ttk.Entry(opt, textvariable=self.vars["bias"], width=8))
        self.opt_row(opt, 1, 0, "Silence phoneme", ttk.Entry(opt, textvariable=self.vars["sil_ph"], width=8))
        self.opt_row(opt, 1, 2, "Silence threshold", ttk.Entry(opt, textvariable=self.vars["sil_thr"], width=10))
        self.opt_row(opt, 1, 4, "Min silence (s)", ttk.Entry(opt, textvariable=self.vars["sil_dur"], width=8))
        ttk.Checkbutton(opt, text="Disable offset refinement", variable=self.vars["no_offset"]).grid(
            row=2, column=0, columnspan=6, sticky="w", pady=(10, 0))

        fa = self.section(root, "Forced alignment")
        fa.columnconfigure(3, weight=1)
        self.use_lab_chk = ttk.Checkbutton(
            fa, text="Use existing .lab files as transcripts", variable=self.vars["use_lab"])
        self.use_lab_chk.grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(fa, text="Each .lab is read for its phoneme column, then overwritten with the aligned result "
                           "(original saved as .lab.bak).\nA .txt next to the wav takes priority over the .lab.",
                  style="Muted.TLabel", justify="left").grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))
        ttk.Checkbutton(fa, text="Strict (align exactly the given phonemes, insert nothing extra)",
                        variable=self.vars["strict_fa"]).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(fa, text="Aliases").grid(row=3, column=0, sticky="w", pady=(8, 0), padx=(0, 10))
        ttk.Entry(fa, textvariable=self.vars["alias"]).grid(row=3, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(fa, text="e.g.  br=AP  cl=SP   (pau and sil already map to the silence phoneme)",
                  style="Muted.TLabel").grid(row=4, column=1, columnspan=3, sticky="w")

        bar = ttk.Frame(root)
        bar.pack(fill="x", pady=(16, 0))
        self.run_btn = ttk.Button(bar, text="Run", width=10, style="Accent.TButton", command=self.run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", width=10, command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Open folder", command=self.open_folder).pack(side="right")

        log_frame = ttk.Frame(root)
        log_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.log = tk.Text(log_frame, height=11, width=70, state="disabled", wrap="word", font=("Consolas", 9),
                           bg="#16171a", fg="#d4d4d4", relief="flat", padx=8, pady=6, highlightthickness=1,
                           highlightbackground="#3f4147", highlightcolor="#3f4147")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("error", foreground="#f48771")
        self.log.tag_configure("info", foreground="#6cb6ff")

        status = ttk.Frame(root)
        status.pack(fill="x", pady=(8, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var, style="Muted.TLabel").pack(side="left")
        ttk.Button(status, text="Clear log", command=self.clear_log).pack(side="right")

        self.update_idletasks()
        self.minsize(self.winfo_reqwidth(), self.winfo_reqheight())

    def file_row(self, parent, r, label, key, cmd):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 10))
        ttk.Entry(parent, textvariable=self.vars[key]).grid(row=r, column=1, sticky="ew", padx=(0, 8), pady=3)
        if cmd:
            ttk.Button(parent, text="Browse", command=cmd).grid(row=r, column=2, columnspan=2, sticky="ew", pady=3)

    def opt_row(self, parent, r, c, label, widget):
        pady = (0, 0) if r == 0 else (8, 0)
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", pady=pady, padx=(0 if c == 0 else 20, 8))
        widget.grid(row=r, column=c + 1, sticky="w", pady=pady)

    # ---------- languages ----------
    def schedule_lang_refresh(self):
        if self._lang_job:
            self.after_cancel(self._lang_job)
        self._lang_job = self.after(400, self.refresh_languages)

    def refresh_languages(self):
        self._lang_job = None
        langs = read_languages(self.vars["cfg"].get().strip())
        self.lang_map = {NO_LANG: ""}
        for name, lid in langs:
            self.lang_map[f"{name} ({lid})"] = lid
        self.lang_box["values"] = list(self.lang_map)
        if not langs:
            self.lang_map = {}
            self.lang_display.set("(no langs.txt found)")
            self.lang_box.configure(state="disabled")
            return
        self.lang_box.configure(state="readonly")
        saved = self.vars["lang"].get().strip()
        current = next((d for d, i in self.lang_map.items() if i == saved and i), NO_LANG)
        self.lang_display.set(current)

    def on_lang_selected(self, _event=None):
        self.vars["lang"].set(self.lang_map.get(self.lang_display.get(), ""))

    def selected_lang_id(self):
        return self.lang_map.get(self.lang_display.get(), "")

    # ---------- browsing ----------
    def _pick_folder(self):
        path = filedialog.askdirectory(title="Select folder with .wav files")
        if path:
            self.vars["input"].set(path)

    def _pick_wav(self):
        path = filedialog.askopenfilename(title="Select .wav file", filetypes=[("WAV", "*.wav")])
        if path:
            self.vars["input"].set(path)

    def browse_file(self, key, types):
        path = filedialog.askopenfilename(filetypes=types)
        if path:
            self.vars[key].set(path)

    def open_folder(self):
        path = self.vars["input"].get().strip()
        if not path:
            return
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if not os.path.isdir(folder):
            return
        if os.name == "nt":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # ---------- log ----------
    def write(self, text, tag=None):
        self.log.configure(state="normal")
        if tag is None and ("Error" in text or "Traceback" in text):
            tag = "error"
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ---------- running ----------
    def validate(self):
        v = {k: x.get().strip() for k, x in self.vars.items() if isinstance(x, tk.StringVar)}
        if not v["input"] or not os.path.exists(v["input"]):
            return "Please choose a valid input folder or .wav file."
        for key, name in (("ckpt", "checkpoint"), ("cfg", "config")):
            path = v[key] if os.path.isabs(v[key]) else os.path.join(HERE, v[key])
            if not os.path.isfile(path):
                return f"The {name} file was not found:\n{v[key]}"
        for key, name in (("bias", "Viterbi bias"), ("sil_thr", "Silence threshold"), ("sil_dur", "Min silence")):
            if v[key]:
                try:
                    float(v[key])
                except ValueError:
                    return f"{name} must be a number."
        for item in v["alias"].replace(",", " ").split():
            if "=" not in item:
                return f"Each alias must look like old=new, got: {item}"
        return None

    def run(self):
        error = self.validate()
        if error:
            messagebox.showwarning("Check your settings", error)
            return
        v = {k: x.get().strip() if isinstance(x.get(), str) else x.get() for k, x in self.vars.items()}
        cmd = [
            sys.executable, "-u", os.path.join(HERE, "infer.py"),
            "-i", v["input"],
            "-ckpt", v["ckpt"],
            "-c", v["cfg"],
            "-d", v["decoder"],
            "--viterbi-bias", v["bias"] or "5",
            "--silence-phoneme", v["sil_ph"] or "SP",
            "--silence-threshold", v["sil_thr"] or "0.005",
            "--min-silence-duration", v["sil_dur"] or "0.5",
        ]
        lang_id = self.selected_lang_id()
        if lang_id:
            cmd += ["-l", lang_id]
        if v["no_offset"]:
            cmd.append("--no_use_offset")
        if v["use_lab"]:
            cmd.append("--use-lab")
        cmd.append("--strict-fa" if v["strict_fa"] else "--no-strict-fa")
        for item in str(v["alias"]).replace(",", " ").split():
            cmd += ["--alias", item]

        self.save_settings()
        self.stopped = False
        self.write("> " + subprocess.list2cmdline(cmd) + "\n", "info")
        self.set_running(True)
        threading.Thread(target=self.worker, args=(cmd,), daemon=True).start()

    def worker(self, cmd):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        code = None
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", creationflags=flags,
            )
            for line in self.proc.stdout:
                self.after(0, self.write, line)
            code = self.proc.wait()
        except Exception as e:
            self.after(0, self.write, f"\nFailed to start: {e}\n", "error")
        finally:
            self.after(0, self.done, code)

    def done(self, code):
        self.proc = None
        self.set_running(False)
        if self.stopped:
            self.status_var.set("Stopped")
            self.write("\nStopped.\n", "info")
        elif code == 0:
            self.status_var.set("Finished")
            self.write("\nFinished.\n", "info")
        else:
            self.status_var.set("Finished with errors" if code is not None else "Failed")
            self.write(f"\nProcess ended (exit code {code}).\n", "error")

    def stop(self):
        if self.proc:
            self.stopped = True
            self.proc.terminate()

    def set_running(self, running):
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        if running:
            self.status_var.set("Running...")

    # ---------- settings ----------
    def load_settings(self):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, var in self.vars.items():
                if key in data:
                    var.set(data[key])
        except (OSError, ValueError):
            pass

    def save_settings(self):
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({k: v.get() for k, v in self.vars.items()}, f, indent=2)
        except OSError:
            pass

    def on_close(self):
        self.save_settings()
        if self.proc:
            self.proc.terminate()
        self.destroy()


if __name__ == "__main__":
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().mainloop()
