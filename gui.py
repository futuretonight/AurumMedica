# gui.py
import customtkinter as ctk
import subprocess
import threading
import queue
import os
import sys
import time
import re
from datetime import datetime
import webbrowser

# Improved Process Thread with error handling


class ProcessThread(threading.Thread):
    def __init__(self, command_args, queue):
        super().__init__()
        self.command_args = command_args
        self.queue = queue
        self.daemon = True

    def run(self):
        try:
            command = [sys.executable] + self.command_args
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            self.queue.put(("SYSTEM", f"Executing: {' '.join(command)}"))

            with subprocess.Popen(command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True,
                                  encoding='utf-8',
                                  bufsize=1,
                                  creationflags=creation_flags) as process:

                for line in iter(process.stdout.readline, ''):
                    self.queue.put(("PROCESS_LOG", line.strip()))

                process.wait()

        except Exception as e:
            self.queue.put(("SYSTEM", f"ERROR in process thread: {str(e)}"))
        finally:
            self.queue.put(("SYSTEM", "---PROCESS-COMPLETE---"))

# Fixed Log Monitor Thread


class LogMonitorThread(threading.Thread):
    def __init__(self, filepath, queue, is_primary_log=False):
        super().__init__()
        self.filepath = filepath
        self.queue = queue
        self.is_primary_log = is_primary_log
        self.daemon = True
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        self.queue.put(
            ("SYSTEM", f"Monitoring: {os.path.basename(self.filepath)}"))

        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'w') as f:
                    f.write("")  # Clear log on start
            except Exception as e:
                self.queue.put(
                    ("SYSTEM", f"Warning: Could not clear log - {str(e)}"))

        last_pos = 0
        while not self._stop_event.is_set():
            try:
                if not os.path.exists(self.filepath):
                    time.sleep(0.5)
                    continue

                with open(self.filepath, 'r', encoding='utf-8') as f:
                    f.seek(last_pos)
                    line = f.readline()

                    while line:
                        log_type = "PRIMARY_LOG" if self.is_primary_log else "TELEMETRY"
                        self.queue.put((log_type, line.strip()))
                        last_pos = f.tell()
                        line = f.readline()

                    time.sleep(0.1)

            except Exception as e:
                self.queue.put(("SYSTEM", f"Log monitor error: {str(e)}"))
                time.sleep(1)

# Robust Tooltip Manager


class TooltipManager:
    _active_tooltips = []

    @classmethod
    def create_tooltip(cls, widget, text):
        def enter(event):
            cls._cleanup()
            tip = ctk.CTkToplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(
                f"+{widget.winfo_rootx() + 25}+{widget.winfo_rooty() + 25}")

            label = ctk.CTkLabel(tip, text=text,
                                 corner_radius=4,
                                 fg_color="#FFFFE0",
                                 text_color="#000000",
                                 padx=8, pady=4,
                                 wraplength=300)
            label.pack()
            cls._active_tooltips.append(tip)

        def leave(event):
            cls._cleanup()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
        widget.bind("<Destroy>", lambda e: cls._cleanup())

    @classmethod
    def _cleanup(cls):
        for tip in cls._active_tooltips:
            try:
                tip.destroy()
            except:
                pass
        cls._active_tooltips = []


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MedAI - Diagnostic Assistant")
        self.geometry("1400x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Layout configuration
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Process tracking
        self.process_queue = queue.Queue()
        self.active_log_monitors = {}
        self.process_history = []

        self._setup_ui()
        self.after(100, self.process_gui_queue)

    def _setup_ui(self):
        """Initialize all UI components"""
        # Control Panel
        control_frame = ctk.CTkFrame(self, width=300, corner_radius=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Log Console
        self.log_frame = ctk.CTkFrame(self, corner_radius=10)
        self.log_frame.grid(row=0, column=1, padx=0, pady=10, sticky="nsew")
        self.log_frame.grid_rowconfigure(0, weight=1)

        # Status Panel
        status_frame = ctk.CTkFrame(self, width=300, corner_radius=10)
        status_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        self._build_control_panel(control_frame)
        self._build_log_console()
        self._build_status_panel(status_frame)

    def _build_control_panel(self, parent):
        """Construct the control panel with tooltips"""
        ctk.CTkLabel(parent, text="COMMAND CENTER",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 15))

        # Knowledge Base Section
        kb_label = ctk.CTkLabel(parent, text="Knowledge Base:",
                                font=ctk.CTkFont(size=13, weight="bold"))
        kb_label.pack(pady=(10, 5), padx=20, anchor="w")
        TooltipManager.create_tooltip(
            kb_label, "Manage medical knowledge database")

        self.build_button = ctk.CTkButton(parent, text="Synchronize Library",
                                          command=self.run_build)
        self.build_button.pack(pady=(0, 5), padx=20, fill="x")
        TooltipManager.create_tooltip(self.build_button,
                                      "Update knowledge base with new documents")

        self.force_rebuild_var = ctk.BooleanVar()
        rebuild_frame = ctk.CTkFrame(parent, fg_color="transparent")
        rebuild_frame.pack(pady=(0, 20), padx=20, fill="x")

        self.rebuild_check = ctk.CTkCheckBox(rebuild_frame, text="Force Rebuild",
                                             variable=self.force_rebuild_var)
        self.rebuild_check.pack(side="left")

        help_btn = ctk.CTkButton(rebuild_frame, text="?", width=30,
                                 command=lambda: self._show_help(
                                     "Force Rebuild completely regenerates the knowledge base"))
        help_btn.pack(side="right")

        # Training Section
        train_label = ctk.CTkLabel(parent, text="AI Training:",
                                   font=ctk.CTkFont(size=13, weight="bold"))
        train_label.pack(pady=(10, 5), padx=20, anchor="w")

        self.train_button = ctk.CTkButton(parent, text="Train Model",
                                          command=self.run_train, state="disabled")
        self.train_button.pack(pady=(0, 5), padx=20, fill="x")

        # Add other UI elements...

    def _build_log_console(self):
        """Configure the logging console"""
        self.log_text = ctk.CTkTextbox(
            self.log_frame, state="disabled", wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Configure log tags
        tags = {
            "INFO": {"foreground": "cyan"},
            "ERROR": {"foreground": "red"},
            "WARNING": {"foreground": "yellow"},
            "SUCCESS": {"foreground": "lime"},
            "DEBUG": {"foreground": "gray"},
            "SYSTEM": {"foreground": "#FFB380"}
        }

        for tag, config in tags.items():
            self.log_text.tag_config(tag, **config)

    def _build_status_panel(self, parent):
        """Build the status monitoring panel"""
        ctk.CTkLabel(parent, text="TRAINING STATUS",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 20))

        self.status_label = ctk.CTkLabel(parent, text="Status: Idle")
        self.status_label.pack(pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(parent)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=(0, 20), padx=20, fill="x")

        self.epoch_label = ctk.CTkLabel(parent, text="Epoch: --")
        self.epoch_label.pack(pady=5, anchor="w")

        self.loss_label = ctk.CTkLabel(parent, text="Loss: --")
        self.loss_label.pack(pady=5, anchor="w")

        # Add other status elements...

    def _show_help(self, message):
        """Show help message in a managed dialog"""
        if hasattr(self, "_help_dialog"):
            try:
                self._help_dialog.destroy()
            except:
                pass

        self._help_dialog = dialog = ctk.CTkToplevel(self)
        dialog.title("Help")
        dialog.geometry("500x300")

        text = ctk.CTkTextbox(dialog, wrap="word")
        text.pack(fill="both", expand=True, padx=20, pady=20)
        text.insert("1.0", message)
        text.configure(state="disabled")

        ctk.CTkButton(dialog, text="Close",
                      command=dialog.destroy).pack(pady=(0, 20))

    # Add other methods (run_build, run_train, process_gui_queue, etc.)...

    def cleanup(self):
        """Proper cleanup on exit"""
        TooltipManager._cleanup()
        for monitor in self.active_log_monitors.values():
            monitor.stop()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.cleanup)
    app.mainloop()
