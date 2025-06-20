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

# --- Thread for Running Backend Scripts ---


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
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1, creationflags=creation_flags)

            for line in iter(process.stdout.readline, ''):
                self.queue.put(("PROCESS_LOG", line.strip()))
            process.stdout.close()
            process.wait()
        except FileNotFoundError:
            self.queue.put(
                ("SYSTEM", f"ERROR: Script '{self.command_args[0]}' not found."))
        except Exception as e:
            self.queue.put(("SYSTEM", f"ERROR in process thread: {e}"))
        finally:
            self.queue.put(("SYSTEM", "---PROCESS-COMPLETE---"))

# --- Thread for Tailing Log Files ---


class LogMonitorThread(threading.Thread):
    def __init__(self, filepath, queue):
        super().__init__()
        self.filepath = filepath
        self.queue = queue
        self.daemon = True
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        self.queue.put(
            ("SYSTEM", f"Activating Log Monitor for: {os.path.basename(self.filepath)}"))
        # Clear the log file on start
        try:
            open(self.filepath, 'w').close()
        except Exception as e:
            self.queue.put(
                ("SYSTEM", f"Warning: Could not clear {self.filepath}: {e}"))

        last_pos = 0
        while not self._stop_event.is_set():
            try:
                if not os.path.exists(self.filepath):
                    time.sleep(0.5)
                    continue
                with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(0, 2)
                    current_pos = f.tell()
                    if current_pos > last_pos:
                        f.seek(last_pos)
                        for line in f:
                            if line.strip():
                                self.queue.put(("PRIMARY_LOG", line.strip()))
                        last_pos = f.tell()
                    else:
                        time.sleep(0.1)  # Wait if no new content
            except Exception as e:
                self.queue.put(
                    ("SYSTEM", f"Err in LogMonitor ({self.filepath}): {e}"))
                time.sleep(1)

# --- Tooltip Class ---


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ctk.CTkLabel(tw, text=self.text, corner_radius=4, fg_color="#FFFFE0",
                             text_color="#000000", padx=8, pady=4, wraplength=350)
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# --- Main Application Window ---


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MedAI - Adaptive Diagnostic System")
        self.geometry("1600x900")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.process_queue = queue.Queue()
        self.active_log_monitor = None
        self.process_history = []

        self.notebook = ctk.CTkTabview(self, corner_radius=10)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.control_tab = self.notebook.add("Control Center")
        self.about_tab = self.notebook.add("About")

        # Configure grid for control tab
        self.control_tab.grid_columnconfigure(0, weight=1)  # Control Panel
        self.control_tab.grid_columnconfigure(1, weight=3)  # Log Console
        self.control_tab.grid_columnconfigure(2, weight=1)  # Telemetry Panel
        self.control_tab.grid_rowconfigure(0, weight=1)

        self.create_control_panel()
        self.create_log_console()
        self.create_telemetry_panel()
        self.create_about_content()
        self.after(100, self.process_gui_queue)

    def create_control_panel(self):
        frame = ctk.CTkFrame(self.control_tab, corner_radius=10)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="COMMAND CENTER", font=ctk.CTkFont(
            size=20, weight="bold")).pack(pady=(15, 20), padx=20, fill="x")

        # Knowledge Base Sync Section
        sync_label = ctk.CTkLabel(
            frame, text="Step 1: Ingest Data", font=ctk.CTkFont(size=14, weight="bold"))
        sync_label.pack(pady=(10, 5), padx=20, anchor="w")

        self.build_button = ctk.CTkButton(
            frame, text="Synchronize Library", command=self.run_build, height=40)
        self.build_button.pack(pady=5, padx=20, fill="x")
        ToolTip(self.build_button,
                "Analyzes 'data/library', processing new or updated PDFs into a searchable knowledge base.")

        rebuild_frame = ctk.CTkFrame(frame, fg_color="transparent")
        rebuild_frame.pack(pady=(5, 20), padx=20, fill="x")
        self.force_rebuild_var = ctk.BooleanVar(value=False)
        rebuild_checkbox = ctk.CTkCheckBox(
            rebuild_frame, text="Force Full Rebuild", variable=self.force_rebuild_var)
        rebuild_checkbox.pack(side="left")
        ToolTip(rebuild_checkbox, "Deletes the cache and rebuilds the entire knowledge base from scratch. Use if you suspect corruption.")

        # AI Adaptation Section
        adapt_label = ctk.CTkLabel(
            frame, text="Step 2: Evolve AI", font=ctk.CTkFont(size=14, weight="bold"))
        adapt_label.pack(pady=(10, 5), padx=20, anchor="w")

        self.adapt_button = ctk.CTkButton(
            frame, text="Adapt AI to New Literature", command=self.run_adapt, height=40)
        self.adapt_button.pack(pady=5, padx=20, fill="x")
        ToolTip(self.adapt_button, "Initiates the self-learning cycle. The AI reads newly added documents (from Step 1) and updates its own understanding.")

        # Optional Training Section
        train_label = ctk.CTkLabel(
            frame, text="Advanced: Full Training Cycle", font=ctk.CTkFont(size=14, weight="bold"))
        train_label.pack(pady=(30, 5), padx=20, anchor="w")

        self.train_button = ctk.CTkButton(frame, text="Initiate Full Training",
                                          command=self.run_train, height=40, state="normal")  # Keep enabled for now
        self.train_button.pack(pady=5, padx=20, fill="x")
        ToolTip(self.train_button,
                "Runs a full supervised training cycle. Slower than adapting. Use for major model updates.")

        # --- Spacer to push clear logs to bottom ---
        ctk.CTkLabel(frame, text="").pack(pady=10, fill="y", expand=True)

        self.clear_logs_button = ctk.CTkButton(
            frame, text="Clear Log Console", command=self.clear_logs, height=35, fg_color="#555555", hover_color="#444444")
        self.clear_logs_button.pack(pady=(10, 20), padx=20, fill="x")
        ToolTip(self.clear_logs_button, "Clear the main log display.")

    def create_log_console(self):
        frame = ctk.CTkFrame(self.control_tab, corner_radius=10)
        frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.log_textbox = ctk.CTkTextbox(frame, state="disabled", wrap="word", font=(
            "Consolas", 10), corner_radius=8, border_width=2)
        self.log_textbox.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Configure tags for color-coded logging
        self.log_textbox.tag_config(
            "INFO", foreground="#6495ED")      # CornflowerBlue
        self.log_textbox.tag_config(
            "SUCCESS", foreground="#32CD32")    # LimeGreen
        self.log_textbox.tag_config("WARNING", foreground="#FFD700")    # Gold
        self.log_textbox.tag_config(
            "ERROR", foreground="#FF4500")      # OrangeRed
        self.log_textbox.tag_config(
            "SYSTEM", foreground="#DA70D6")     # Orchid
        self.log_textbox.tag_config("DEBUG", foreground="#808080")      # Gray
        self.log_textbox.tag_config("COMMAND", foreground="#DDA0DD")    # Plum
        self.log_textbox.tag_config(
            "TIMESTAMP", foreground="#AAAAAA")  # LightGray

    def create_telemetry_panel(self):
        frame = ctk.CTkFrame(self.control_tab, corner_radius=10)
        frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="SYSTEM TELEMETRY", font=ctk.CTkFont(
            size=20, weight="bold")).pack(pady=(15, 10))

        # --- Dynamic status grid ---
        status_frame = ctk.CTkFrame(frame, fg_color="transparent")
        status_frame.pack(fill="x", padx=15, pady=10)
        status_frame.grid_columnconfigure(1, weight=1)

        def add_telemetry_row(parent, row_idx, label_text, default_value, tooltip_text):
            label = ctk.CTkLabel(
                parent, text=f"{label_text}:", font=("Consolas", 14))
            label.grid(row=row_idx, column=0, sticky="w", padx=(0, 10))
            value_label = ctk.CTkLabel(parent, text=default_value, font=(
                "Consolas", 14, "bold"), anchor="e")
            value_label.grid(row=row_idx, column=1, sticky="ew")
            ToolTip(value_label, tooltip_text)
            return value_label

        self.status_label = add_telemetry_row(
            status_frame, 0, "System Status", "Idle", "Current system status")
        self.progress_label = add_telemetry_row(
            status_frame, 1, "Progress", "0%", "Current operation progress")
        self.reward_label = add_telemetry_row(
            status_frame, 2, "Adapt Reward", "N/A", "Performance change from last self-edit. Positive is good.")
        self.init_score_label = add_telemetry_row(
            status_frame, 3, "Initial Score", "N/A", "AI's performance score before starting the adaptation cycle.")
        self.final_score_label = add_telemetry_row(
            status_frame, 4, "Final Score", "N/A", "AI's performance score after the adaptation cycle.")

    def create_about_content(self):
        frame = ctk.CTkScrollableFrame(self.about_tab, corner_radius=10)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        ctk.CTkLabel(frame, text="MedAI: Adaptive Diagnostic System",
                     font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(20, 15))
        about_text = ("This system uses an advanced AI model capable of self-adaptation. "
                      "By processing new medical literature, it continuously refines its own "
                      "internal knowledge, improving its diagnostic reasoning over time without "
                      "constant manual retraining.")
        ctk.CTkLabel(frame, text=about_text, font=ctk.CTkFont(
            size=14), wraplength=800, justify="left").pack(pady=(0, 30), padx=20)
        # Placeholder for more content...

    def start_process_and_monitor(self, script_name, log_file, button_to_disable):
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        if not os.path.exists(script_path):
            self.log_to_box(
                f"SYSTEM|ERROR|Script '{script_name}' not found.\n")
            return

        self.log_to_box(f"SYSTEM|COMMAND|Launching '{script_name}'...\n")
        self.status_label.configure(
            text=f"Running {script_name.split('.')[0]}")
        self.clear_logs()

        # Disable all major action buttons
        self.build_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        self.adapt_button.configure(state="disabled")

        if self.active_log_monitor and self.active_log_monitor.is_alive():
            self.active_log_monitor.stop()

        self.active_log_monitor = LogMonitorThread(
            log_file, self.process_queue)
        self.active_log_monitor.start()

        command_args = [script_path]
        if script_name == "builder.py" and self.force_rebuild_var.get():
            command_args.append("--force-rebuild")

        ProcessThread(command_args=command_args,
                      queue=self.process_queue).start()

    def run_build(self):
        self.start_process_and_monitor(
            "builder.py", "builder.log", self.build_button)

    def run_train(self):
        self.start_process_and_monitor(
            "train.py", "trainer.log", self.train_button)

    def run_adapt(self):
        self.start_process_and_monitor(
            "adapter.py", "adapter.log", self.adapt_button)

    def process_gui_queue(self):
        while not self.process_queue.empty():
            log_type, msg_content = self.process_queue.get_nowait()

            if log_type == "SYSTEM" and "---PROCESS-COMPLETE---" in msg_content:
                if self.active_log_monitor:
                    self.active_log_monitor.stop()
                self.status_label.configure(text="Idle")
                # Re-enable buttons
                self.build_button.configure(state="normal")
                self.train_button.configure(state="normal")
                self.adapt_button.configure(state="normal")
                continue

            # Universal Log Parser
            is_telemetry = False
            parts = msg_content.split('|')
            if len(parts) > 2 and parts[0] == "STATUS":
                is_telemetry = True
                telemetry_key = parts[1].upper()
                telemetry_value = parts[2]

                # Update Telemetry Panel based on key
                if telemetry_key == "PROGRESS":
                    try:
                        percent = f"{float(telemetry_value) * 100:.1f}%"
                        self.progress_label.configure(text=percent)
                    except:
                        pass
                elif telemetry_key == "REWARD":
                    try:
                        reward_val = float(telemetry_value)
                        self.reward_label.configure(
                            text=f"{reward_val:+.4f}", text_color="lime" if reward_val > 0 else "red")
                    except:
                        self.reward_label.configure(
                            text=telemetry_value, text_color="white")
                elif telemetry_key == "INITIAL_SCORE":
                    self.init_score_label.configure(text=telemetry_value)
                elif telemetry_key == "FINAL_SCORE":
                    self.final_score_label.configure(text=telemetry_value)
                else:
                    is_telemetry = False  # If key not recognized, treat as regular log

            if not is_telemetry:
                self.log_to_box(msg_content)

        self.after(100, self.process_gui_queue)

    def log_to_box(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Default level
        log_level = "INFO"
        clean_msg = msg

        # Check for keywords to determine level
        msg_lower = msg.lower()
        if "error" in msg_lower or "failed" in msg_lower or "fatal" in msg_lower:
            log_level = "ERROR"
        elif "warning" in msg_lower:
            log_level = "WARNING"
        elif "success" in msg_lower:
            log_level = "SUCCESS"
        elif "debug" in msg_lower:
            log_level = "DEBUG"
        elif "command" in msg_lower:
            log_level = "COMMAND"

        # Check for structured log format e.g. "LOG|LEVEL|Message"
        parts = msg.split('|')
        if len(parts) >= 3 and parts[0].upper() in ["LOG", "SYSTEM"]:
            log_level = parts[1].upper()
            clean_msg = "|".join(parts[2:])

        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"[{timestamp}] ", ("TIMESTAMP",))
        self.log_textbox.insert("end", f"[{log_level}] ", (log_level,))
        self.log_textbox.insert("end", f"{clean_msg.strip()}\n", (log_level,))
        self.log_textbox.configure(state="disabled")
        self.log_textbox.see("end")

    def clear_logs(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")
        self.log_to_box("SYSTEM|INFO|Logs cleared by user.")
        # Also reset telemetry
        self.progress_label.configure(text="0%")
        self.reward_label.configure(text="N/A", text_color="white")
        self.init_score_label.configure(text="N/A")
        self.final_score_label.configure(text="N/A")


if __name__ == "__main__":
    app = App()
    app.mainloop()
