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
import webbrowser  # Added from patch


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
                                       text=True, encoding='utf-8', bufsize=1, creationflags=creation_flags)

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
            ("SYSTEM", f"Activating Log Monitor for: {os.path.basename(self.filepath)}"))
        if os.path.exists(self.filepath):
            try:
                open(self.filepath, 'w').close()
            except Exception as e:
                self.queue.put(
                    ("SYSTEM", f"Warning: Could not clear {self.filepath}: {e}"))

        last_pos = 0
        first_read = True
        while not self._stop_event.is_set():
            try:
                if not os.path.exists(self.filepath):
                    time.sleep(0.5)
                    continue
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    if first_read:
                        f.seek(0, 2)
                        last_pos = f.tell()
                        first_read = False
                    else:
                        f.seek(last_pos)

                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    log_entry_type = "PRIMARY_LOG" if self.is_primary_log else "TELEMETRY"
                    self.queue.put((log_entry_type, line.strip()))
                    last_pos = f.tell()
            except FileNotFoundError:
                time.sleep(0.5)
            except Exception as e:
                self.queue.put(
                    ("SYSTEM", f"Err in LogMonitor ({self.filepath}): {e}"))
                time.sleep(1)


class ToolTip:
    """Custom tooltip class for CTk"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        """Display tooltip"""
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = ctk.CTkLabel(tw, text=self.text,
                             corner_radius=4,
                             fg_color="#FFFFE0",
                             text_color="#000000",
                             padx=8, pady=4,
                             wraplength=300)
        label.pack()

    def hide_tip(self, event=None):
        """Hide tooltip"""
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MedAI - Medical Diagnosis Assistant")  # From patch
        self.geometry("1500x900")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Configure grid layout (from patch)
        self.grid_columnconfigure(0, weight=1)  # Adjusted for notebook to span
        self.grid_rowconfigure(0, weight=1)
        # Columns 1,2,3 are now handled within the control_tab if needed,
        # but the main window now has the notebook spanning its columns.
        # The patch used columnspan=4 for notebook, implying 4 main columns.
        # Let's adjust main grid config for the notebook
        # self.grid_columnconfigure(1, weight=3) # Old from base
        # self.grid_rowconfigure(0, weight=1)    # Old from base
        # self.grid_columnconfigure(2, weight=1) # Old from base
        # self.grid_columnconfigure(3, weight=1) # Old from base (For history panel)
        # With notebook, we'll have one main column (0) for the notebook itself.

        self.process_queue = queue.Queue()
        self.active_log_monitors = {}
        self.process_history = []

        # Create notebook for tabs (from patch)
        self.notebook = ctk.CTkTabview(self)
        # Spanning all if only one main column
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create tabs (from patch)
        self.control_tab = self.notebook.add("Control Center")
        self.about_tab = self.notebook.add("About")

        # Configure control_tab grid (from patch)
        self.control_tab.grid_columnconfigure(1, weight=3)  # Log console
        self.control_tab.grid_rowconfigure(
            0, weight=1)    # All panels in row 0
        self.control_tab.grid_columnconfigure(
            0, weight=0)  # Control panel (fixed width)
        self.control_tab.grid_columnconfigure(2, weight=1)  # Status panel
        self.control_tab.grid_columnconfigure(3, weight=1)  # History panel

        # Build widgets in tabs
        self.create_control_panel()
        self.create_log_console()
        self.create_status_panel()
        self.create_process_history_panel()
        self.create_about_content()  # From patch

        self.after(100, self.process_gui_queue)

    def create_control_panel(self):
        # Parent is now self.control_tab
        frame = ctk.CTkFrame(self.control_tab, width=280, corner_radius=10)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsw")
        frame.grid_propagate(False)

        ctk.CTkLabel(frame, text="COMMAND CENTER",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 15))

        # Knowledge Base Sync Section (from patch)
        sync_label = ctk.CTkLabel(frame, text="Knowledge Base Sync:",
                                  font=ctk.CTkFont(size=13, weight="bold"))
        sync_label.pack(pady=(10, 5), padx=20, anchor="w")
        ToolTip(
            sync_label, "Synchronize medical knowledge base with latest research and data")

        self.build_button = ctk.CTkButton(
            frame, text="1. Synchronize Library",  # Text from patch
            command=self.run_build, height=35)
        self.build_button.pack(pady=(0, 5), padx=20, fill="x")
        ToolTip(self.build_button,
                "Build/update the medical knowledge database from source documents")

        self.force_rebuild_var = ctk.BooleanVar(value=False)
        # Frame for checkbox and help
        rebuild_frame = ctk.CTkFrame(frame, fg_color="transparent")
        rebuild_frame.pack(pady=(0, 20), padx=20, fill="x")
        self.rebuild_checkbox = ctk.CTkCheckBox(
            rebuild_frame, text="Force Full Rebuild",
            variable=self.force_rebuild_var)
        self.rebuild_checkbox.pack(side="left")
        ToolTip(self.rebuild_checkbox,
                "Completely rebuild the knowledge base from scratch")

        rebuild_help = ctk.CTkButton(rebuild_frame, text="?", width=20, height=20,  # Help button from patch
                                     fg_color="#333333", hover_color="#444444",
                                     command=lambda: self.show_help("Force Rebuild rebuilds the entire knowledge base from scratch. Use this when you've added new documents or suspect corruption."))
        rebuild_help.pack(side="right", padx=(5, 0))

        # AI Training Section (from patch)
        train_label = ctk.CTkLabel(frame, text="AI Training:",
                                   font=ctk.CTkFont(size=13, weight="bold"))
        train_label.pack(pady=(10, 5), padx=20, anchor="w")
        ToolTip(train_label, "Train the diagnostic AI model")

        self.train_button = ctk.CTkButton(
            frame, text="2. Initiate Training Cycle",  # Text from patch
            command=self.run_train, height=35, state="disabled")
        self.train_button.pack(pady=(0, 5), padx=20, fill="x")
        ToolTip(self.train_button,
                "Train the diagnostic model using the latest knowledge base")

        verbosity_label = ctk.CTkLabel(frame, text="Trainer Log Verbosity:")
        verbosity_label.pack(padx=20, pady=(10, 0), anchor="w")
        ToolTip(verbosity_label, "Set the level of detail in training logs")

        self.log_level_var = ctk.StringVar(value="INFO")
        self.log_level_menu = ctk.CTkOptionMenu(
            frame, variable=self.log_level_var, values=["INFO", "DEBUG"])
        self.log_level_menu.pack(pady=(0, 10), padx=20, fill="x")
        ToolTip(self.log_level_menu,
                "DEBUG shows detailed technical information, INFO shows general progress")

        rag_label = ctk.CTkLabel(frame, text="AI Reasoning Profile (RAG):")
        rag_label.pack(padx=20, pady=(0, 0), anchor="w")
        ToolTip(
            rag_label, "Select how the AI retrieves and reasons with medical knowledge")

        self.reasoning_mode_var = ctk.StringVar(value="balanced")
        self.reasoning_mode_menu = ctk.CTkOptionMenu(
            frame, variable=self.reasoning_mode_var,
            # "diagnostic" added from patch
            values=["balanced", "deep_dive", "quick_fact", "diagnostic"])
        self.reasoning_mode_menu.pack(pady=(0, 20), padx=20, fill="x")
        ToolTip(self.reasoning_mode_menu,
                "Balanced: General medical reasoning\n"
                "Deep Dive: Comprehensive analysis\n"
                "Quick Fact: Rapid information retrieval\n"
                "Diagnostic: Focused on identifying conditions")

        self.clear_logs_button = ctk.CTkButton(
            frame, text="Clear Logs", command=self.clear_logs,
            height=30, fg_color="#555555", hover_color="#444444")
        self.clear_logs_button.pack(pady=(10, 5), padx=20, fill="x")
        ToolTip(self.clear_logs_button, "Clear the console log display")

    def create_log_console(self):
        # Parent is now self.control_tab
        frame = ctk.CTkFrame(self.control_tab, corner_radius=10)
        frame.grid(row=0, column=1, padx=0, pady=10, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)  # Ensure textbox expands
        frame.grid_columnconfigure(0, weight=1)  # Ensure textbox expands

        self.log_textbox = ctk.CTkTextbox(frame, state="disabled", wrap="word",
                                          activate_scrollbars=True)
        # Added row/col for clarity
        self.log_textbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.log_textbox.tag_config("INFO", foreground="cyan")
        self.log_textbox.tag_config("SUCCESS", foreground="lime")
        self.log_textbox.tag_config("WARNING", foreground="yellow")
        self.log_textbox.tag_config("ERROR", foreground="red")
        self.log_textbox.tag_config("COMMAND", foreground="#D8BFD8")
        self.log_textbox.tag_config("SYSTEM", foreground="#FFB380")
        self.log_textbox.tag_config("DEBUG", foreground="#808080")
        self.log_textbox.tag_config("TIMESTAMP", foreground="#AAAAAA")

    def create_status_panel(self):
        # Parent is now self.control_tab
        frame = ctk.CTkFrame(self.control_tab, width=280, corner_radius=10)
        # Changed sticky to nsew from nsw
        frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        frame.grid_propagate(False)

        ctk.CTkLabel(frame, text="TRAINING TELEMETRY",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 20))

        self.status_indicator = ctk.CTkLabel(frame, text="Status: Idle",
                                             font=("Consolas", 12))
        self.status_indicator.pack(pady=(0, 10))
        ToolTip(self.status_indicator, "Current system status")

        self.epoch_label = ctk.CTkLabel(frame, text="Epoch: Standby",
                                        font=("Consolas", 14))
        self.epoch_label.pack(pady=5, padx=20, anchor="w")
        ToolTip(self.epoch_label, "Current training epoch")

        self.progress_bar = ctk.CTkProgressBar(frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=(10, 20), padx=20, fill="x")
        ToolTip(self.progress_bar, "Training progress")

        self.loss_batch_label = ctk.CTkLabel(frame, text="Batch Loss: --",
                                             font=("Consolas", 14))
        self.loss_batch_label.pack(pady=5, padx=20, anchor="w")
        ToolTip(self.loss_batch_label, "Loss for current batch of data")

        self.loss_avg_label = ctk.CTkLabel(frame, text="Avg. Loss: --",
                                           font=("Consolas", 14))
        self.loss_avg_label.pack(pady=5, padx=20, anchor="w")
        ToolTip(self.loss_avg_label, "Average loss across all batches")

        self.execution_time_label = ctk.CTkLabel(frame, text="Time: --",
                                                 font=("Consolas", 12))
        self.execution_time_label.pack(pady=(20, 5), padx=20, anchor="w")
        ToolTip(self.execution_time_label,
                "Time elapsed for current operation")

    def create_process_history_panel(self):
        # Parent is now self.control_tab
        self.history_frame = ctk.CTkFrame(
            self.control_tab, width=280, corner_radius=10)
        # Changed sticky to nsew
        self.history_frame.grid(
            row=0, column=3, padx=10, pady=10, sticky="nsew")
        self.history_frame.grid_propagate(False)
        self.history_frame.grid_rowconfigure(
            1, weight=1)  # Allow textbox to expand

        ctk.CTkLabel(self.history_frame, text="PROCESS HISTORY",
                     # Changed from grid to pack
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 15))

        self.history_textbox = ctk.CTkTextbox(
            self.history_frame, state="disabled")  # Removed height, rely on expand
        # Changed from grid to pack
        self.history_textbox.pack(
            pady=(0, 10), padx=10, fill="both", expand=True)

        self.history_textbox.tag_config("SUCCESS", foreground="lime")
        self.history_textbox.tag_config("ERROR", foreground="red")

        clear_btn = ctk.CTkButton(self.history_frame, text="Clear History",
                                  command=self.clear_history,
                                  fg_color="#555555", hover_color="#444444")
        # Changed from grid to pack
        clear_btn.pack(pady=(0, 10), padx=10, fill="x")
        ToolTip(clear_btn, "Clear process history")

    def create_about_content(self):  # New method from patch
        about_frame = ctk.CTkFrame(self.about_tab, corner_radius=10)
        about_frame.pack(fill="both", expand=True, padx=20, pady=20)

        title = ctk.CTkLabel(about_frame, text="MedAI Diagnostic System",
                             font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=(20, 10))

        desc = ctk.CTkLabel(about_frame,
                            text="An AI-powered diagnostic assistant that analyzes medical images and reports to identify health conditions",
                            # Added wraplength
                            font=ctk.CTkFont(size=14), wraplength=about_frame.winfo_width() - 60)
        desc.pack(pady=(0, 20), padx=20)

        workflow_frame = ctk.CTkFrame(
            about_frame, height=200)  # Placeholder height
        workflow_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(workflow_frame, text="System Workflow Diagram (Placeholder)",
                     # expand for centering
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=40, expand=True)

        workflow_text = """
1. Data Ingestion:
   - Medical images (CT, X-ray, MRI)
   - Patient reports and histories
   - Lab results and diagnostic tests

2. Knowledge Base:
   - Medical literature and research
   - Clinical guidelines and protocols
   - Historical case data

3. Multi-Modal Analysis:
   - Computer vision for image analysis
   - NLP for report understanding
   - Temporal analysis for progression tracking

4. Diagnostic Reasoning:
   - Symptom-disease correlation
   - Differential diagnosis generation
   - Confidence scoring for predictions

5. Reporting & Recommendations:
   - Diagnostic report generation
   - Treatment options and alternatives
   - Follow-up recommendations
"""
        workflow_label = ctk.CTkLabel(about_frame, text=workflow_text,
                                      justify="left", anchor="w")
        workflow_label.pack(pady=20, padx=30, anchor="w")

        doc_frame = ctk.CTkFrame(about_frame, fg_color="transparent")
        doc_frame.pack(pady=20)
        ctk.CTkLabel(doc_frame, text="For full documentation:").pack(
            side="left")
        doc_link = ctk.CTkButton(doc_frame, text="Open Documentation",
                                 command=lambda: webbrowser.open(
                                     "https://example.com/medai-docs"),
                                 fg_color="transparent", text_color="#1E90FF", hover=False)
        doc_link.pack(side="left", padx=5)

    def show_help(self, message):  # New method from patch
        dialog = ctk.CTkToplevel(self)
        dialog.title("Help")
        dialog.geometry("500x300")  # Adjusted for better text visibility
        dialog.transient(self)
        dialog.grab_set()

        # Center the dialog
        self.update_idletasks()
        dialog_width = 500
        dialog_height = 300
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (dialog_width // 2)
        y = (screen_height // 2) - (dialog_height // 2)
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")

        text_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        text = ctk.CTkTextbox(text_frame, wrap="word", activate_scrollbars=True,
                              # Slightly larger font
                              font=ctk.CTkFont(size=13))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", message)
        text.configure(state="disabled")

        close_btn = ctk.CTkButton(
            dialog, text="Close", command=dialog.destroy, width=100)
        close_btn.pack(pady=(0, 20))

    def start_process_and_monitor(self, script_name, log_file, button_to_disable, button_text_running):
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        if not os.path.exists(script_path):
            self.log_to_box(
                f"SYSTEM|ERROR|Script '{script_name}' not found at {script_path}\n", "ERROR")
            button_to_disable.configure(
                state="normal", text="ERROR: Script Not Found")  # Keep original text on error
            return

        self.log_to_box(
            f"SYSTEM|COMMAND|Launching '{script_name}'...\n", "COMMAND")
        button_to_disable.configure(state="disabled", text=button_text_running)
        self.status_indicator.configure(
            # Cleaner name
            text=f"Status: Running {script_name.split('.')[0]}")

        process_record = {
            "name": script_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Running"
        }
        self.process_history.append(process_record)
        self.update_history_display()

        # Disable both buttons when one process starts
        self.build_button.configure(state="disabled")
        self.train_button.configure(state="disabled")

        self.start_log_monitor(log_file, is_primary_log=True)

        command_args = [script_path]
        if script_name == "builder.py" and self.force_rebuild_var.get():
            command_args.append("--force-rebuild")
            self.log_to_box(
                "SYSTEM|INFO|Force rebuild selected for builder.\n", "INFO")
        elif script_name == "train.py":
            command_args.extend(["--level", self.log_level_var.get()])
            command_args.extend(["--reasoning", self.reasoning_mode_var.get()])
            self.log_to_box(
                f"SYSTEM|INFO|Trainer started with Log: {self.log_level_var.get()}, Reasoning: {self.reasoning_mode_var.get()}\n", "INFO")

        ProcessThread(command_args=command_args,
                      queue=self.process_queue).start()

    def update_history_display(self):
        self.history_textbox.configure(state="normal")
        self.history_textbox.delete("1.0", "end")

        for record in reversed(self.process_history[-10:]):  # Show last 10
            timestamp = record["start_time"]
            name = record["name"]
            status = record.get("status", "Unknown")
            end_time_str = record.get("end_time", "")

            duration_str = ""
            if status != "Running" and end_time_str:
                try:
                    start_dt = datetime.strptime(
                        timestamp, "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(
                        end_time_str, "%Y-%m-%d %H:%M:%S")
                    duration = end_dt - start_dt
                    # Remove microseconds
                    duration_str = f" (Took: {str(duration).split('.')[0]})"
                except ValueError:
                    pass  # In case of parsing error

            tag = "SUCCESS" if status == "Completed" else "ERROR" if status == "Failed" else None
            entry = f"[{timestamp}] {name} - {status}{duration_str}\n"

            if tag:
                # Insert at beginning for newest first
                self.history_textbox.insert("1.0", entry, tag)
            else:
                self.history_textbox.insert("1.0", entry)

        self.history_textbox.configure(state="disabled")
        # self.history_textbox.see("end") # Not needed if inserting at 1.0

    def clear_logs(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")
        self.log_to_box("SYSTEM|INFO|Logs cleared by user.\n", "INFO")

    def clear_history(self):  # Added to connect to button
        self.process_history.clear()
        self.update_history_display()
        self.log_to_box(
            "SYSTEM|INFO|Process history cleared by user.\n", "INFO")

    def run_build(self):
        self.start_process_and_monitor(
            "builder.py", "builder.log", self.build_button, "Synchronizing...")

    def run_train(self):
        # Check if builder.py has run successfully at least once if needed
        # For now, just enable it if no process is running
        self.start_process_and_monitor(
            "train.py", "trainer.log", self.train_button, "Training...")

    def start_log_monitor(self, log_file, is_primary_log):
        if log_file in self.active_log_monitors and self.active_log_monitors[log_file].is_alive():
            self.active_log_monitors[log_file].stop()
            self.active_log_monitors[log_file].join(timeout=0.5)

        if is_primary_log:
            self.log_textbox.configure(state="normal")
            self.log_textbox.delete("1.0", "end")  # Clear previous primary log
            self.log_textbox.configure(state="disabled")

        monitor_thread = LogMonitorThread(
            log_file, self.process_queue, is_primary_log=is_primary_log)
        monitor_thread.start()
        self.active_log_monitors[log_file] = monitor_thread

    def process_gui_queue(self):
        while not self.process_queue.empty():
            log_type, msg_content = self.process_queue.get_nowait()

            if log_type == "SYSTEM" and "PROCESS-COMPLETE" in msg_content:
                process_failed = "error" in self.status_indicator.cget("text").lower() or \
                                 any("error" in entry.lower() for entry in self.log_textbox.get(
                                     "1.0", "end").splitlines()[-5:])

                if self.process_history:
                    last_process = self.process_history[-1]
                    if last_process["status"] == "Running":
                        last_process["status"] = "Failed" if process_failed else "Completed"
                        last_process["end_time"] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S")
                        self.update_history_display()

                # Determine which process just finished based on active log monitor or button text
                finished_process_name = ""
                if self.active_log_monitors.get("builder.log") and not self.active_log_monitors["builder.log"].is_alive():
                    finished_process_name = "Knowledge Base Sync"
                    # Ensure it's stopped
                    self.active_log_monitors["builder.log"].stop()
                elif self.active_log_monitors.get("trainer.log") and not self.active_log_monitors["trainer.log"].is_alive():
                    finished_process_name = "AI Training"
                    # Ensure it's stopped
                    self.active_log_monitors["trainer.log"].stop()

                # Re-enable buttons
                self.build_button.configure(
                    state="normal", text="1. Synchronize Library")
                # Train button should only be enabled if build was successful (or at least one successful build happened)
                # For simplicity now, just re-enable it. A more robust check might be needed.
                self.train_button.configure(
                    state="normal", text="2. Initiate Training Cycle")

                self.status_indicator.configure(
                    text=f"Status: Idle ({'Failed' if process_failed else 'Completed'})")

                if finished_process_name:
                    log_msg = f"SYSTEM|{'ERROR' if process_failed else 'SUCCESS'}|{finished_process_name} {'failed' if process_failed else 'finished'}.\n"
                    self.log_to_box(
                        log_msg, "ERROR" if process_failed else "SUCCESS")
                else:  # Fallback if specific process couldn't be identified
                    log_msg = f"SYSTEM|{'ERROR' if process_failed else 'SUCCESS'}|A process {'failed' if process_failed else 'finished'}.\n"
                    self.log_to_box(
                        log_msg, "ERROR" if process_failed else "SUCCESS")

                continue  # Skip further processing for this special message

            timestamp = datetime.now().strftime("%H:%M:%S")
            # For primary logs, don't prepend our own timestamp if the log already has one.
            # However, for simplicity and consistency in the box, we will add our timestamp.
            # formatted_msg = f"[{timestamp}] {msg_content}"
            # This logic needs refinement based on expected log format.
            # The current log_to_box adds its own timestamp and level.

            # Simplified processing logic for now
            parts = msg_content.strip().split('|')
            is_status_msg = (len(parts) >= 3 and parts[0] == "STATUS") or \
                any(kw in msg_content.upper()
                    for kw in ["EPOCH", "PROGRESS", "LOSS_BATCH", "LOSS_AVG"])

            if log_type == "TELEMETRY" or is_status_msg:
                # Try to parse specific telemetry data
                parsed_telemetry = False
                if re.search(r'EPOCH(?:[^\|]*\|){1}([\d\s\/]+)', msg_content.upper()):
                    m = re.search(
                        r'EPOCH(?:[^\|]*\|){1}([\d\s\/]+)', msg_content.upper())
                    self.epoch_label.configure(
                        text=f"Epoch: {m.group(1).strip()}")
                    parsed_telemetry = True
                if re.search(r'PROGRESS(?:[^\|]*\|){1}([\d\.]+)', msg_content.upper()):
                    m = re.search(
                        r'PROGRESS(?:[^\|]*\|){1}([\d\.]+)', msg_content.upper())
                    self.progress_bar.set(float(m.group(1)))
                    parsed_telemetry = True
                if re.search(r'LOSS_BATCH(?:[^\|]*\|){1}([\d\.]+)', msg_content.upper()):
                    m = re.search(
                        r'LOSS_BATCH(?:[^\|]*\|){1}([\d\.]+)', msg_content.upper())
                    self.loss_batch_label.configure(
                        text=f"Batch Loss: {m.group(1)}")
                    parsed_telemetry = True
                if re.search(r'LOSS_AVG(?:[^\|]*\|){1}([\d\.]+)', msg_content.upper()):
                    m = re.search(
                        r'LOSS_AVG(?:[^\|]*\|){1}([\d\.]+)', msg_content.upper())
                    self.loss_avg_label.configure(
                        text=f"Avg. Loss: {m.group(1)}")
                    parsed_telemetry = True

                # If it's a generic telemetry message not caught above, log it as DEBUG
                if not parsed_telemetry and log_type == "TELEMETRY":
                    self.log_to_box(
                        f"[{timestamp}] [TELEMETRY] {msg_content}\n", "DEBUG")

            elif log_type == "PROCESS_LOG" or log_type == "PRIMARY_LOG":
                # Determine log level from message content if possible (e.g., "ERROR:", "WARNING:")
                log_level = "INFO"  # Default
                cleaned_msg = msg_content
                if "error" in msg_content.lower():
                    log_level = "ERROR"
                elif "warning" in msg_content.lower():
                    log_level = "WARNING"
                elif "debug" in msg_content.lower():
                    log_level = "DEBUG"
                elif "success" in msg_content.lower():
                    log_level = "SUCCESS"

                # Check for explicit log format like "LOG|LEVEL|Message"
                if len(parts) >= 3 and parts[0].upper() == "LOG":
                    log_level = parts[1].upper()
                    cleaned_msg = "|".join(parts[2:])

                self.log_to_box(
                    f"[{timestamp}] [{log_level}] {cleaned_msg}\n", log_level)
                if log_level == "ERROR" and "running" in self.status_indicator.cget("text").lower():
                    self.status_indicator.configure(
                        text=self.status_indicator.cget("text") + " (with errors)")

            elif log_type == "SYSTEM":
                log_level = "SYSTEM"
                if "error" in msg_content.lower():
                    log_level = "ERROR"
                elif "warning" in msg_content.lower():
                    log_level = "WARNING"
                self.log_to_box(
                    f"[{timestamp}] [{log_level}] {msg_content}\n", log_level)

        self.after(100, self.process_gui_queue)

    def log_to_box(self, msg, level="INFO"):
        self.log_textbox.configure(state="normal")
        # Message already contains timestamp and level from process_gui_queue
        # So we just pass the level for tagging.
        self.log_textbox.insert("end", msg, (level.upper(),))
        self.log_textbox.configure(state="disabled")
        self.log_textbox.see("end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
