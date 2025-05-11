import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import subprocess

class ConfiguratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Configurator")
        self.root.geometry("800x900")

        # Create main frame with scrollbar
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Initialize variables
        self.source_type = tk.StringVar(value="Live Camera")
        self.video_path = tk.StringVar()
        self.similarity_threshold = tk.DoubleVar(value=0.6)

        # Create sections
        self.create_input_source_section()
        self.create_recognition_section()
        self.create_output_visualization_section()
        self.create_alerting_section()
        self.create_save_load_section()

        # Initialize configuration
        self.config = {}

    def create_input_source_section(self):
        # Input Source Section
        input_frame = ttk.LabelFrame(self.scrollable_frame, text="Input Source Configuration", padding=10)
        input_frame.pack(fill="x", padx=5, pady=5)

        # Source Type
        ttk.Label(input_frame, text="Source Type:").pack(anchor="w")
        source_frame = ttk.Frame(input_frame)
        source_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(source_frame, text="Live Camera", variable=self.source_type,
                       value="Live Camera", command=self.update_source_specifics).pack(side="left", padx=5)
        ttk.Radiobutton(source_frame, text="Video File", variable=self.source_type,
                       value="Video File", command=self.update_source_specifics).pack(side="left", padx=5)

        # Source Specifics Frame
        self.source_specifics_frame = ttk.Frame(input_frame)
        self.source_specifics_frame.pack(fill="x", pady=5)

        # Video File Frame
        self.video_frame = ttk.Frame(self.source_specifics_frame)
        ttk.Label(self.video_frame, text="Video File:").pack(side="left")
        ttk.Entry(self.video_frame, textvariable=self.video_path, width=40).pack(side="left", padx=5)
        ttk.Button(self.video_frame, text="Browse",
                  command=self.browse_video).pack(side="left", padx=5)

    def create_recognition_section(self):
        # Recognition Section
        recognition_frame = ttk.LabelFrame(self.scrollable_frame, text="Recognition Configuration", padding=10)
        recognition_frame.pack(fill="x", padx=5, pady=5)

        # Similarity Threshold
        threshold_frame = ttk.Frame(recognition_frame)
        threshold_frame.pack(fill="x", pady=5)
        ttk.Label(threshold_frame, text="Recognition Similarity Threshold:").pack(side="left")
        ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.similarity_threshold,
                 orient="horizontal", length=200).pack(side="left", padx=5)
        ttk.Label(threshold_frame, textvariable=self.similarity_threshold).pack(side="left")

    def create_output_visualization_section(self):
        # Output & Visualization Section
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="Output & Visualization Configuration", padding=10)
        output_frame.pack(fill="x", padx=5, pady=5)

        # Bounding Boxes
        self.show_boxes = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Show Bounding Boxes",
                       variable=self.show_boxes).pack(anchor="w")

        # Box Colors
        colors_frame = ttk.Frame(output_frame)
        colors_frame.pack(fill="x", pady=5)
        ttk.Label(colors_frame, text="Box Color (No Match):").pack(side="left")
        self.no_match_color = tk.StringVar(value="#FF0000")  # Red in hex
        ttk.Entry(colors_frame, textvariable=self.no_match_color, width=10).pack(side="left", padx=5)

        ttk.Label(colors_frame, text="Box Color (Match):").pack(side="left")
        self.match_color = tk.StringVar(value="#00FF00")  # Green in hex
        ttk.Entry(colors_frame, textvariable=self.match_color, width=10).pack(side="left", padx=5)

        # Other visualization options
        self.show_confidence = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Show Detection Confidence",
                       variable=self.show_confidence).pack(anchor="w")

        self.show_similarity = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Show Recognition Similarity",
                       variable=self.show_similarity).pack(anchor="w")

        self.show_name = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Show Attendee Name/ID on Match",
                       variable=self.show_name).pack(anchor="w")

    def create_alerting_section(self):
        # Alerting Section
        alert_frame = ttk.LabelFrame(self.scrollable_frame, text="Alerting Configuration", padding=10)
        alert_frame.pack(fill="x", padx=5, pady=5)

        # Visual Alert
        visual_frame = ttk.LabelFrame(alert_frame, text="Visual Alert", padding=5)
        visual_frame.pack(fill="x", pady=5)

        self.show_text_alert = tk.BooleanVar(value=True)
        ttk.Checkbutton(visual_frame, text="Show Text Alert",
                       variable=self.show_text_alert).pack(anchor="w")

        text_frame = ttk.Frame(visual_frame)
        text_frame.pack(fill="x", pady=5)
        ttk.Label(text_frame, text="Match Text:").pack(side="left")
        self.match_text = tk.StringVar(value="ACCESS GRANTED")
        ttk.Entry(text_frame, textvariable=self.match_text, width=20).pack(side="left", padx=5)

        ttk.Label(text_frame, text="No Match Text:").pack(side="left")
        self.no_match_text = tk.StringVar(value="ACCESS DENIED")
        ttk.Entry(text_frame, textvariable=self.no_match_text, width=20).pack(side="left", padx=5)

        # Sound Alert
        sound_frame = ttk.LabelFrame(alert_frame, text="Sound Alert", padding=5)
        sound_frame.pack(fill="x", pady=5)

        self.enable_match_sound = tk.BooleanVar(value=True)
        ttk.Checkbutton(sound_frame, text="Enable Sound on Match",
                       variable=self.enable_match_sound).pack(anchor="w")

        self.match_sound_path = tk.StringVar()
        sound_match_frame = ttk.Frame(sound_frame)
        sound_match_frame.pack(fill="x", pady=5)
        ttk.Entry(sound_match_frame, textvariable=self.match_sound_path, width=40).pack(side="left", padx=5)
        ttk.Button(sound_match_frame, text="Browse",
                  command=lambda: self.browse_sound("match")).pack(side="left", padx=5)

        self.enable_no_match_sound = tk.BooleanVar(value=True)
        ttk.Checkbutton(sound_frame, text="Enable Sound on No Match",
                       variable=self.enable_no_match_sound).pack(anchor="w")

        self.no_match_sound_path = tk.StringVar()
        sound_no_match_frame = ttk.Frame(sound_frame)
        sound_no_match_frame.pack(fill="x", pady=5)
        ttk.Entry(sound_no_match_frame, textvariable=self.no_match_sound_path, width=40).pack(side="left", padx=5)
        ttk.Button(sound_no_match_frame, text="Browse",
                  command=lambda: self.browse_sound("no_match")).pack(side="left", padx=5)

    def create_save_load_section(self):
        # Save/Load Section
        save_frame = ttk.LabelFrame(self.scrollable_frame, text="Save/Load Configuration", padding=10)
        save_frame.pack(fill="x", padx=5, pady=5)

        # Config file path
        path_frame = ttk.Frame(save_frame)
        path_frame.pack(fill="x", pady=5)
        ttk.Label(path_frame, text="Configuration File:").pack(side="left")
        self.config_path = tk.StringVar(value="config/config.json")
        ttk.Entry(path_frame, textvariable=self.config_path, width=40).pack(side="left", padx=5)
        ttk.Button(path_frame, text="Browse",
                  command=self.browse_config).pack(side="left", padx=5)

        # Buttons
        button_frame = ttk.Frame(save_frame)
        button_frame.pack(fill="x", pady=5)
        ttk.Button(button_frame, text="Save Configuration",
                  command=self.save_config).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load Configuration",
                  command=self.load_config).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Start Process",
                  command=self.start_process).pack(side="left", padx=5)

    def update_source_specifics(self):
        # Clear all frames
        for widget in self.source_specifics_frame.winfo_children():
            widget.pack_forget()

        # Show appropriate frame based on source type
        if self.source_type.get() == "Video File":
            self.video_frame.pack(fill="x")

    def refresh_cameras(self):
        # This is a placeholder - in a real implementation, you would detect available cameras
        messagebox.showinfo("Info", "Camera refresh functionality would be implemented here")

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)

    def browse_sound(self, sound_type):
        filename = filedialog.askopenfilename(
            title=f"Select {sound_type} Sound File",
            filetypes=[("Sound files", "*.wav *.mp3"), ("All files", "*.*")]
        )
        if filename:
            if sound_type == "match":
                self.match_sound_path.set(filename)
            else:
                self.no_match_sound_path.set(filename)

    def browse_config(self):
        filename = filedialog.asksaveasfilename(
            title="Select Configuration File",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.config_path.set(filename)

    def save_config(self):
        config = {
            "input_source": {
                "type": self.source_type.get(),
                "video_path": self.video_path.get()
            },
            "recognition": {
                "similarity_threshold": self.similarity_threshold.get()
            },
            "visualization": {
                "show_boxes": self.show_boxes.get(),
                "no_match_color": self.no_match_color.get(),
                "match_color": self.match_color.get(),
                "show_confidence": self.show_confidence.get(),
                "show_similarity": self.show_similarity.get(),
                "show_name": self.show_name.get()
            },
            "alerts": {
                "visual": {
                    "show_text": self.show_text_alert.get(),
                    "match_text": self.match_text.get(),
                    "no_match_text": self.no_match_text.get()
                },
                "sound": {
                    "enable_match": self.enable_match_sound.get(),
                    "match_sound": self.match_sound_path.get(),
                    "enable_no_match": self.enable_no_match_sound.get(),
                    "no_match_sound": self.no_match_sound_path.get()
                }
            }
        }

        # Ensure config directory exists
        config_dir = os.path.dirname(self.config_path.get())
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)

        try:
            with open(self.config_path.get(), 'w') as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def load_config(self):
        try:
            with open(self.config_path.get(), 'r') as f:
                config = json.load(f)

            # Update input source
            self.source_type.set(config["input_source"]["type"])
            self.video_path.set(config["input_source"]["video_path"])

            # Update recognition
            self.similarity_threshold.set(config["recognition"]["similarity_threshold"])

            # Update visualization
            self.show_boxes.set(config["visualization"]["show_boxes"])
            self.no_match_color.set(config["visualization"]["no_match_color"])
            self.match_color.set(config["visualization"]["match_color"])
            self.show_confidence.set(config["visualization"]["show_confidence"])
            self.show_similarity.set(config["visualization"]["show_similarity"])
            self.show_name.set(config["visualization"]["show_name"])

            # Update alerts
            self.show_text_alert.set(config["alerts"]["visual"]["show_text"])
            self.match_text.set(config["alerts"]["visual"]["match_text"])
            self.no_match_text.set(config["alerts"]["visual"]["no_match_text"])
            self.enable_match_sound.set(config["alerts"]["sound"]["enable_match"])
            self.match_sound_path.set(config["alerts"]["sound"]["match_sound"])
            self.enable_no_match_sound.set(config["alerts"]["sound"]["enable_no_match"])
            self.no_match_sound_path.set(config["alerts"]["sound"]["no_match_sound"])

            self.update_source_specifics()
            messagebox.showinfo("Success", "Configuration loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def start_process(self):
        try:
            # Get the absolute path to the config file
            config_path = os.path.abspath(self.config_path.get())

            # Get the absolute path to the main.py file
            main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.py"))

            # Construct the command
            cmd = ["python3", main_path, "--load", config_path, "-m", "recognition"]

            # Start the process
            subprocess.Popen(cmd)
            messagebox.showinfo("Success", "Process started successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start process: {str(e)}")

def main():
    root = tk.Tk()
    app = ConfiguratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
