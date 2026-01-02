import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageTk
import os
import json
import xml.etree.ElementTree as ET

# --- Core Image Processing Functions (No Changes Here) ---

def load_yolo_classes(classes_file_path):
    """Loads class names from a .txt or .names file."""
    class_names = []
    if classes_file_path and os.path.exists(classes_file_path):
        try:
            with open(classes_file_path, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading classes file {classes_file_path}: {e}")
    return class_names

def parse_annotations(annotation_path, image_width, image_height, class_names=None):
    """
    Parses annotations from JSON (custom), XML (COCO-like), or TXT (YOLO) files.
    Returns a list of dictionaries, each containing 'label', 'x_min', 'y_min', 'x_max', 'y_max'.
    """
    annotations = []
    file_extension = os.path.splitext(annotation_path)[1].lower()

    try:
        if file_extension == '.json':
            with open(annotation_path, 'r') as f:
                annotations_data = json.load(f)

            if annotations_data and isinstance(annotations_data, list) and 'annotations' in annotations_data[0]:
                for anno in annotations_data[0]['annotations']:
                    label = anno.get('label', 'No Label')
                    coords = anno.get('coordinates')
                    if coords:
                        x = coords.get('x')
                        y = coords.get('y')
                        width = coords.get('width')
                        height = coords.get('height')
                        if all(v is not None for v in [x, y, width, height]):
                            annotations.append({
                                'label': label,
                                'x_min': int(x),
                                'y_min': int(y),
                                'x_max': int(x + width),
                                'y_max': int(y + height)
                            })
            else:
                print(f"Warning: JSON structure not as expected for {annotation_path}. 'annotations' key not found in the first object.")

        elif file_extension == '.xml':
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                label = obj.find('name').text if obj.find('name') is not None else 'No Label'
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    x_min = int(float(bndbox.find('xmin').text))
                    y_min = int(float(bndbox.find('ymin').text))
                    x_max = int(float(bndbox.find('xmax').text))
                    y_max = int(float(bndbox.find('ymax').text))
                    annotations.append({
                        'label': label,
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max
                    })

        elif file_extension == '.txt': # YOLO format: class_id x_center y_center width height (normalized)
            if not class_names:
                print(f"Warning: No class names provided for YOLO .txt file: {annotation_path}. Labels will be class IDs.")
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])

                        x_center = x_center_norm * image_width
                        y_center = y_center_norm * image_height
                        bbox_width = width_norm * image_width
                        bbox_height = height_norm * image_height

                        x_min = int(x_center - bbox_width / 2)
                        y_min = int(y_center - bbox_height / 2)
                        x_max = int(x_center + bbox_width / 2)
                        y_max = int(y_center + bbox_height / 2)

                        label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
                        annotations.append({
                            'label': label,
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max
                        })
        else:
            print(f"Unsupported annotation file format: {file_extension} for {annotation_path}")

    except FileNotFoundError:
        print(f"Annotation file not found: {annotation_path}")
    except (json.JSONDecodeError, ET.ParseError, ValueError) as e:
        print(f"Error parsing annotation file {annotation_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while parsing {annotation_path}: {e}")

    return annotations

def draw_bounding_boxes(image_path, annotations, draw_on_copy=True):
    """
    Draws bounding boxes on an image based on parsed annotations.
    Returns the PIL Image object.
    """
    try:
        original_image = Image.open(image_path).convert("RGB")
        image_to_draw = original_image.copy() if draw_on_copy else original_image
        draw = ImageDraw.Draw(image_to_draw)

        for annotation in annotations:
            label = annotation.get('label', 'No Label')
            x_min, y_min, x_max, y_max = annotation.get('x_min'), annotation.get('y_min'), annotation.get('x_max'), annotation.get('y_max')

            if all(v is not None for v in [x_min, y_min, x_max, y_max]):
                if "silver marking" in label.lower():
                    box_color = "yellow"
                elif "balancing weight" in label.lower():
                    box_color = "cyan"
                else:
                    box_color = "red" 

                draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=3)
                text_position = (x_min, y_min - 20 if y_min - 20 > 0 else y_min + 5)
                draw.text(text_position, label, fill=box_color)
            else:
                print(f"Warning: Missing coordinate data in an annotation: {annotation}")
        return image_to_draw
    except FileNotFoundError as e:
        print(f"Error: Image file not found. {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during drawing: {e}")
        return None

# --- Tkinter GUI Application ---

class BoundingBoxViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Inspection Tool")
        self.root.geometry("1200x800")

        self.image_path = tk.StringVar()
        self.label_path = tk.StringVar()
        self.classes_path = tk.StringVar()

        self.original_image_pil = None
        self.current_photo_image = None
        self.zoom_level = 1.0
        # NEW: State variables for stable centered zooming
        self.image_x_offset = 0
        self.image_y_offset = 0

        self.image_files_for_auto = []
        self.current_image_index = 0

        self.drawings_on_canvas = []
        self.last_draw_x, self.last_draw_y = None, None
        self.drawing_active = False

        self.current_mode = "pan"
        self.is_fullscreen = False

        self.style = ttk.Style()
        self.style.configure('Active.TButton', background='lightblue', foreground='black')
        self.style.configure('Inactive.TButton', background='lightgray', foreground='black')
        
        self._after_id = None # For debouncing resize events

        self.create_widgets()
        self.setup_canvas_bindings()

        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)


    def create_widgets(self):
        # --- Frame for File Selection (Manual Mode) ---
        self.file_frame = ttk.LabelFrame(self.root, text="Manual File Selection")
        self.file_frame.pack(pady=10, padx=10, fill="x")
        # (Widgets for manual selection remain unchanged)
        ttk.Label(self.file_frame, text="Image File (JPG/PNG/JPEG):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.file_frame, textvariable=self.image_path, width=70).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.file_frame, text="Browse", command=lambda: self.browse_file(self.image_path, [("Image Files", "*.jpg *.jpeg *.png")])).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(self.file_frame, text="Label File (XML/JSON/TXT):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.file_frame, textvariable=self.label_path, width=70).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.file_frame, text="Browse", command=lambda: self.browse_file(self.label_path, [("Annotation Files", "*.xml *.json *.txt")])).grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(self.file_frame, text="Classes File (Optional - for YOLO):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.file_frame, textvariable=self.classes_path, width=70).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.file_frame, text="Browse", command=lambda: self.browse_file(self.classes_path, [("Text Files", "*.txt *.names")])).grid(row=2, column=2, padx=5, pady=5)
        ttk.Button(self.file_frame, text="Process Single Image", command=self.process_single_image).grid(row=3, column=1, columnspan=2, pady=10)
        self.file_frame.grid_columnconfigure(1, weight=1)

        # --- Frame for Automated Processing ---
        self.auto_frame = ttk.LabelFrame(self.root, text="Automated Folder Processing")
        self.auto_frame.pack(pady=10, padx=10, fill="x")
        # (Widgets for automated folder selection remain unchanged)
        self.auto_folder_path = tk.StringVar()
        ttk.Label(self.auto_frame, text="Select Root Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.auto_frame, textvariable=self.auto_folder_path, width=70).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.auto_frame, text="Browse Folder", command=self.browse_auto_folder).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.auto_frame, text="Start Automated Processing", command=self.start_automated_processing).grid(row=1, column=1, columnspan=2, pady=10)
        self.auto_frame.grid_columnconfigure(1, weight=1)
        
        # --- MODIFIED: Navigation Frame with Direct Navigation Box ---
        self.nav_frame = ttk.Frame(self.root)
        self.nav_frame.pack(pady=5, padx=10, fill="x")
        self.prev_button = ttk.Button(self.nav_frame, text="Previous", command=self.show_previous_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.image_counter_label = ttk.Label(self.nav_frame, text="0/0")
        self.image_counter_label.pack(side=tk.LEFT, padx=10, expand=True)

        self.next_button = ttk.Button(self.nav_frame, text="Next", command=self.show_next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT, padx=5)
        self.nav_button = ttk.Button(self.nav_frame, text="Go", command=self.go_to_image, state=tk.DISABLED)
        self.nav_button.pack(side=tk.RIGHT)
        self.nav_entry_var = tk.StringVar()
        self.nav_entry = ttk.Entry(self.nav_frame, textvariable=self.nav_entry_var, width=6, justify='center', state=tk.DISABLED)
        self.nav_entry.pack(side=tk.RIGHT, padx=5)
        self.nav_entry.bind("<Return>", lambda event: self.go_to_image())

        # --- Tool Mode Selection and Save Button ---
        self.tool_frame = ttk.LabelFrame(self.root, text="Tools")
        self.tool_frame.pack(pady=10, padx=10, fill="x")
        # (Tool widgets remain unchanged)
        self.pan_mode_btn = ttk.Button(self.tool_frame, text="Pan Tool", command=lambda: self.set_mode("pan"))
        self.pan_mode_btn.pack(side=tk.LEFT, padx=5)
        self.draw_mode_btn = ttk.Button(self.tool_frame, text="Paint Tool", command=lambda: self.set_mode("draw"))
        self.draw_mode_btn.pack(side=tk.LEFT, padx=5)
        self.erase_mode_btn = ttk.Button(self.tool_frame, text="Erase Tool", command=lambda: self.set_mode("erase"))
        self.erase_mode_btn.pack(side=tk.LEFT, padx=5)
        self.fullscreen_btn = ttk.Button(self.tool_frame, text="Fullscreen (F11)", command=self.toggle_fullscreen)
        self.fullscreen_btn.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(self.tool_frame, text="Save Annotated Image", command=self.save_current_image)
        self.save_button.pack(side=tk.RIGHT, padx=5)

        # --- Image Display Area ---
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.canvas = tk.Canvas(self.image_frame, bg="gray", bd=2, relief="sunken")
        
        self.xscroll = ttk.Scrollbar(self.image_frame, orient="horizontal", command=self.canvas.xview)
        self.yscroll = ttk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.config(xscrollcommand=self.xscroll.set, yscrollcommand=self.yscroll.set)

        self.yscroll.pack(side="right", fill="y")
        self.xscroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # --- Status Bar ---
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.set_mode("pan")

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)
        if self.is_fullscreen:
            for w in [self.file_frame, self.auto_frame, self.nav_frame]:
                w.pack_forget()
        else:
            self.file_frame.pack(pady=10, padx=10, fill="x")
            self.auto_frame.pack(pady=10, padx=10, fill="x")
            self.nav_frame.pack(pady=5, padx=10, fill="x")
        self.root.after(50, self.fit_image_to_canvas)

    def exit_fullscreen(self, event=None):
        if self.is_fullscreen:
            self.toggle_fullscreen()
    
    # NEW: Function to navigate to a specific image number
    def go_to_image(self):
        if not self.image_files_for_auto: return
        try:
            target_index = int(self.nav_entry_var.get()) - 1
            if 0 <= target_index < len(self.image_files_for_auto):
                if self.current_image_index != target_index:
                    self.current_image_index = target_index
                    self.show_current_automated_image()
            else:
                messagebox.showerror("Invalid Number", f"Please enter a number between 1 and {len(self.image_files_for_auto)}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")
        finally:
            self.nav_entry_var.set(str(self.current_image_index + 1))

    def set_mode(self, mode):
        self.current_mode = mode
        self.status_bar.config(text=f"Mode: {mode.capitalize()} Tool - Use Ctrl+Scroll for Zoom")
        for btn, m in [(self.pan_mode_btn, "pan"), (self.draw_mode_btn, "draw"), (self.erase_mode_btn, "erase")]:
            btn.config(style='Active.TButton' if mode == m else 'Inactive.TButton')
        self.setup_canvas_bindings()

    def browse_file(self, var, filetypes):
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            var.set(file_path)
            self.drawings_on_canvas.clear()

    def browse_auto_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.auto_folder_path.set(folder_path)
            self.drawings_on_canvas.clear()

    def display_image_on_canvas(self, pil_image):
        if pil_image:
            self.original_image_pil = pil_image
            self.fit_image_to_canvas()

    # REWRITTEN: Handles stable zooming, centering, and aspect ratio
    def redraw_canvas(self):
        if not self.original_image_pil:
            self.canvas.delete("all")
            return

        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_w, img_h = self.original_image_pil.size

        disp_w = int(img_w * self.zoom_level)
        disp_h = int(img_h * self.zoom_level)

        self.canvas.config(scrollregion=(0, 0, max(disp_w, canvas_w), max(disp_h, canvas_h)))

        self.image_x_offset = max(0, (canvas_w - disp_w) / 2)
        self.image_y_offset = max(0, (canvas_h - disp_h) / 2)

        if disp_w > 0 and disp_h > 0:
            resized_image = self.original_image_pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
            self.current_photo_image = ImageTk.PhotoImage(resized_image)

            self.canvas.delete("all")
            self.canvas.create_image(self.image_x_offset, self.image_y_offset, anchor="nw", image=self.current_photo_image)

            for coords, dtype, options in self.drawings_on_canvas:
                scaled_coords = [c * self.zoom_level for c in coords]
                offset_coords = [scaled_coords[i] + (self.image_x_offset if i % 2 == 0 else self.image_y_offset) for i in range(len(scaled_coords))]
                if dtype == "line":
                    self.canvas.create_line(offset_coords, **{k:v for k,v in options.items() if k != 'canvas_id'})
    
    # NEW: Calculates the best zoom level to fit the image and calls redraw
    def fit_image_to_canvas(self):
        if not self.original_image_pil: return
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_w, img_h = self.original_image_pil.size
        if canvas_w <= 1 or canvas_h <= 1 or img_w <= 0 or img_h <= 0: return
        
        self.zoom_level = min(canvas_w / img_w, canvas_h / img_h)
        self.redraw_canvas()

    # MODIFIED: Debounces redraw calls on resize for performance
    def on_canvas_configure(self, event):
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(50, self.fit_image_to_canvas)

    def setup_canvas_bindings(self):
        for seq in ["<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>", "<ButtonPress-2>", "<B2-Motion>", "<ButtonRelease-2>", "<Control-MouseWheel>", "<Control-Button-4>", "<Control-Button-5>"]:
            self.canvas.unbind(seq)

        self.canvas.bind("<Control-MouseWheel>", self.on_mouse_wheel_zoom)
        self.canvas.bind("<Control-Button-4>", self.on_mouse_wheel_zoom) # Linux zoom in
        self.canvas.bind("<Control-Button-5>", self.on_mouse_wheel_zoom) # Linux zoom out

        self.canvas.bind("<ButtonPress-2>", self.start_pan) # Middle mouse button pan
        self.canvas.bind("<B2-Motion>", self.pan_image)

        if self.current_mode == "pan":
            self.canvas.config(cursor="hand2")
            self.canvas.bind("<ButtonPress-1>", self.start_pan)
            self.canvas.bind("<B1-Motion>", self.pan_image)
        elif self.current_mode == "draw":
            self.canvas.config(cursor="cross")
            self.canvas.bind("<ButtonPress-1>", self.start_draw)
            self.canvas.bind("<B1-Motion>", self.do_draw)
            self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        elif self.current_mode == "erase":
            self.canvas.config(cursor="dotbox")
            self.canvas.bind("<B1-Motion>", self.do_erase) # Erase on drag

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan_image(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # REWRITTEN: For stable, cursor-centric zooming
    def on_mouse_wheel_zoom(self, event):
        if self.original_image_pil is None: return

        zoom_factor = 1.1 if (event.delta > 0 or event.num == 4) else 1/1.1
        
        # Determine the point on the canvas to zoom into
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Apply the zoom
        self.canvas.scale("all", x, y, zoom_factor, zoom_factor)
        self.zoom_level *= zoom_factor
        self.redraw_canvas() # Redraw to recenter if needed and fix scrollregion
        self.status_bar.config(text=f"Zoom: {self.zoom_level:.2f}x")

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Converts canvas coordinates to original image coordinates."""
        img_x = (self.canvas.canvasx(canvas_x) - self.image_x_offset) / self.zoom_level
        img_y = (self.canvas.canvasy(canvas_y) - self.image_y_offset) / self.zoom_level
        return img_x, img_y

    def start_draw(self, event):
        if self.original_image_pil is None: return
        self.drawing_active = True
        self.last_draw_x, self.last_draw_y = self._canvas_to_image_coords(event.x, event.y)
        self.current_drawing_coords_unscaled = [self.last_draw_x, self.last_draw_y]
        
    def do_draw(self, event):
        if not self.drawing_active: return
        x, y = self._canvas_to_image_coords(event.x, event.y)
        self.current_drawing_coords_unscaled.extend([x, y])
        
        # Create a temporary line on the canvas for visual feedback
        self.redraw_canvas() # Redraw base image
        # Draw the line currently being drawn
        scaled_coords = [c * self.zoom_level for c in self.current_drawing_coords_unscaled]
        offset_coords = [scaled_coords[i] + (self.image_x_offset if i % 2 == 0 else self.image_y_offset) for i in range(len(scaled_coords))]
        self.canvas.create_line(offset_coords, fill="cyan", width=2, tags="temp_drawing")
        self.last_draw_x, self.last_draw_y = x, y

    def end_draw(self, event):
        if not self.drawing_active: return
        self.canvas.delete("temp_drawing")
        self.drawings_on_canvas.append(
            (list(self.current_drawing_coords_unscaled), "line", {"fill": "blue", "width": 2})
        )
        self.drawing_active = False
        self.redraw_canvas()

    def do_erase(self, event):
        if self.original_image_pil is None: return
        erase_x, erase_y = self._canvas_to_image_coords(event.x, event.y)
        
        # Find drawings close to the cursor on the original image scale
        tolerance = 5 / self.zoom_level # 5 pixel tolerance on screen
        original_len = len(self.drawings_on_canvas)
        
        self.drawings_on_canvas = [
            drawing for drawing in self.drawings_on_canvas 
            if not any(
                abs(drawing[0][i] - erase_x) < tolerance and abs(drawing[0][i+1] - erase_y) < tolerance
                for i in range(0, len(drawing[0]), 2)
            )
        ]
        if len(self.drawings_on_canvas) < original_len:
            self.redraw_canvas()

    def process_single_image(self):
        image_p, label_p, classes_p = self.image_path.get(), self.label_path.get(), self.classes_path.get()
        if not image_p or not os.path.exists(image_p) or not label_p or not os.path.exists(label_p):
            messagebox.showerror("Error", "Please select valid image and label files.")
            return
        try:
            self.status_bar.config(text="Processing...")
            with Image.open(image_p) as img:
                img_width, img_height = img.size
            
            class_names = load_yolo_classes(classes_p) if classes_p else None
            annotations = parse_annotations(label_p, img_width, img_height, class_names)
            
            annotated_image_pil = draw_bounding_boxes(image_p, annotations)
            if annotated_image_pil:
                self.drawings_on_canvas.clear()
                self.display_image_on_canvas(annotated_image_pil)
                self.status_bar.config(text=f"Successfully processed {os.path.basename(image_p)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def start_automated_processing(self):
        root_folder = self.auto_folder_path.get()
        if not root_folder or not os.path.isdir(root_folder):
            messagebox.showerror("Error", "Please select a valid root folder.")
            return

        self.status_bar.config(text="Scanning folder...")
        images_dir = os.path.join(root_folder, "images")
        labels_dir = os.path.join(root_folder, "labels")
        classes_file = next((os.path.join(root_folder, f) for f in ["classes.txt", "classes.names"] if os.path.exists(os.path.join(root_folder, f))), None)

        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            images_dir, labels_dir = root_folder, root_folder
        
        image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_names = load_yolo_classes(classes_file)
        
        found_pairs = []
        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            for ext in ['.json', '.xml', '.txt']:
                label_path = os.path.join(labels_dir, base_name + ext)
                if os.path.exists(label_path):
                    found_pairs.append((img_path, label_path, class_names))
                    break
        
        self.image_files_for_auto = found_pairs
        if not self.image_files_for_auto:
            messagebox.showinfo("Info", "No image-label pairs found.")
            return

        self.current_image_index = 0
        for w, state in [(self.prev_button, tk.NORMAL), (self.next_button, tk.NORMAL), (self.nav_entry, tk.NORMAL), (self.nav_button, tk.NORMAL)]:
            w.config(state=state)
        self.show_current_automated_image()

    def show_current_automated_image(self):
        if not self.image_files_for_auto: return
        
        img_path, label_path, class_names = self.image_files_for_auto[self.current_image_index]
        self.status_bar.config(text=f"Loading image {self.current_image_index + 1}/{len(self.image_files_for_auto)}...")
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            annotations = parse_annotations(label_path, img_width, img_height, class_names)
            annotated_image_pil = draw_bounding_boxes(img_path, annotations)
            if annotated_image_pil:
                self.drawings_on_canvas.clear()
                self.display_image_on_canvas(annotated_image_pil)
                self.image_counter_label.config(text=f"{self.current_image_index + 1}/{len(self.image_files_for_auto)}")
                self.nav_entry_var.set(str(self.current_image_index + 1))
                self.status_bar.config(text=f"Viewing {os.path.basename(img_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load {os.path.basename(img_path)}: {e}")

    def show_next_image(self):
        if self.current_image_index < len(self.image_files_for_auto) - 1:
            self.current_image_index += 1
            self.show_current_automated_image()

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_automated_image()

    def save_current_image(self):
        if self.original_image_pil is None:
            messagebox.showwarning("Warning", "No image loaded to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
            title="Save Annotated Image"
        )
        if not file_path: return

        try:
            image_to_save = self.original_image_pil.copy()
            draw = ImageDraw.Draw(image_to_save)
            for coords, dtype, options in self.drawings_on_canvas:
                if dtype == "line":
                    draw.line(coords, fill=options.get("fill", "blue"), width=options.get("width", 2))
            
            image_to_save.save(file_path)
            self.status_bar.config(text=f"Image saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BoundingBoxViewerApp(root)
    root.mainloop()
