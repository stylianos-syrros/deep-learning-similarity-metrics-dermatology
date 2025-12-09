import tkinter as tk
from tkinter import filedialog, messagebox
from torchvision.utils import save_image
from PIL import Image, ImageTk, ImageDraw
from DISTS_pytorch import DISTS
import lpips
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import os, sys
from torchvision import models, transforms
import torch.nn as nn

warnings.filterwarnings("ignore")

class ImageSelectorApp:
    def __init__(self, master):
        self.region1_checked = False  # Μεταβλητή για το εάν έχει γίνει επιλογή του Region 1
        self.region2_checked = False  # Μεταβλητή για το εάν έχει γίνει επιλογή του Region 2
        
        self.master = master
        self.master.title("Image Selector")

        self.image_path = tk.StringVar()
        self.image_path.set("")

        self.select_button = tk.Button(master, text="Select Option", command=self.select_option)
        self.select_button.pack()

        self.image_coords = None
        self.region_window = None
        self.draw = None

    def select_option(self):
        option_window = tk.Toplevel(self.master)
        option_window.title("Select Option")
        
        option1_button = tk.Button(option_window, text="Select Two Different Images", command=self.select_two_images)
        option1_button.pack()

        option2_button = tk.Button(option_window, text="Select One Image with Regions", command=self.select_image)
        option2_button.pack()

    def select_two_images(self):
        self.image1_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        self.image2_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        
        if self.image1_path and self.image2_path:
            self.calculate_dists_for_two_images()
            self.calculate_lpips_for_two_images()

    @staticmethod
    def compare_images(image_path1, image_path2):
        # Άνοιγμα των εικόνων
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)
        
        # Μετατροπή των εικόνων σε numpy arrays
        image1_array = np.array(image1)
        image2_array = np.array(image2)
        
        # Έλεγχος αν οι διαστάσεις των εικόνων είναι ίδιες
        if image1_array.shape != image2_array.shape:
            return False
        
        # Σύγκριση των εικόνων
        comparison = np.array_equal(image1_array, image2_array)
        
        return comparison

    def calculate_dists_for_two_images(self):
        # Φορτώνουμε τις δύο εικόνες
        image1 = Image.open(self.image1_path).convert("RGB")
        image2 = Image.open(self.image2_path).convert("RGB")
        image1 = transforms.functional.resize(image1, (224, 224))
        image2 = transforms.functional.resize(image2, (224, 224))
        transform = transforms.ToTensor()

        image1 = transform(image1).unsqueeze(0)
        image2 = transform(image2).unsqueeze(0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = DISTS().to(device)
        image1 = image1.to(device)
        image2 = image2.to(device)

        # Υπολογισμός DISTS
        score = model(image1, image2)
        print("DISTS Score: ",score.item())    
        messagebox.showinfo("DISTS Score", f"DISTS Score for the two images: {score}")          

    def calculate_lpips_for_two_images(self):
        # Δημιουργία αντικειμένου LPIPS για το δίκτυο VGG
        loss_fn_vgg = lpips.LPIPS(net='vgg')
        
        # Φορτώνουμε τις εικόνες με χρήση του lpips
        img0 = lpips.im2tensor(lpips.load_image(self.image1_path))
        img1 = lpips.im2tensor(lpips.load_image(self.image2_path))
        
        # Αλλαγή μεγέθους των εικόνων σε (224, 224)
        img0 = TF.resize(img0, (224, 224))
        img1 = TF.resize(img1, (224, 224))
        
        # Υπολογισμός της μετρικής LPIPS για τις δύο εικόνες χρησιμοποιώντας το δίκτυο VGG
        d_vgg = loss_fn_vgg(img0, img1)
        print("Right here")
        print("LPIPS Score(VGG):", d_vgg.item())  
        messagebox.showinfo("LPIPS Score(VGG)", f"LPIPS Score for the two images: {d_vgg.item()}") 

    def select_image(self):
        # Επαναφορά όλων των μεταβλητών
        self.image_path.set("")
        self.region1_coords = None
        self.region2_coords = None
        self.region1_checked = False  # Επαναφορά της μεταβλητής όταν επιλέγεται νέα εικόνα
        self.region2_checked = False

        # Καθαρισμός της εμφάνισης των περιοχών
        if hasattr(self, 'region_window') and self.region_window:
            self.region_window.destroy()

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])
        if file_path:
            self.image_path.set(file_path)
            self.display_image(file_path)
            self.image = Image.open(file_path)
            self.select_regions()

    def create_draw(self):
        self.image_pil = Image.open(self.image_path.get())
        self.draw = ImageDraw.Draw(self.image_pil)

    def select_regions(self):
        self.region_window = tk.Toplevel(self.master)
        self.region_window.title("Select Regions")

        self.create_draw()

        image = Image.open(self.image_path.get())
        self.photo = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(self.region_window, image=self.photo)
        self.image_label.image = self.photo
        self.image_label.pack()

        self.region1_var = tk.IntVar()
        self.region1_checkbutton = tk.Checkbutton(self.region_window, text="Select Region 1", variable=self.region1_var, command=lambda img=image: self.select_region_1(img))
        self.region1_checkbutton.pack()

        self.region2_var = tk.IntVar()
        self.region2_checkbutton = tk.Checkbutton(self.region_window, text="Select Region 2", variable=self.region2_var, command=self.select_region_2)
        self.region2_checkbutton.pack()

    def select_region_1(self, image):
        self.region1_checked = True  # Επισημαίνει ότι έχει γίνει επιλογή του Region 1

        self.region1_frame = tk.Frame(self.region_window, bd=2, relief=tk.SUNKEN)
        self.region1_frame.pack(padx=10, pady=10)

        self.instruction_label = tk.Label(self.region1_frame, text="Click and drag to select Region 1")
        self.instruction_label.pack()

        self.region1_coords = None

        self.image_label.bind("<Button-1>", self.start_selection)
        self.image_label.bind("<B1-Motion>", self.track_selection)
        self.image_label.bind("<ButtonRelease-1>", self.end_selection)

        # Ενεργοποίηση του κουμπίου "Select Region 2" αφού έχει επιλεγεί το Region 1
        self.region2_checkbutton.config(state=tk.NORMAL)

    def start_selection(self, event):
        self.start_x = self.region_window.winfo_pointerx() - self.region_window.winfo_rootx()
        self.start_y = self.region_window.winfo_pointery() - self.region_window.winfo_rooty()

    def track_selection(self, event):
        x = self.region_window.winfo_pointerx() - self.region_window.winfo_rootx()
        y = self.region_window.winfo_pointery() - self.region_window.winfo_rooty()
        x1 = min(self.start_x, x)
        x2 = max(self.start_x, x)
        y1 = min(self.start_y, y)
        y2 = max(self.start_y, y)
        self.draw.rectangle([x1, y1, x2, y2], outline="red")
    
    def end_selection(self, event):
        x = self.region_window.winfo_pointerx() - self.region_window.winfo_rootx()
        y = self.region_window.winfo_pointery() - self.region_window.winfo_rooty()
        
        # Ταξινομήστε τις συντεταγμένες
        x1, x2 = sorted([self.start_x, x])
        y1, y2 = sorted([self.start_y, y])
        
        # Έλεγχος αναφοράς στο region1 ή region2
        if self.region2_checked:
            self.region2_coords = [x1, y1, x2, y2]
            print("The region2 coordinates are :",self.region2_coords)
            self.display_image_coordinates(self.region2_coords)
        elif self.region1_checked:
            self.region1_coords = [x1, y1, x2, y2]
            print("The region1 coordinates are :",self.region1_coords)
            self.display_image_coordinates(self.region1_coords)
        else:
            # Αν δεν έχει επιλεγεί κάποιο region, δεν κάνουμε τίποτα
            return

    def select_region_2(self):
        if not self.region1_checked:
            messagebox.showerror("Error", "Please select Region 1 first.")
            self.region2_var.set(0)  # Αποεπιλογή του κουμπιού "Select Region 2" 
            return

        self.region2_checked = True  # Επισημαίνει ότι έχει γίνει επιλογή του Region 2

        self.region2_frame = tk.Frame(self.region_window, bd=2, relief=tk.SUNKEN)
        self.region2_frame.pack(padx=10, pady=10)

        self.instruction_label = tk.Label(self.region2_frame, text="Click and drag to select Region 2")
        self.instruction_label.pack()

        self.region2_coords = None

        self.image_label.bind("<Button-1>", self.start_selection)
        self.image_label.bind("<B1-Motion>", self.track_selection)
        self.image_label.bind("<ButtonRelease-1>", self.end_selection)

    def display_image_coordinates(self, coordinates):
        if self.region1_checked and self.region2_checked:
            print("Both regions are checked")

            # Συντεταγμένες των επιλεγμένων περιοχών
            region1_coords = self.region1_coords  # [x1, y1, x2, y2]
            region2_coords = self.region2_coords  # [x1, y1, x2, y2]

            # Φορτώνουμε την εικόνα
            image_path = self.image_path.get() if isinstance(self.image_path, tk.StringVar) else self.image_path
            original_image = Image.open(image_path).convert("RGB")
            transform = transforms.ToTensor()
            original_image = transform(original_image).unsqueeze(0)

            # Κόβουμε τις περιοχές
            region1_img = original_image[:, :, region1_coords[1]:region1_coords[3], region1_coords[0]:region1_coords[2]]
            region2_img = original_image[:, :, region2_coords[1]:region2_coords[3], region2_coords[0]:region2_coords[2]]

            # Υπολογισμός DISTS
            dists_score = self.calculate_DISTS(region1_img, region2_img)
            # Υπολογισμός LPIPS
            lpips_score = self.calculate_LPIPS(region1_img, region2_img)
        else:
            print("Only one region is checked")

        # Αποκτήστε τις συντεταγμένες της περιοχής
        x1, y1, x2, y2 = coordinates
        
        # Δημιουργήστε μια νέα εικόνα με την περιοχή που επιλέχθηκε
        cropped_image = self.image.crop((x1, y1, x2, y2))
        
        # Εμφανίστε την εικόνα σε ένα νέο παράθυρο
        new_window = tk.Toplevel(self.master)
        new_window.title("Selected Region")
        photo = ImageTk.PhotoImage(cropped_image)
        label = tk.Label(new_window, image=photo)
        label.image = photo
        label.pack()
        
        # Δέστε την μέθοδο maximize_image με το κουμπί μεγιστοποίησης
        maximize_button = tk.Button(new_window, text="Maximize", command=lambda: self.maximize_image(label, cropped_image))
        maximize_button.pack()

    def calculate_DISTS(self, region1_img, region2_img):
        # Αφαίρεση της διάστασης batch (πρώτη διάσταση)
        region1_img = region1_img.squeeze(0)
        region2_img = region2_img.squeeze(0)

        # Έλεγχος διαστάσεων
        if len(region1_img.size()) != 3 or len(region2_img.size()) != 3:
            raise ValueError("Οι διαστάσεις των εικόνων δεν είναι έγκυρες. Πρέπει να είναι 3D tensors.")

        # Μετατροπή από torch.FloatTensor σε PIL Image και σε RGB
        region1_img_np = region1_img.cpu().numpy().transpose(1, 2, 0)
        region2_img_np = region2_img.cpu().numpy().transpose(1, 2, 0)

        region1_img_pil = Image.fromarray((region1_img_np * 255).astype(np.uint8))
        region2_img_pil = Image.fromarray((region2_img_np * 255).astype(np.uint8))

        region1_img_pil = region1_img_pil.convert("RGB")
        region2_img_pil = region2_img_pil.convert("RGB")
        
        region1_img_pil = transforms.functional.resize(region1_img_pil, (224, 224))
        region2_img_pil = transforms.functional.resize(region2_img_pil, (224, 224))

        transform = transforms.ToTensor()
        region1_img_pil = transform(region1_img_pil).unsqueeze(0)
        region2_img_pil = transform(region2_img_pil).unsqueeze(0)   

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DISTS().to(device)
        region1_img_pil = region1_img_pil.to(device)
        region2_img_pil = region2_img_pil.to(device)

        # Υπολογισμός DISTS
        score = model(region1_img_pil, region2_img_pil)#
        print("DISTS Score: ",score.item())    

        # Αποθήκευση των εικόνων
        save_folder = r"C:\Users\steli\DIPLOMA\myProgramms\GUI\testImages"
        os.makedirs(save_folder, exist_ok=True)
        save_image(region1_img_pil, os.path.join(save_folder, "img0DISTS.png"))
        save_image(region2_img_pil, os.path.join(save_folder, "img1DISTS.png"))
        
        print("Οι εικόνες αποθηκεύτηκαν επιτυχώς στο φάκελο:", save_folder)
        messagebox.showinfo("DISTS Score", f"DISTS Score for the two regions of the image: {score}")          

        return score.item()

    def calculate_LPIPS(self, region1_img, region2_img):
        # Αλλαγή του μεγέθους των εικόνων σε (224, 224)
        region1_img = TF.resize(region1_img, (224, 224))
        region2_img = TF.resize(region2_img, (224, 224))

        # Αποθήκευση των εικόνων 
        save_folder = r"C:\Users\steli\DIPLOMA\myProgramms\GUI\testImages"
        os.makedirs(save_folder, exist_ok=True)
        save_image(region1_img, os.path.join(save_folder, "img0LPIPS.png"))
        save_image(region2_img, os.path.join(save_folder, "img1LPIPS.png"))

        img0_path = os.path.join(save_folder, "img0LPIPS.png")
        img1_path = os.path.join(save_folder, "img1LPIPS.png")
        
        # Δημιουργία αντικειμένου LPIPS για το δίκτυο VGG
        loss_fn_vgg = lpips.LPIPS(net='vgg')
        
        # Φορτώνουμε τις εικόνες με χρήση του lpips
        region1_img = lpips.im2tensor(lpips.load_image(img0_path))
        region2_img = lpips.im2tensor(lpips.load_image(img1_path))
        
        # Υπολογισμός της μετρικής LPIPS για τις δύο περιοχές χρησιμοποιώντας το δίκτυο VGG
        d_vgg = loss_fn_vgg(region1_img, region2_img)
        print("LPIPS Score(VGG):", d_vgg.item())

        print("Οι εικόνες αποθηκεύτηκαν επιτυχώς στο φάκελο:", save_folder)
        messagebox.showinfo("LPIPS Score(VGG)", f"LPIPS Score for the two regions of the image: {d_vgg.item()}")   

    def maximize_image(self, label, image):
        # Αποκτήστε το πλάτος και το ύψος της αρχικής εικόνας
        initial_width = self.image.width
        initial_height = self.image.height

        # Αποκτήστε το πλάτος και το ύψος της εικόνας της περιοχής
        width = label.winfo_width()
        height = label.winfo_height()

        # Υπολογισμός του λόγου width_εικόνας/width_οθόνης και height_εικόνας/height_οθόνης
        ratio_width = initial_width / width
        ratio_height = initial_height / height

        # Επιλογή του μεγαλύτερου λόγου
        min_ratio = min(ratio_width, ratio_height)

        # Διαιρούμε τις διαστάσεις της εικόνας με τον μέγιστο λόγο
        resized_width = int(width * min_ratio)
        resized_height = int(height * min_ratio)

        # Μεγέθυνση της εικόνας
        zoomed_img = image.resize((resized_width, resized_height), resample=Image.BICUBIC)

        # Δημιουργία του αντικειμένου PhotoImage
        photo_image = ImageTk.PhotoImage(zoomed_img)

        # Ενημέρωση της ετικέτας με την ενημερωμένη εικόνα
        label.config(image=photo_image)
        label.image = photo_image

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        if hasattr(self, 'image_display'):
            self.image_display.config(image=photo)
            self.image_display.image = photo
        else:
            self.image_display = tk.Label(self.master, image=photo)
            self.image_display.pack()

    

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSelectorApp(root)
    root.mainloop()
