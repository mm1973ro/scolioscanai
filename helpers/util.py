import numpy as np
import cv2
import torch
import datetime

from PIL import Image

from segment_anything import sam_model_registry, SamPredictor
from fpdf import FPDF


def load_sam():
    # Asigură-te că fișierul .pth este în folder
    sam = sam_model_registry["vit_b"](checkpoint="pth/sam_vit_b_01ec64.pth")
    sam.to(device="cpu") # Sau "cuda" dacă ai GPU
    return SamPredictor(sam)

def load_medsam():
    device = torch.device("cpu") # Forțăm utilizarea procesorului pe Mac
    medsam_model = sam_model_registry["vit_b"](checkpoint=None) # Creăm structura
    
    # Încărcăm greutățile cu map_location pentru a evita eroarea CUDA
    checkpoint = torch.load("pth/medsam_vit_b.pth", map_location=device)
    
    # Dacă fișierul conține un dicționar de stare (state_dict)
    if "model" in checkpoint:
        medsam_model.load_state_dict(checkpoint["model"])
    else:
        medsam_model.load_state_dict(checkpoint)
        
    medsam_model.to(device)
    return SamPredictor(medsam_model)

def set_predictor_params(param):
    if param == "SAM (General)":
        return [False, 0]
    else: # MedSAM
        return [True, 2]

def resize_with_aspect_ratio(image, target_size=1024):
    # image este un obiect PIL Image
    img = np.array(image)
    h, w = img.shape[:2]
    
    # Calculăm raportul de scalare
    scaling_factor = target_size / max(h, w)
    new_h, new_w = int(h * scaling_factor), int(w * scaling_factor)
    
    # Redimensionare
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Opțional: Putem adăuga padding pentru a face imaginea pătrată (1024x1024)
    # Acest lucru ajută enorm modelele SAM/MedSAM
    result = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    result[:new_h, :new_w, :] = resized_img
    
    return Image.fromarray(result), scaling_factor

def clean_mask(mask):
    # Umple găurile mici din interiorul vertebrelor
    kernel = np.ones((7, 7), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Netezește marginile "zimțate"
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
    _, refined_mask = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)
    
    return refined_mask

def enhance_medical_image(img_np):
    # Convertim în grayscale dacă nu este deja
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Aplicăm CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Creștem clipLimit la 5.0 pentru a forța contrastul vertebrelor
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10,10))
    enhanced = clahe.apply(gray)

    # 2. Unsharp Masking (accentuarea marginilor oaselor)
    gaussian_3 = cv2.GaussianBlur(enhanced, (0, 0), 3)
    unsharp = cv2.addWeighted(enhanced, 1.5, gaussian_3, -0.5, 0)
    
    # Revenim la RGB pentru că SAM cere 3 canale
    return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2RGB)

# Funcție pentru a desena tangenta (linia de diagnostic)
def draw_tangent(img, x, y, slope, w, color):
    length = w // 4 # Lungimea liniei de diagnostic
    # Calculăm punctele de capăt folosind panta
    dx = length / np.sqrt(1 + slope**2)
    dy = -slope * dx
    p1 = (int(x - dx), int(y - dy))
    p2 = (int(x + dx), int(y + dy))
    cv2.line(img, p1, p2, color, 4)
    
    

def tag_cobb(img, x, y_min, y_max, index):
    y = int((y_min + y_max)/2)
    print(y)
    
    if index == 0:
        tag_text = "Curbura principala"
    else:
        tag_text = f"Curbura {index}"

    cv2.putText(img, tag_text, (int(x) + 40, int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def generate_pdf_report(patient_name, curves, scoliosis_type, mask_image_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Header - Design Medical 2026
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 5, "Raport ScolioScan AI 2026", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, f"Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='R')
    pdf.ln(3)

    # Date Pacient
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 5, f"Pacient: {patient_name}", ln=True)
    pdf.ln(1)

    # Rezultate Analiza AI
    n_curves = len(curves)
    if n_curves > 0:
        max_angles = 0
        pdf.set_font("Arial", size=12)
        for idx, curve in enumerate(curves):
            label = "Curbura Principala" if idx == 0 else f"Curbura Secundara {idx}"
            pdf.set_x(20)
            pdf.cell(0, 5, f"{label}: {curve['value']} grade", ln=True)
            if max_angles < curve['value']:
                max_angles = curve['value']
        pdf.set_x(20)
        pdf.cell(0, 5, f"Unghi Cobb Maxim: {max_angles} grade", ln=True)
        pdf.set_x(20)
        pdf.cell(0, 5, f"Tipologie Scolioza: {scoliosis_type}", ln=True)
        pdf.ln(1)
    else:
        pdf.set_font("Arial", size=12)
        pdf.set_x(20)
        pdf.cell(0, 5, "Nu au fost detectate curburi pentru analiza.", ln=True)
        pdf.ln(1)

    # Imaginea cu Segmentarea
    pdf.cell(0, 5, "Analiza Vizuala a Segmentarii Vertebrale:", ln=True)
    pdf.ln(1)
    # Inserăm imaginea măștii (salvată temporar)
    pdf.image(mask_image_path, x=20, y=None, w=80)
    
    pdf.ln(3)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, "Nota: Acest raport este generat automat de un sistem AI la nivel de prototip academic. Necesita validarea unui medic specialist.")

    pdf_output = pdf.output()
    return bytes(pdf_output)