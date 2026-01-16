import pandas as pd
import numpy as np
import cv2


def classify_scoliosis_type(angles):
    """
    Analizează lista de unghiuri locale pentru a detecta numărul de curburi.
    angles: listă de grade calculate de-a lungul coloanei
    """
    # Filtrăm unghiurile pentru a elimina zgomotul (moving average)
    smoothed_angles = pd.Series(angles).rolling(window=5, center=True).mean().dropna().values
    
    # Detectăm trecerile prin zero (schimbarea direcției curburii)
    # Ignorăm variațiile mici sub 2 grade pentru a evita alarmele false
    sign_changes = 0
    for i in range(1, len(smoothed_angles)):
        if smoothed_angles[i-1] * smoothed_angles[i] < 0: # Schimbare de semn
            if abs(smoothed_angles[i-1] - smoothed_angles[i]) > 2:
                sign_changes += 1
    
    if sign_changes >= 1:
        return "Tip S", "🚨"
    else:
        return "Tip C", "⚠️"


def calculate_advanced_diagnostics(mask):
    
    # 1. Curățare morfologică
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) < 200: return None
    
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    height = y_max - y_min

    # 2. Analiză doar în zona centrală stabilă (80% din coloană)
    centers_x, centers_y = [], []
    centers_x_mean, centers_y_mean = [], []
    for y in range(int(y_min + height*0.1), int(y_max - height*0.1), 5):
        row_pixels = x_coords[y_coords == y]
        if len(row_pixels) > 5:
            # Folosim mediana în loc de medie pentru a nu fi atrași de pixeli izolați la dreapta
            mid_x = np.median(row_pixels)
            mean_x = np.mean(row_pixels)
            # Verificăm dacă punctul este valid (nu la marginea extremă a imaginii)
            centers_x.append(mid_x)
            centers_y.append(y)
            centers_x_mean.append(mean_x)
            centers_y_mean.append(y)

    # Dacă avem prea puține puncte, ieșim
    if len(centers_x) < 20: return None

    # 3. NETEZIRE GLOBALĂ (Cheia succesului)
    # Fit la un polinom de gradul 3 (suficient pentru forma în S, dar elimină zgomotul)
    poly_coeffs = np.polyfit(centers_y, centers_x, 3)
    poly_func = np.poly1d(poly_coeffs)
    
    # Calculăm derivata (panta) polinomului în fiecare punct
    # Derivata ne dă înclinația exactă matematică, fără zgomot de pixeli
    derivative_func = np.polyder(poly_func)
    
    # Generăm punctele de eșantionare strict între limitele detectate
    sample_y = np.linspace(y_min + height*0.05, y_max - height*0.05, 100)
    
    # Calculăm valorile X pe baza polinomului
    fitted_x = poly_func(sample_y)

    # Calculăm unghiurile în grade de-a lungul coloanei netezite
    slopes = derivative_func(sample_y)
    #print(slopes)
    angles = np.degrees(np.arctan(slopes))
    #print(angles)

    # 4. Unghiul Cobb este diferența dintre înclinația maximă la stânga și cea la dreapta
    # (Specific pentru forma în "S")
    max_angle = np.max(angles)
    min_angle = np.min(angles)
    raw_cobb = abs(max_angle - min_angle)

    # Dacă coloana este aproape dreaptă, micile variații sunt zgomot
    if raw_cobb < 7.0: # Prag de sensibilitate
        cobb_result = 0.0
        type_label = "Normal"
        icon = "✅"
    else:
        cobb_result = round(raw_cobb, 2)
        type_label, icon = classify_scoliosis_type(angles)

    # Date pentru vizualizare (punctele unde coloana este cea mai înclinată)
    idx_max = np.argmax(angles)
    idx_min = np.argmin(angles)

    # Calculăm coordonatele și pantele pentru vizualizare
    pt_max = (poly_func(sample_y[idx_max]), sample_y[idx_max], slopes[idx_max])
    pt_min = (poly_func(sample_y[idx_min]), sample_y[idx_min], slopes[idx_min])

    #debug_pts = (centers_x[idx_max], centers_y[idx_max], centers_x[idx_min], centers_y[idx_min])
    debug_pts = (centers_x_mean[np.argmax(angles)], centers_y_mean[np.argmax(angles)], 
                      centers_x_mean[np.argmin(angles)], centers_y_mean[np.argmin(angles)])

    
    return {
        "angle": cobb_result,
        "type": type_label,
        "icon": icon,
        "debug_pts": debug_pts,
        "pt_max": pt_max,
        "pt_min": pt_min,
        "curve_points": (poly_func(sample_y), sample_y) # Pentru desenarea axului coloanei

    }