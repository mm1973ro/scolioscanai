import pandas as pd
import numpy as np
import cv2


def classify_scoliosis_type_1(angles):
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

def classify_scoliosis_type_2(poly_func, y_range):
    """
    Clasificare avansată 2026 folosind a doua derivată (curbura).
    """
    # Calculăm a doua derivată a polinomului (f'')
    second_derivative = np.polyder(poly_func, 2)
    
    # Eșantionăm a doua derivată de-a lungul coloanei
    y_samples = np.linspace(y_range[0], y_range[1], 100)
    curvature_values = second_derivative(y_samples)
    
    # Numărăm de câte ori se schimbă semnul curburii (inflexiuni)
    # Ignorăm variațiile infime (zgomot matematic)
    threshold = 0.00001 # Ajustabil în funcție de rezoluție
    sign_changes = 0
    for i in range(1, len(curvature_values)):
        if curvature_values[i-1] * curvature_values[i] < 0:
            if abs(curvature_values[i-1] - curvature_values[i]) > threshold:
                sign_changes += 1
                
    # În scolioza S, coloana își schimbă concavitatea cel puțin o dată (o trecere prin zero a f'')
    if sign_changes >= 1:
        return "Tip S", "🚨"
    else:
        return "Tip C", "⚠️"

def find_inflection_points(poly_func, y_min, y_max):
    # Calculăm a doua derivată (f'')
    second_der = np.polyder(poly_func, 2)
    
    # Găsim rădăcinile celei de-a doua derivate (unde f'' = 0)
    roots = second_der.roots
    
    inflection_pts = []
    for root in roots:
        # Păstrăm doar rădăcinile reale care se află în interiorul coloanei noastre
        if np.isreal(root) and y_min < root.real < y_max:
            y_val = root.real
            x_val = poly_func(y_val)
            inflection_pts.append((int(x_val), int(y_val)))
            
    return inflection_pts

def find_apical_points(poly_func, y_min, y_max):
    # Derivata întâi și a doua
    f_p = np.polyder(poly_func, 1)
    f_pp = np.polyder(poly_func, 2)
    
    y_samples = np.linspace(y_min, y_max, 500)
    curvatures = []
    
    for y in y_samples:
        # Formula curburii
        k = abs(f_pp(y)) / (1 + f_p(y)**2)**(1.5)
        curvatures.append(k)
    
    curvatures = np.array(curvatures)
    
    # Găsim indexul pentru vârful curbei (maximul local al curburii)
    # Dacă e Tip S, pot fi două vârfuri (unul pt fiecare curbă)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(curvatures, distance=100, prominence=0.00001) # Prag minim pentru a evita zgomotul
    
    apical_pts = []
    for p in peaks:
        y_val = y_samples[p]
        #x_val = poly_func(y_val)
        #apical_pts.append((int(x_val), int(y_val), curvatures[p]))
        if y_min + 10 < y_val < y_max - 10:
            apical_pts.append((int(poly_func(y_val)), int(y_val), curvatures[p]))
        
    return apical_pts


def calculate_advanced_diagnostics(mask):
    
    # Curățare morfologică
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) < 200: return None
    
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    height = y_max - y_min

    # Analiză doar în zona centrală stabilă (90% din coloană)
    centers_x, centers_y = [], []
    for y in range(int(y_min + height*0.05), int(y_max - height*0.05), 5):
        row_pixels = x_coords[y_coords == y]
        if len(row_pixels) > 5:
            # Folosim mediana în loc de medie pentru a nu fi atrași de pixeli izolați la dreapta
            mid_x = np.median(row_pixels)
            # Verificăm dacă punctul este valid (nu la marginea extremă a imaginii)
            centers_x.append(mid_x)
            centers_y.append(y)

    # Dacă avem prea puține puncte, ieșim
    if len(centers_x) < 20: return None

    # NETEZIRE GLOBALĂ
    # Fit la un polinom de gradul 4
    poly_coeffs = np.polyfit(centers_y, centers_x, 4)
    poly_func = np.poly1d(poly_coeffs)
    
    # Calculăm derivata (panta) polinomului în fiecare punct
    # Derivata ne dă înclinația exactă matematică, fără zgomot de pixeli
    derivative_func = np.polyder(poly_func)
    
    # Generăm punctele de eșantionare strict între limitele detectate
    sample_y = np.linspace(int(y_min + height*0.05), int(y_max - height*0.05), 100)
    
    # Calculăm valorile X pe baza polinomului
    fitted_x = poly_func(sample_y)

    # Calculăm unghiurile în grade de-a lungul coloanei netezite
    slopes = derivative_func(sample_y)
    #print(slopes)
    raw_angles = np.degrees(np.arctan(slopes))
    angles = -raw_angles
    #print(angles)

    # Unghiul Cobb este diferența dintre înclinația maximă la stânga și cea la dreapta
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
        #type_label, icon = classify_scoliosis_type_1(angles)
        type_label, icon = classify_scoliosis_type_2(poly_func, (y_min, y_max))

    # Date pentru vizualizare (punctele unde coloana este cea mai înclinată)
    idx_max = np.argmax(angles)
    idx_min = np.argmin(angles)

    # Calculăm coordonatele și pantele pentru vizualizare
    pt_max = (poly_func(sample_y[idx_max]), sample_y[idx_max], slopes[idx_max])
    pt_min = (poly_func(sample_y[idx_min]), sample_y[idx_min], slopes[idx_min])

    inflections = find_inflection_points(poly_func, y_min, y_max)
    apical_points = find_apical_points(poly_func, y_min, y_max)

    # Găsim indicii unde a doua derivată își schimbă semnul
    second_der = np.polyder(poly_func, 2)
    curvature_values = second_der(sample_y)
    inflection_indices = np.where(np.diff(np.sign(curvature_values)) != 0)[0]

    # Calculăm unghiurile Cobb pe segmente
    all_curves = calculate_multi_cobb(angles, inflection_indices)

    # Sortăm curburile după magnitudine (cea mai mare este Curbura Principală)
    all_curves = sorted(all_curves, key=lambda x: x['value'], reverse=True)

    return {
        "angle": cobb_result,
        "curves": all_curves,
        "poly_func": poly_func,
        #"type": type_label,
        "type": "Tip S" if len(all_curves) > 1 else "Tip C",
        "icon": icon,
        "inflection_points": inflections,
        "apical_points": apical_points,
        "pt_max": pt_max,
        "pt_min": pt_min,
        "sample_y": sample_y,
        "curve_points": (poly_func(sample_y), sample_y) # Pentru desenarea axului coloanei

    }

def calculate_multi_cobb(angles, inflection_indices):
    """
    Calculăm unghiurile Cobb pentru fiecare segment delimitat de inflexiuni.
    inflection_indices: indicii din lista 'angles' unde f''(y) trece prin zero.
    """
    # Adăugăm începutul și sfârșitul coloanei ca limite
    limits = [0] + sorted(inflection_indices) + [len(angles) - 1]
    cobb_results = []

    for i in range(len(limits) - 1):
        start, end = limits[i], limits[i+1]
        if end - start < 10: continue # Ignorăm segmentele prea mici

        segment_angles = angles[start:end]
        # Unghiul Cobb pentru acest segment este diferența dintre extreme
        cobb_val = abs(np.max(segment_angles) - np.min(segment_angles))
        
        if cobb_val >= 7.0 :
            # Salvăm valoarea și indicii punctelor limită pentru desenare
            cobb_results.append({
                "value": round(cobb_val, 2),
                "idx_max": start + np.argmax(segment_angles),
                "idx_min": start + np.argmin(segment_angles)
            })
    
    return cobb_results