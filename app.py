import streamlit as st
import numpy as np
import cv2
import torch

from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
#from sklearn.linear_model import LinearRegression

from helpers.util import (
    generate_pdf_report,
    load_sam,
    load_medsam,
    set_predictor_params,
    resize_with_aspect_ratio,
    clean_mask,
    enhance_medical_image,
    draw_tangent,
    tag_cobb
)
from helpers.diagnostics import (
    calculate_advanced_diagnostics
)

@st.cache_resource
def get_predictor(param):
    if param == "SAM (General)":
        return load_sam()
    else: # MedSAM
        return load_medsam()

st.set_page_config(page_title="ScolioScan Pro AI 2026", layout="wide")
st.title("🏥 Diagnostic Scolioză: Segmentare Hibrida")
st.sidebar.header("Control Segmentare")

model_choice = st.sidebar.selectbox(
    "Alege modelul de segmentare:",
    ["SAM (General)", "MedSAM (Medical)"],
    index=0  # SAM implicit
)

predictor = get_predictor(model_choice)

predictor_settings = set_predictor_params(model_choice)

# Modul de selecție: Adăugare sau Ștergere
mode = st.sidebar.radio(
    "Alege parametrii de segmentare:",
    ["Defineste Box (ROI)", "Adauga Vertebra (+)", "Sterge Fundal (-)"],
    index=0,
    help="1. Incepe cu Box-ul (2 click-uri). 2. Adauga puncte."
)
label_type = 1 if "Adauga" in mode else 0
results = []

if "box" not in st.session_state:
    st.session_state["box"] = []
if "points" not in st.session_state:
    st.session_state["points"] = []
if "labels" not in st.session_state:    
    st.session_state["labels"] = []
if "widget_id" not in st.session_state:
    st.session_state["widget_id"] = 0

uploaded_file = st.sidebar.file_uploader("Încarcă Radiografia", type=['jpg', 'png'])

if uploaded_file:
    img_uploaded = Image.open(uploaded_file).convert("RGB")

    # Aplicăm resize pentru afișare și procesare ușoară
    #img, scale = resize_with_aspect_ratio(img_uploaded, target_size=800)
    img = img_uploaded

    img_np = np.array(img)
    # Pre-procesare imagine pentru contrast mai bun
    medical_image = enhance_medical_image(img_np)
    h, w = medical_image.shape[:2]


    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Imagine Interactivă")

        # Desenăm Box-ul pe imagine pentru vizualizare în editor
        img_with_box = medical_image.copy()

        # Folosim o culoare distinctă (Galben) și o grosime mai mare pentru Box
        if len(st.session_state["box"]) > 0:
            # Dacă avem doar primul punct, desenăm o mică ancoră
            if len(st.session_state["box"]) == 1:
                p1 = st.session_state["box"][0]
                cv2.drawMarker(img_with_box, tuple(p1), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            # Dacă avem ambele puncte, desenăm dreptunghiul galben
            elif len(st.session_state["box"]) == 2:
                p1, p2 = st.session_state["box"]
                cv2.rectangle(img_with_box, tuple(p1), tuple(p2), (255, 255, 0), 3)
                # Adăugăm o etichetă discretă
                cv2.putText(img_with_box, "ROI Coloana", (p1[0], p1[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        

        # Desenăm Punctele (Verde pentru Pozitiv, Roșu pentru Negativ)
        for i, p in enumerate(st.session_state["points"]):
            label = st.session_state["labels"][i]
            
            if label == 1: # Vertebră
                color = (0, 255, 0) # Verde
                marker_type = cv2.MARKER_TILTED_CROSS
            else: # Fundal
                color = (255, 0, 0) # Roșu
                marker_type = cv2.MARKER_CROSS

            # Desenăm un cerc cu contur pentru vizibilitate maximă pe radiografie
            cv2.circle(img_with_box, tuple(p), 7, (255, 255, 255), -1) # Fundal alb pentru contrast
            cv2.circle(img_with_box, tuple(p), 5, color, -1) # Punctul colorat

        # Afișăm imaginea și captăm click-urile
        coords = streamlit_image_coordinates(Image.fromarray(img_with_box), key=f"coords_widget_{st.session_state['widget_id']}")

        if coords:
            pt = [coords["x"], coords["y"]]
            if mode == "Defineste Box (ROI)":
                if len(st.session_state["box"]) < 2:
                    if pt not in st.session_state["box"]:
                        st.session_state["box"].append(pt)
                        st.rerun()
                        
            elif mode == "Adauga Vertebra (+)":
                if pt not in st.session_state["points"]:
                    st.session_state["points"].append(pt)
                    st.session_state["labels"].append(1) # Label 1 = Pozitiv
                    st.rerun()
                    
            elif mode == "Sterge Fundal (-)":
                if pt not in st.session_state["points"]:
                    st.session_state["points"].append(pt)
                    st.session_state["labels"].append(0) # Label 0 = Negativ
                    st.rerun()
            if pt not in st.session_state["points"]:
                st.session_state["points"].append(pt)
                st.session_state["labels"].append(label_type)
                st.rerun()

        if st.button("Reset"):
            st.session_state["points"] = []
            st.session_state["labels"] = []
            st.session_state["box"] = []
            st.session_state["widget_id"] = np.random.randint(0, 1000000)
            st.rerun()

    with col2:
        st.subheader("Masca Predictor")

        if len(st.session_state["box"]) == 2:
            p1 = st.session_state["box"][0]
            p2 = st.session_state["box"][1]
            xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
            ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
            
            # Creăm box-ul doar dacă avem ambele puncte
            input_box = np.array([
                xmin, ymin, 
                xmax, ymax
            ])
            
            # Adăugăm batch dimension [1, 4] pentru SAM/MedSAM
            input_box_final = input_box.reshape(1, 4) 
        else:
            input_box_final = None

        if len(st.session_state["points"]) > 0:
            
            predictor.set_image(medical_image)

            if len(st.session_state["box"]) == 2:
                filtered_points = []
                filtered_labels = []

                for pt, lbl in zip(st.session_state["points"], st.session_state["labels"]):
                    # Păstrăm punctul DOAR dacă este în interiorul Box-ului
                    if xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax:
                        filtered_points.append(pt)
                        filtered_labels.append(lbl)
                    else:
                        # Opțional: Poți afișa un mesaj de avertizare în Streamlit
                        st.sidebar.warning(f"Punctul de la {pt} a fost ignorat (este în afara Box-ului).")

                # Folosim listele filtrate pentru AI
                input_pts = np.array(filtered_points) if filtered_points else None
                input_lbl = np.array(filtered_labels) if filtered_labels else None
            else:
                # Folosim listele filtrate pentru AI
                input_pts = np.array(st.session_state["points"]) if st.session_state["points"] else None
                input_lbl = np.array(st.session_state["labels"]) if st.session_state["labels"] else None

            # Predicție folosind punctele pozitive și negative din box
            if input_box_final is not None:
                masks, scores, _ = predictor.predict(
                    point_coords=input_pts,
                    point_labels=input_lbl,
                    box=input_box_final,
                    multimask_output=predictor_settings[0]
                )

                # Alegem masca - de obicei cea mai bună pentru organe/oase
                mask_idx = predictor_settings[1] 
                best_mask = masks[mask_idx]
                mask_uint8 = (best_mask.astype(np.uint8) * 255)

                mask_clean = clean_mask(mask_uint8)
                
                st.image(mask_uint8, use_container_width=True)
            
                # Afișăm scorul de încredere al AI-ului
                st.caption(f"Scor încredere segmentare: {scores[mask_idx]:.2f}")

                # calculam metrici pe masca rezultată
                results = calculate_advanced_diagnostics(mask_uint8)

            
        else:
            st.info("Puneți puncte pe coloană pentru a începe.")

    with col3:
            st.subheader("Rezultat Analiză")
    
            if results:
                # Vizualizarea axei coloanei
                display_image = np.array(img).copy()
                mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
                
                # Facem masca verde semitransparentă (doar unde este alb)
                mask_rgb[np.where((mask_rgb == [255, 255, 255]).all(axis=2))] = [0, 255, 0]
                display_image = cv2.addWeighted(display_image, 0.8, mask_rgb, 0.2, 0)
                h, w = display_image.shape[:2]
                
                # Desenăm axul central netezit al coloanei (linia albastră subțire)
                cx, cy = results["curve_points"]
                pts = np.array([np.transpose(np.vstack([cx, cy]))], np.int32)
                cv2.polylines(display_image, pts, False, (0, 100, 255), 2)

                # Desenăm punctele de INFLEXIUNE (Galben)
                for (ix, iy) in results["inflection_points"]:
                    # Un cerc cu marginea neagră pentru vizibilitate
                    cv2.circle(display_image, (ix, iy), 8, (0, 255, 255), -1) 
                    cv2.circle(display_image, (ix, iy), 9, (0, 0, 0), 1)
                    cv2.putText(display_image, "Inflexiune", (ix + 15, iy - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                k = 0
                for (ax, ay, k) in results["apical_points"]:
                    cv2.drawMarker(display_image, (int(ax), int(ay)), (255, 0, 255), cv2.MARKER_DIAMOND, 15, 3)
                    cv2.putText(display_image, "APEX", (int(ax) + 15, int(ay) - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                

                # Desenăm perechile de linii pentru fiecare curbă
                sample_y = results["sample_y"]
                p_func = results["poly_func"] 
                derivative_func = np.polyder(p_func)
                indexare = 0
                for curve in results["curves"]:
                    # Extragem coordonatele pentru idx_max și idx_min din acest segment
                    y_max_val = sample_y[curve['idx_max']]
                    y_min_val = sample_y[curve['idx_min']]
                    
                    # desenare:
                    draw_tangent(display_image, p_func(y_max_val), y_max_val, derivative_func(y_max_val), w, (255, 0, 0))
                    draw_tangent(display_image, p_func(y_min_val), y_min_val, derivative_func(y_min_val), w, (0, 255, 0))
                    tag_cobb(display_image, p_func(y_min_val), y_min_val, y_max_val, indexare)
                    indexare += 1
                
    
                st.image(display_image, use_container_width=True)
                st.caption("Liniile de înclinație Cobb, punctele de inflexiune si de APEX")  

                # Afișăm metrici pentru toate curburile detectate
                n_curves = len(results.get("curves", []))
                if n_curves > 0:
                    cols = st.columns(len(results["curves"]))
                    max_angles = 0
                    for idx, curve in enumerate(results["curves"]):
                        label = "Curbură Principală" if idx == 0 else f"Curbură Secundară {idx}"
                        cols[idx].metric(label, f"{curve['value']}°")
                        if max_angles < curve['value']:
                            max_angles = curve['value']

                    c1, c2 = st.columns(2)
                    c1.metric("Tipologie", results['type'])
                    c2.metric(label="Intensitate Curbură (Kappa)", value=f"{k:.5f}")

                    # Clasificare medicală automată (2026 Guidelines)
                    if max_angles < 10:
                        st.success("Aliniament Normal")
                    elif 10 <= max_angles < 20:
                        st.warning("Scolioză Ușoară (Monitorizare recomandată)")
                    else:
                        st.error("Scolioză Semnificativă (Necesită intervenție)")
                else:
                    st.info("Nu au fost detectate curburi pentru analiză.")                    

    with col1:
        if results:
            patient_name = st.text_input("Nume Pacient", "Pacient")
                
            if st.button("Generează Raport PDF"):
                # Salvăm masca temporar pentru a o pune în PDF
                temp_mask_path = "temp_mask.png"
                cv2.imwrite(temp_mask_path, cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
                
                pdf_bytes = generate_pdf_report(
                    patient_name, 
                    results["curves"] if len(results["curves"]) > 0 else [], 
                    results['type'] if len(results["curves"]) > 0 else '', 
                    temp_mask_path
                )
                
                st.download_button(
                    label="Descarc Raportul PDF",
                    data=pdf_bytes,
                    file_name=f"Raport_{patient_name}.pdf",
                    mime="application/pdf"
                )
