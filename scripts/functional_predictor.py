# --------------------------------------------------------------------------------------------------
# functional_predictor.py
# --------------------------------------------------------------------------------------------------

import sklearn.inspection
import sklearn.utils._typedefs
import sklearn.utils._heap
import sklearn.neighbors._partition_nodes
import sklearn.neighbors._ball_tree
import sklearn.neighbors._kd_tree
import sklearn.tree
import sklearn.ensemble
import sklearn.neighbors
import sklearn.base

import subprocess
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import pickle
import os
import sys
if getattr(sys, 'frozen', False):

    application_path = os.path.dirname(sys.executable)
    sys.path.append(application_path)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import requests
from bs4 import BeautifulSoup
import tldextract

import re
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import json

if getattr(sys, 'frozen', False):
    # CAS 1: Si estem executant el .EXE
    # La ruta base és la carpeta on està el fitxer .exe
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # CAS 2: Si estem executant l'script Python normal
    # Mirem a quina carpeta està aquest fitxer (scripts/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Si detectem que estem dins de la carpeta 'scripts', pugem un nivell
    if os.path.basename(current_dir) == 'scripts':
        BASE_DIR = os.path.dirname(current_dir)  # Pugem a l'arrel del projecte
    else:
        BASE_DIR = current_dir  # Ja som a l'arrel

# Construïm les rutes absolutes (a prova de bombes)
AGENT_PATH = os.path.join(BASE_DIR, "data", "objects", "q_agent_knn.pkl")
DISC_PATH = os.path.join(BASE_DIR, "data", "objects", "discretizer.pkl")

# Per depurar: Imprimeix on està buscant els fitxers
print(f"DEBUG: Buscant models a: {AGENT_PATH}")

COLUMNES_MODEL = [
    'URLCharProb',
    'NoOfEmptyRef',
    'HasPasswordField',
    'NoOfiFrame',
    'NoOfDegitsInURL',
    'TLDLegitimateProb',
    'HasCopyrightInfo',
    'CharContinuationRate',
    'NoOfJS',
    'HasSocialNet',
    'LargestLineLength',
    'HasDescription',
    'NoOfImage',
    'DomainTitleMatchScore',
    'NoOfExternalRef',
    'NoOfSelfRef',
    'URLLength',
    'NoOfSubDomain',
    'IsHTTPS',
    'LineOfCode'
]

CHAR_FREQS = {
    'e': 0.1116, 'a': 0.0849, 'r': 0.0758, 'i': 0.0754, 'o': 0.0716, 't': 0.0695, 'n': 0.0665,
    's': 0.0573, 'l': 0.0548, 'c': 0.0454, 'u': 0.0363, 'd': 0.0338, 'p': 0.0316, 'm': 0.0301,
    'h': 0.0300, 'g': 0.0247, 'b': 0.0207, 'f': 0.0181, 'y': 0.0177, 'w': 0.0129, 'k': 0.0110,
    'v': 0.0100, 'x': 0.0029, 'z': 0.0027, 'j': 0.0019, 'q': 0.0019,
    '.': 0.05, '-': 0.02, '_': 0.01, '/': 0.01, ':': 0.01
}




def calc_url_char_prob(url):
    clean_url = url.lower().replace("https://", "").replace("http://", "")
    if len(clean_url) == 0: return 0
    total_prob = 0
    for char in clean_url:
        total_prob += CHAR_FREQS.get(char, 0.001)
    return (total_prob / len(clean_url)) * 1.2


def calc_char_continuation_rate(url):
    clean_url = url.replace("https://", "").replace("http://", "")
    if len(clean_url) == 0: return 0
    alnum_count = sum(c.isalnum() for c in clean_url)
    return alnum_count / len(clean_url)


def calc_no_of_empty_ref(soup, domain):
    count = 0
    tags = soup.find_all('a', href=True)
    for tag in tags:
        href = tag['href'].strip().lower()
        if href in ["", "#", "javascript:void(0)", "javascript:;", "javascript:"]:
            count += 1
        elif href == domain or href == "/" or href == domain + "/":
            count += 1
    return count


def extreure_features(url):
    features = {}
    url = url.strip()

    # 1. URL
    features["URL"] = url
    features["URLLength"] = len(url)
    features["NoOfDegitsInURL"] = sum(c.isdigit() for c in url)
    features["IsHTTPS"] = 1 if url.lower().startswith("https://") else 0

    extracted = tldextract.extract(url)
    domain_part = extracted.domain
    suffix = extracted.suffix
    subdomain = extracted.subdomain
    full_domain = f"{subdomain}.{domain_part}.{suffix}".strip(".")
    if full_domain.startswith("."): full_domain = full_domain[1:]

    features["Domain"] = full_domain
    features["TLD"] = suffix
    features["NoOfSubDomain"] = subdomain.count('.') + 1 if subdomain else 0
    features["URLCharProb"] = calc_url_char_prob(url)
    features["CharContinuationRate"] = calc_char_continuation_rate(url)

    common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'uk', 'de', 'jp', 'fr', 'au', 'es', 'cat']
    features["TLDLegitimateProb"] = 0.5 if suffix in common_tlds else 0.01

    #DESCÀRREGA
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, timeout=10, headers=headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        status_ok = True
        pretty_html = soup.prettify()
        lines = pretty_html.splitlines()
    except:
        status_ok = False
        html_content = ""
        soup = BeautifulSoup("", "html.parser")
        lines = []

    # 3. HTML
    if status_ok:
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        features["Title"] = title

        def clean_str(s):
            return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

        if clean_str(domain_part) in clean_str(title):
            features["DomainTitleMatchScore"] = 100.0
        else:
            features["DomainTitleMatchScore"] = 0.0

        features["NoOfImage"] = len(soup.find_all('img')) + len(soup.find_all('svg'))
        features["NoOfJS"] = len(soup.find_all('script'))
        features["NoOfiFrame"] = len(soup.find_all('iframe'))

        has_desc = 0
        if soup.find('meta', attrs={'name': re.compile(r'description', re.I)}):
            has_desc = 1
        elif soup.find('meta', attrs={'property': re.compile(r'description', re.I)}):
            has_desc = 1
        features["HasDescription"] = has_desc

        socials = ["facebook", "twitter", "instagram", "linkedin", "youtube", "tiktok", "github"]
        all_hrefs = [a.get('href', '') for a in soup.find_all('a', href=True)]
        features["HasSocialNet"] = 1 if any(any(s in h for s in socials) for h in all_hrefs) else 0

        features["HasSubmitButton"] = 1 if soup.find(['input', 'button'], {'type': 'submit'}) else 0
        features["HasPasswordField"] = 1 if soup.find('input', {'type': 'password'}) else 0
        features["HasHiddenFields"] = 1 if soup.find('input', {'type': 'hidden'}) else 0

        features["LineOfCode"] = len(lines)

        #CLIPPING OPTIMITZAT
        max_len = max(len(line) for line in lines) if lines else 0
        features["LargestLineLength"] = min(max_len, 7400)

        features["HasCopyrightInfo"] = 1 if "copyright" in html_content.lower() or "©" in html_content else 0
        features["NoOfEmptyRef"] = calc_no_of_empty_ref(soup, full_domain)


        self_refs = 0
        ext_refs = 0

        for link in all_hrefs:
            if domain_part in link or link.startswith('/'):
                self_refs += 1
            elif link.startswith('http'):
                ext_refs += 1

        external_resources = []
        external_resources.extend([link.get('href', '') for link in soup.find_all('link', href=True)])
        external_resources.extend([script.get('src', '') for script in soup.find_all('script', src=True)])
        external_resources.extend([img.get('src', '') for img in soup.find_all('img', src=True)])
        external_resources.extend([iframe.get('src', '') for iframe in soup.find_all('iframe', src=True)])

        for res in external_resources:
            if res.startswith('http') and domain_part not in res:
                ext_refs += 1

        features["NoOfSelfRef"] = self_refs
        features["NoOfExternalRef"] = ext_refs

    else:
        #Valors segurs per defecte
        for k in COLUMNES_MODEL:
            features[k] = 0
        features["DomainTitleMatchScore"] = 0.0
        features["Title"] = "Error"

    return features


def load_models():
    if not os.path.exists(AGENT_PATH) or not os.path.exists(DISC_PATH):
        raise FileNotFoundError("Models no trobats.")

    print("Loading models...")
    with open(DISC_PATH, 'rb') as f: discretizer = pickle.load(f)
    with open(AGENT_PATH, 'rb') as f: agent = pickle.load(f)
    return discretizer, agent


def predict_single_url(features_dict, discretizer, agent):
    """
    Fa la predicció assegurant l'ordre correcte de les columnes.
    """
    #Convertim el diccionari a DataFrame
    df_new = pd.DataFrame([features_dict])


    #FORCEM L'ORDRE CORRECTE

    #Això agafa només les columnes del model en l'ordre correcte.
    #Si en falta alguna, posa 0 automàticament (fill_value=0).
    df_ordered = df_new.reindex(columns=COLUMNES_MODEL, fill_value=0)

    #Depuració: Comprova que no hi ha NaNs
    #print("Dades ordenades:", df_ordered.iloc[0].values)

    #Discretització
    df_discrete = discretizer.transform(df_ordered)

    #Convertim a l'estat que espera l'agent
    state = df_discrete.values[0]

    #Predicció
    q_hits = agent.q_hits
    action = agent.choose_action(state, is_test=True)

    method = "Q-Table (Memòria)" if agent.q_hits > q_hits else "KNN (Similitud)"
    return action, method


def iniciar_interfaz(discretizer, agent):
    # Función que se ejecuta al pulsar el botón
    def procesar():
        url = entrada_url.get().strip()
        if not url:
            messagebox.showwarning("Aviso", "Por favor, introduce una URL.")
            return

        # Feedback visual de carga
        btn_analizar.config(text="Analizando...", state="disabled")
        lbl_resultado.config(text="Procesando...", fg="blue")
        ventana.update()  # Forzar actualización visual

        try:
            # Extracció
            features = extreure_features(url)

            # Predicció
            pred, method = predict_single_url(features, discretizer, agent)

            # Mostrar Resultats
            if pred == 1:
                lbl_resultado.config(text="URL LEGÍTIMA", fg="green")
            else:
                lbl_resultado.config(text="PHISHING DETECTADO", fg="red")



        except Exception as e:
            messagebox.showerror("Error", f"Ha ocurrido un error al analizar:\n{e}")
            lbl_resultado.config(text="Error", fg="black")

        finally:
            # Restaurar botó
            btn_analizar.config(text="Analizar URL", state="normal")

    # --- Configuració ---
    ventana = tk.Tk()
    ventana.title("Detector de Phishing")
    ventana.geometry("500x350")

    # Títul
    tk.Label(ventana, text="Introduzca la URL a analizar", font=("Arial", 14, "bold")).pack(pady=15)

    # Input
    entrada_url = tk.Entry(ventana, width=50, font=("Arial", 11))
    entrada_url.pack(pady=5)
    entrada_url.bind('<Return>', lambda event: procesar())  # Enter para enviar

    # Botó
    btn_analizar = tk.Button(ventana, text="Analizar URL", command=procesar, bg="#007BFF", fg="white",
                             font=("Arial", 11, "bold"))
    btn_analizar.pack(pady=15)

    # Separador
    tk.Frame(ventana, height=2, bd=1, relief="sunken").pack(fill="x", padx=20, pady=10)

    # Resultat
    lbl_resultado = tk.Label(ventana, text="Esperando URL...", font=("Arial", 16, "bold"))
    lbl_resultado.pack(pady=10)



    # Iniciar bucle
    ventana.mainloop()


def reentrenar_amb_gui(ruta_script="scripts/rl_training.py"):
    """
    Obre una finestra mentre s'executa l'entrenament en segon pla.
    SOLUCIÓ ERROR 'ModuleNotFoundError: No module named src':
    S'afegeix l'arrel del projecte al PYTHONPATH.
    """
    # Configurar finestra
    loading_window = tk.Tk()
    loading_window.title("Auto-Reparació del Sistema")
    loading_window.geometry("400x160")
    loading_window.resizable(False, False)

    # Centrar
    screen_width = loading_window.winfo_screenwidth()
    screen_height = loading_window.winfo_screenheight()
    x = (screen_width // 2) - (400 // 2)
    y = (screen_height // 2) - (160 // 2)
    loading_window.geometry(f"400x160+{x}+{y}")

    # Widgets
    tk.Label(loading_window, text="⚠️ Models no trobats", font=("Arial", 12, "bold"), fg="#d9534f").pack(pady=(15, 5))
    tk.Label(loading_window, text=f"Executant {ruta_script}...\nAixò pot trigar uns minuts. No tanquis l'aplicació.",
             justify="center").pack(pady=5)

    progress = ttk.Progressbar(loading_window, orient="horizontal", length=320, mode="indeterminate")
    progress.pack(pady=10)
    progress.start(10)

    # Lògica del fil (Thread)
    def run_training_thread():
        try:
            # 1. Verifiquem que l'script existeix
            if not os.path.exists(ruta_script):
                print(f"❌ Error: No trobo el fitxer a {ruta_script}")
                loading_window.after(0, loading_window.destroy)
                return

            # 2. Amagar consola a Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # ---------------------------------------------------------
            # 3. SOLUCIÓ CLAU: AFEGIR L'ARREL AL PYTHONPATH
            # ---------------------------------------------------------
            my_env = os.environ.copy()

            # Obtenim la carpeta on està el teu main.py (l'arrel del projecte)
            project_root = os.getcwd()

            # Afegim aquesta ruta al PYTHONPATH perquè Python trobi 'src'
            if "PYTHONPATH" in my_env:
                my_env["PYTHONPATH"] = project_root + os.pathsep + my_env["PYTHONPATH"]
            else:
                my_env["PYTHONPATH"] = project_root

            # ---------------------------------------------------------

            print(f"🔄 Llançant re-entrenament des de: {project_root}")

            if getattr(sys, 'frozen', False):
                # Si som un EXE, necessitem cridar el Python del sistema per entrenar
                executable_cmd = "python"
            else:
                # Si estem a PyCharm/Terminal, fem servir el mateix entorn actual
                executable_cmd = sys.executable

            # Passem 'env=my_env' perquè l'altre script sàpiga on és 'src'
            subprocess.run([executable_cmd, ruta_script], check=True, startupinfo=startupinfo, env=my_env)

            print("✅ Entrenament finalitzat.")
            loading_window.after(0, loading_window.destroy)

        except subprocess.CalledProcessError as e:
            print(f"❌ Error durant l'entrenament (Codi sortida: {e.returncode}).")
            loading_window.after(0, loading_window.destroy)
        except Exception as e:
            print(f"❌ Error inesperat: {e}")
            loading_window.after(0, loading_window.destroy)

    # Arrenquem el fil
    t = threading.Thread(target=run_training_thread)
    t.start()

    loading_window.mainloop()


if __name__ == "__main__":
    disc = None
    agent = None

    try:
        # INTENT 1: Carregar models normals
        disc, agent = load_models()
        print("✅ Models carregats correctament.")

        # AQUÍ CRIDES A LA TEVA APP NORMAL
        iniciar_interfaz(disc, agent)

    except (FileNotFoundError, Exception):
        print("⚠️ Models no trobats. Iniciant protocol de recuperació...")

        # CRIDEM A LA GUI INDICANT LA RUTA CORRECTA
        reentrenar_amb_gui("scripts/rl_training.py")

        try:
            # INTENT 2: Carregar després d'entrenar
            disc, agent = load_models()
            print("✅ Models recuperats i carregats.")

            # AQUÍ CRIDES A LA TEVA APP NORMAL
            iniciar_interfaz(disc, agent)

        except Exception as e:
            print(f"❌ Error Fatal: No s'han pogut generar els models. {e}")
            tk.messagebox.showerror("Error Fatal", f"No s'ha pogut iniciar l'aplicació.\nError: {e}")
"""
if __name__ == "__main__":
    try:

        disc, ag = load_models()
        print("Models carregats. Iniciant interfície...")


        iniciar_interfaz(disc, ag)

    except Exception as e:
        print(f"Error fatal: {e}")
        input("Prem Enter per sortir...")
        
        
if __name__ == "__main__":
    try:
        discretizer, agent = load_models()
        print("Models carregats correctament.\n")
    except Exception as e:
        print(f"Error carregant models: {e}")
        sys.exit(1)

    # URL A PROVAR

    url_to_test = "http://allegrolokalnie.85432652315.sbs"
    print(f"Analitzant: {url_to_test} ...")

    features = extreure_features(url_to_test)

    # OPCIONAL: Veure el JSON per depurar
    # print(json.dumps(features, indent=4))

    # PREDICCIÓ
    pred, method = predict_single_url(features, discretizer, agent)

    # Al teu model: 0 = PHISHING, 1 = LEGIT (normalment)
    # OJO: Revisa si al teu '200.csv' el 1 era Legit. Si és així:
    if pred == 1:
        print(f"\nRESULTAT: LEGÍTIMA (0)")  # Poso 0 perquè a la UI sol ser verd
    else:
        print(f"\nRESULTAT: PHISHING (1)")

    print(f"Mètode utilitzat: {method}")

"""