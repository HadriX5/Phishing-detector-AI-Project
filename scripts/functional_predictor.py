# --------------------------------------------------------------------------------------------------
# functional_predictor.py
# --------------------------------------------------------------------------------------------------

import sys
import os
import pickle
import threading
import subprocess
import re
import json
import tkinter as tk
from tkinter import ttk, messagebox

# Imports de tercers
import requests
import tldextract
import pandas as pd
from bs4 import BeautifulSoup
import difflib

# Imports específics per a PyInstaller/Sklearn (evita errors de dll perdudes)
import sklearn.utils._typedefs
import sklearn.utils._heap
import sklearn.neighbors._partition_nodes
import sklearn.neighbors._ball_tree
import sklearn.neighbors._kd_tree

# -- CONFIGURACIÓ DE RUTES I SISTEMA ---------------------------------------------------------------

if getattr(sys, 'frozen', False):
    # Si estem executant el .EXE
    BASE_DIR = os.path.dirname(sys.executable)
    sys.path.append(BASE_DIR)
else:
    # Si estem executant l'script Python normal
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Si estem dins de 'scripts', pugem un nivell
    if os.path.basename(current_dir) == 'scripts':
        BASE_DIR = os.path.dirname(current_dir)
    else:
        BASE_DIR = current_dir
    
    # Afegim el directori pare al path per imports relatius si cal
    sys.path.append(os.path.dirname(current_dir))

AGENT_PATH = os.path.join(BASE_DIR, "data", "objects", "q_agent_knn.pkl")
DISC_PATH = os.path.join(BASE_DIR, "data", "objects", "discretizer.pkl")
TLD_JSON_PATH = os.path.join(BASE_DIR, "data", "others", "tld_data.json")

# -- CONSTANTS DEL MODEL ---------------------------------------------------------------------------

COLUMNES_MODEL = [
    'URLCharProb', 'NoOfEmptyRef', 'HasPasswordField', 'NoOfiFrame', 
    'NoOfDegitsInURL', 'TLDLegitimateProb', 'HasCopyrightInfo', 
    'CharContinuationRate', 'NoOfJS', 'HasSocialNet', 'LargestLineLength', 
    'HasDescription', 'NoOfImage', 'DomainTitleMatchScore', 'NoOfExternalRef', 
    'NoOfSelfRef', 'URLLength', 'NoOfSubDomain', 'IsHTTPS', 'LineOfCode'
]

CHAR_FREQS = {
    'h': 0.04338299627140599, 't': 0.08929736950179214, 'p': 0.050491186590696825, 
    's': 0.056619822272369016, ':': 0.02990917648496307, '/': 0.07074633415917157, 
    'w': 0.07931485480293878, '.': 0.06600616331409985, 'o': 0.05006988070181445, 
    'u': 0.01738525022101958, 'b': 0.015075077299669788, 'a': 0.04402107916581139, 
    'n': 0.02913828262733731, 'k': 0.009552207178019372, 'm': 0.031082487561146538, 
    'i': 0.03578517620104274, 'c': 0.03821355087965866, '-': 0.009086040697115891, 
    'z': 0.004174838844963712, 'd': 0.018451574687942195, 'e': 0.04975644682861439, 
    'v': 0.007765161884020908, 'f': 0.012807256318813602, 'r': 0.03218585153003394, 
    'j': 0.0045778463344690636, 'l': 0.024156838532415247, 'g': 0.01580022515885298, 
    'y': 0.009452303820130275, '6': 0.004214239430719087, '1': 0.005198516235918902, 
    'x': 0.004179708580281792, '0': 0.005208550842028885, '5': 0.004300714124549236, 
    '9': 0.003338424912149237, '4': 0.004575337682941568, 'q': 0.0029923785690917307, 
    '2': 0.0051771189140667325, '3': 0.0045381506132398655, '8': 0.003322635164299705, 
    '?': 0.000654315345465664, '=': 0.0012983009493475219, '@': 0.0001818034518749883, 
    '7': 0.00390995475720754, '_': 0.0008524988161378306, '&': 0.0006428050619865657, 
    ';': 0.00046041133916393135, '#': 8.293306814427225e-05, '~': 8.558928740867955e-06, 
    '%': 0.00047605351927655214, '(': 1.1952986689832833e-05, '[': 2.50865152749578e-06, 
    ']': 2.6562192644072963e-06, '*': 7.968657793221889e-06, '+': 2.3463270168931117e-05, 
    ',': 1.578974784953226e-05, ')': 1.16578512160098e-05, '!': 2.951354738230329e-06, 
    '$': 1.6232451060266812e-06, "'": 1.4756773691151646e-07, 'ã': 2.9513547382303293e-07, 
    'œ': 1.4756773691151646e-07, '¤': 1.4756773691151646e-07
}

# -- CARREGAR DADES AUXILIARS (TLD MAP) ------------------------------------------------------------

DEFAULT_TLD_PROB = 0.261534 # Valor de fallback per si falla el JSON
TLD_PROB_MAP = {}

def carregar_tld_data():
    """
    Carrega el mapa de probabilitats TLD des d'un fitxer JSON extern.
    """
    global TLD_PROB_MAP, DEFAULT_TLD_PROB
    
    if os.path.exists(TLD_JSON_PATH):
        try:
            with open(TLD_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                TLD_PROB_MAP = data.get("tld_map", {})
                DEFAULT_TLD_PROB = data.get("default_prob", 0.261534)
            print(f"TLD Data carregat ({len(TLD_PROB_MAP)} entrades).")

        except Exception as e:
            print(f"Error carregant {TLD_JSON_PATH}: {e}")
            print("   S'utilitzaran valors per defecte.")

    else:
        print(f"Fitxer no trobat: {TLD_JSON_PATH}")
        print("   Assegura't d'haver executat 'generate_json.py'.")

carregar_tld_data()

# -- FUNCIONS AUXILIARS ----------------------------------------------------------------------------

def calc_tld_legitimate_prob(url):
    """Consulta el mapa carregat des del JSON."""
    try:
        ext = tldextract.extract(url)
        suffix = ext.suffix.lower()
        # Retorna el valor del mapa o el per defecte
        return TLD_PROB_MAP.get(suffix, DEFAULT_TLD_PROB)
    except Exception:
        return DEFAULT_TLD_PROB

def calc_domain_title_match_score(domain, title):
    """Calcula similitud difusa (0-100) amb difflib."""
    if not title or not domain:
        return 0.0
        
    clean_dom = re.sub(r'[^a-z0-9]', '', domain.lower())
    clean_tit = re.sub(r'[^a-z0-9]', '', title.lower())
    
    if not clean_dom or not clean_tit:
        return 0.0

    matcher = difflib.SequenceMatcher(None, clean_dom, clean_tit)
    return matcher.ratio() * 100.0

def calc_url_char_prob(url):
    clean_url = url.lower().replace("https://", "").replace("http://", "")
    if not clean_url: 
        return 0
    total_prob = sum(CHAR_FREQS.get(char, 0.001) for char in clean_url)
    return (total_prob / len(clean_url))

def calc_char_continuation_rate(url):
    clean_url = url.replace("https://", "").replace("http://", "")
    if not clean_url: 
        return 0
    alnum_count = sum(c.isalnum() for c in clean_url)
    return alnum_count / len(clean_url)

def calc_no_of_empty_ref(soup, domain):
    count = 0
    tags = soup.find_all('a', href=True)
    for tag in tags:
        href = tag['href'].strip().lower()
        if href in ["", "#", "javascript:void(0)", "javascript:;", "javascript:"]:
            count += 1
        elif href == domain or href == "/" or href == f"{domain}/":
            count += 1
    return count

def extreure_features(url):
    """Extreu les característiques de la URL per al model."""
    features = {}
    url = url.strip()

    # 1. Característiques basades en text de la URL
    features["URL"] = url
    features["URLLength"] = len(url)
    features["NoOfDegitsInURL"] = sum(c.isdigit() for c in url)
    features["IsHTTPS"] = 1 if url.lower().startswith("https://") else 0

    extracted = tldextract.extract(url)
    domain_part = extracted.domain
    suffix = extracted.suffix
    subdomain = extracted.subdomain
    
    full_domain = f"{subdomain}.{domain_part}.{suffix}".strip(".")
    if full_domain.startswith("."): 
        full_domain = full_domain[1:]

    features["Domain"] = full_domain
    features["TLD"] = suffix
    features["NoOfSubDomain"] = subdomain.count('.') + 1 if subdomain else 0
    features["URLCharProb"] = calc_url_char_prob(url)
    features["CharContinuationRate"] = calc_char_continuation_rate(url)

    features["TLDLegitimateProb"] = calc_tld_legitimate_prob(url)

    # 2. Descàrrega del contingut HTML
    status_ok = False
    html_content = ""
    lines = []
    soup = BeautifulSoup("", "html.parser")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, timeout=10, headers=headers)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            status_ok = True
            lines = soup.prettify().splitlines()
    except Exception:
        status_ok = False

    # 3. Anàlisi del HTML
    if status_ok:
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        features["Title"] = title

        clean_dom = re.sub(r'[^a-zA-Z0-9]', '', domain_part).lower()
        clean_tit = re.sub(r'[^a-zA-Z0-9]', '', title).lower()
        
        features["DomainTitleMatchScore"] = calc_domain_title_match_score(domain_part, title)
        features["NoOfImage"] = len(soup.find_all('img')) + len(soup.find_all('svg'))
        features["NoOfJS"] = len(soup.find_all('script'))
        features["NoOfiFrame"] = len(soup.find_all('iframe'))

        # Meta descripció
        has_desc = 0
        if soup.find('meta', attrs={'name': re.compile(r'description', re.I)}) or \
           soup.find('meta', attrs={'property': re.compile(r'description', re.I)}):
            has_desc = 1
        features["HasDescription"] = has_desc

        # Xarxes socials
        socials = ["facebook", "twitter", "instagram", "linkedin", "youtube", "tiktok", "github"]
        all_hrefs = [a.get('href', '') for a in soup.find_all('a', href=True)]
        features["HasSocialNet"] = 1 if any(any(s in h for s in socials) for h in all_hrefs) else 0

        features["HasSubmitButton"] = 1 if \
            soup.find(['input', 'button'], {'type': 'submit'}) else 0
        
        features["HasPasswordField"] = 1 if soup.find('input', {'type': 'password'}) else 0
        features["HasHiddenFields"] = 1 if soup.find('input', {'type': 'hidden'}) else 0
        features["LineOfCode"] = len(lines)

        max_len = max(len(line) for line in lines) if lines else 0

        features["LargestLineLength"] = min(max_len, 7400)

        features["HasCopyrightInfo"] = 1 if \
            "copyright" in html_content.lower() or \
                "©" in html_content else 0
        
        features["NoOfEmptyRef"] = calc_no_of_empty_ref(soup, full_domain)

        # Referències internes vs externes
        self_refs = 0
        ext_refs = 0
        for link in all_hrefs:
            if domain_part in link or link.startswith('/'):
                self_refs += 1
            elif link.startswith('http'):
                ext_refs += 1
        
        # Recursos externs (img, script, iframe, link)
        external_resources = []
        external_resources.extend([link.get('href', '') 
                                   for link in soup.find_all('link', href=True)])
        
        external_resources.extend([script.get('src', '') 
                                   for script in soup.find_all('script', src=True)])
        
        external_resources.extend([img.get('src', '') 
                                   for img in soup.find_all('img', src=True)])
        
        external_resources.extend([iframe.get('src', '') 
                                   for iframe in soup.find_all('iframe', src=True)])

        for res in external_resources:
            if res.startswith('http') and domain_part not in res:
                ext_refs += 1

        features["NoOfSelfRef"] = self_refs
        features["NoOfExternalRef"] = ext_refs

    else:
        # Valors per defecte en cas d'error de càrrega
        for k in COLUMNES_MODEL:
            features[k] = 0
        features["DomainTitleMatchScore"] = 0.0
        features["Title"] = "Error de connexió"

    return features

# -- GESTIÓ DE MODELS I PREDICCIÓ ------------------------------------------------------------------

def load_models():
    """Carrega el discretitzador i l'agent RL."""
    if not os.path.exists(AGENT_PATH) or not os.path.exists(DISC_PATH):
        raise FileNotFoundError(f"Models no trobats a: {AGENT_PATH}")

    print(f"Loading models from {AGENT_PATH}...")
    with open(DISC_PATH, 'rb') as f: 
        discretizer = pickle.load(f)
    with open(AGENT_PATH, 'rb') as f: 
        agent = pickle.load(f)
    return discretizer, agent

def predict_single_url(features_dict, discretizer, agent):
    """
    Fa la predicció assegurant l'ordre correcte de les columnes.
    Retorna: (predicció, mètode)
    """
    df_new = pd.DataFrame([features_dict])
    
    # Forcem l'ordre de columnes i omplim buits amb 0
    df_ordered = df_new.reindex(columns=COLUMNES_MODEL, fill_value=0)

    # Discretització i predicció
    df_discrete = discretizer.transform(df_ordered)
    state = df_discrete.values[0]

    q_hits_start = agent.q_hits
    action = agent.choose_action(state, is_test=True)
    
    # Determinem si ha usat la taula Q o KNN
    method = "Q-Table (Memòria)" if agent.q_hits > q_hits_start else "KNN (Similitud)"
    return action, method

# -- GUI -------------------------------------------------------------------------------------------

def iniciar_interfaz(discretizer, agent):
    
    def procesar():
        url = entrada_url.get().strip()
        if not url:
            messagebox.showwarning("Avis", "Si us plau, introdueix una URL.")
            return

        # UI Updates
        btn_analizar.config(text="Analitzant...", state="disabled")
        lbl_resultado.config(text="Processant...", fg="blue")
        ventana.update()

        try:
            features = extreure_features(url)
            pred, method = predict_single_url(features, discretizer, agent)

            # Interpretació del resultat
            # Assumim: 1 = LEGIT, 0 = PHISHING (segons el teu codi original)
            if pred == 1:
                lbl_resultado.config(text=f"URL LEGÍTIMA\n({method})", fg="green")
            else:
                lbl_resultado.config(text=f"PHISHING DETECTAT\n({method})", fg="red")

        except Exception as e:
            messagebox.showerror("Error", f"Error durant l'anàlisi:\n{e}")
            lbl_resultado.config(text="Error", fg="black")
        finally:
            btn_analizar.config(text="Analitzar URL", state="normal")

    # Configuració finestra
    ventana = tk.Tk()
    ventana.title("Detector de Phishing (AI)")
    ventana.geometry("500x400")

    tk.Label(ventana, text="Introdueix la URL a analitzar", font=("Arial", 14, "bold")).pack(pady=20)

    entrada_url = tk.Entry(ventana, width=50, font=("Arial", 11))
    entrada_url.pack(pady=5)
    entrada_url.bind('<Return>', lambda event: procesar())

    btn_analizar = tk.Button(ventana, text="Analitzar URL", command=procesar, 
                           bg="#007BFF", fg="white", font=("Arial", 11, "bold"))
    btn_analizar.pack(pady=20)

    tk.Frame(ventana, height=2, bd=1, relief="sunken").pack(fill="x", padx=20, pady=10)
    
    lbl_resultado = tk.Label(ventana, text="Esperant URL...", font=("Arial", 16, "bold"))
    lbl_resultado.pack(pady=10)

    ventana.mainloop()

# -- RE-ENTRENAMENT --------------------------------------------------------------------------------

def reentrenar_amb_gui(ruta_script="scripts/rl_training.py"):
    """
    Obre una finestra de càrrega i executa l'entrenament en segon pla.
    """
    loading_window = tk.Tk()
    loading_window.title("Auto-Reparació del Sistema")
    loading_window.geometry("400x180")
    loading_window.resizable(False, False)

    # Centrar finestra
    sw, sh = loading_window.winfo_screenwidth(), loading_window.winfo_screenheight()
    loading_window.geometry(f"400x180+{(sw//2)-200}+{(sh//2)-90}")

    tk.Label(loading_window, 
             text = "Models no trobats", 
             font = ("Arial", 12, "bold"), 
             fg = "#d9534f").pack(pady = (15, 5))
    
    tk.Label(loading_window, 
             text = f"Executant re-entrenament...\nAixò pot trigar uns minuts.", 
             justify = "center").pack(pady = 5)

    progress = ttk.Progressbar(loading_window, 
                               orient = "horizontal", 
                               length = 320, 
                               mode = "indeterminate").pack(pady = 10)
    progress.start(10)

    def run_training_thread():
        try:
            full_script_path = os.path.join(BASE_DIR, ruta_script) if \
                not os.path.isabs(ruta_script) else ruta_script
            
            if not os.path.exists(full_script_path):
                print(f"Error: No trobo l'script a {full_script_path}")
                return

            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Afegir PYTHONPATH
            my_env = os.environ.copy()
            project_root = BASE_DIR
            my_env["PYTHONPATH"] = project_root + os.pathsep + my_env.get("PYTHONPATH", "")

            executable_cmd = "python" if getattr(sys, 'frozen', False) else sys.executable
            
            print(f"Iniciant re-entrenament: {full_script_path}")
            subprocess.run([executable_cmd, full_script_path], 
                           check = True, 
                           startupinfo = startupinfo, 
                           env = my_env)
            
            print("Entrenament finalitzat.")

        except Exception as e:
            print(f"Error durant l'entrenament: {e}")
        finally:
            loading_window.after(0, loading_window.destroy)

    t = threading.Thread(target=run_training_thread)
    t.start()
    loading_window.mainloop()

# --- BLOC PRINCIPAL ---

if __name__ == "__main__":
    discretizer = None
    agent = None

    try:
        # Carregar models existents
        discretizer, agent = load_models()
        print("Models carregats correctament.")
        iniciar_interfaz(discretizer, agent)

    except (FileNotFoundError, Exception) as e:
        print(f"Alerta: {e}")
        print("Iniciant protocol de recuperació (re-entrenament)...")

        # Intentem re-entrenar
        reentrenar_amb_gui("scripts/rl_training.py")

        try:
            # Carregar després d'entrenar
            discretizer, agent = load_models()
            print("Models recuperats.")
            iniciar_interfaz(discretizer, agent)

        except Exception as ex:
            print(f"Error Fatal: {ex}")
            # Fiquem un root tk temporal per mostrar l'error si la GUI no ha arrencat
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error Fatal", f"No s'ha pogut iniciar l'aplicació.\nError: {ex}")