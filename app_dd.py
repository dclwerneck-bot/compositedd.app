import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Adiciona o diretório atual ao sys.path para importar o motor engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.clt import Lamina, Laminate
from engine.failure import FailureAnalysis
from engine.optimization import DDOptimizer
from engine.ml_surrogate import MLSurrogate

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS PERSONALIZADO
# ==========================================
st.set_page_config(
    page_title="Composite Double-Double vs Quad Expert",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# Gerenciamento do Estado da Sessão para Navegação
if 'started' not in st.session_state:
    st.session_state.started = False

def start_app():
    st.session_state.started = True

def go_home():
    st.session_state.started = False

# Estilização CSS Dark Glassmorphism com Alto Contraste para Leitura Perfeita
st.markdown("""
<style>
    /* Fundo Principal Escuro com Alto Contraste */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Garantir Legibilidade de Textos em Geral */
    p, span, label, li, h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC;
    }
    
    .subtext {
        color: #CBD5E1 !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* POP-OVER E DROPDOWNS (Lista Suspensa de Materiais) */
    div[data-baseweb="popover"], 
    div[data-baseweb="menu"], 
    ul[role="listbox"],
    div[data-baseweb="select"] ul {
        background-color: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px !important;
    }
    
    div[data-baseweb="popover"] li, 
    div[data-baseweb="popover"] span,
    div[data-baseweb="menu"] div,
    div[data-baseweb="menu"] span,
    li[role="option"],
    div[role="option"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    
    li[role="option"]:hover, 
    div[role="option"]:hover,
    li[aria-selected="true"],
    div[aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
    }

    /* EXPANDERS / ACORDEÕES DA SEÇÃO TEÓRICA */
    div[data-testid="stExpander"] {
        background-color: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }
    
    div[data-testid="stExpander"] summary {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
    }
    
    div[data-testid="stExpander"] summary:hover {
        background-color: #2D3748 !important;
    }
    
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary div,
    div[data-testid="stExpander"] details summary {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] li {
        color: #E2E8F0 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    /* CARDS E MÉTRICAS */
    .glass-card {
        background-color: #1E293B !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    
    .metric-card {
        background-color: #151D2A !important;
        border-left: 4px solid #38BDF8 !important;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-title {
        color: #94A3B8 !important;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #38BDF8 !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Abas estilizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #111827;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    .stTabs [data-baseweb="tab"] {
        height: 52px;
        border-radius: 10px;
        color: #CBD5E1 !important;
        font-weight: 700;
        font-size: 16px;
        padding: 0 24px;
        background-color: transparent;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.5);
    }

    /* Botão Iniciar da Landing Page */
    .start-btn-container div.stButton > button {
        background: linear-gradient(90deg, #0284C7 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        border-radius: 16px !important;
        padding: 22px 48px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(2, 132, 199, 0.6) !important;
        transition: all 0.3s ease !important;
        display: block !important;
        margin: 0 auto !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
    }
    
    .start-btn-container div.stButton > button:hover {
        transform: scale(1.05) translateY(-3px) !important;
        box-shadow: 0 14px 35px rgba(2, 132, 199, 0.8) !important;
        background: linear-gradient(90deg, #0369A1 0%, #1D4ED8 100%) !important;
    }
    
    /* Botões Padrão do Dashboard */
    div.stButton > button {
        background: linear-gradient(90deg, #0284C7 0%, #2563EB 100%);
        color: #FFFFFF !important;
        font-size: 16px;
        font-weight: 700;
        border-radius: 12px;
        padding: 14px 28px;
        border: none;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.6);
        background: linear-gradient(90deg, #0369A1 0%, #1D4ED8 100%);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
        color: #F1F5F9 !important;
    }
    
    /* Inputs e Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MATERIAIS PREDEFINIDOS
# ==========================================
MATERIALS = {
    "Carbono AS4 / Epóxi 3501-6": Lamina(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, XT=2100e6, XC=1400e6, YT=50e6, YC=200e6, S=90e6, t=0.125e-3, name="Carbono AS4/Epóxi"),
    "Carbono T300 / Epóxi 914": Lamina(E1=138e9, E2=9.0e9, G12=6.9e9, nu12=0.3, XT=1500e6, XC=1200e6, YT=40e6, YC=180e6, S=70e6, t=0.125e-3, name="Carbono T300/Epóxi"),
    "Vidro E / Epóxi": Lamina(E1=45e9, E2=12e9, G12=5.5e9, nu12=0.28, XT=1000e6, XC=800e6, YT=30e6, YC=120e6, S=40e6, t=0.15e-3, name="Vidro E/Epóxi"),
    "Kevlar 49 / Epóxi": Lamina(E1=76e9, E2=5.5e9, G12=2.1e9, nu12=0.34, XT=1400e6, XC=280e6, YT=30e6, YC=140e6, S=60e6, t=0.125e-3, name="Kevlar 49/Epóxi")
}

# ==========================================
# 3. PÁGINA INICIAL (LANDING PAGE)
# ==========================================
if not st.session_state.started:
    col_hero_left, col_hero_center, col_hero_right = st.columns([0.1, 3.8, 0.1])
    with col_hero_center:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding-top: 10px;">
                <div style="font-size: 3.8rem; margin-bottom: 10px;">🧬</div>
                <h1 style="font-size: 3rem; margin-bottom: 15px;" class="gradient-text">Composite Double-Double Expert</h1>
                <p style="font-size: 1.2rem; color: #E2E8F0; max-width: 750px; line-height: 1.6; margin: 0 auto 25px auto;">
                    Plataforma Científica Interativa para Análise, Comparação Analítica (CLT) e Otimização via Aprendizado de Máquinas 
                    de Laminados <b>Double-Double (DD)</b> vs <b>Quad</b> baseada na Teoria de <b>Stephen W. Tsai</b>.
                </p>
                <div style="display: flex; gap: 12px; margin-bottom: 35px; flex-wrap: wrap; justify-content: center;">
                    <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38BDF8; padding: 8px 18px; border-radius: 20px; font-weight: 600; color: #38BDF8;">📚 FAQ & Teoria de Tsai</span>
                    <span style="background: rgba(129, 140, 248, 0.15); border: 1px solid #818CF8; padding: 8px 18px; border-radius: 20px; font-weight: 600; color: #818CF8;">📐 Calculadora CLT & Polares Ex(θ)</span>
                    <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; padding: 8px 18px; border-radius: 20px; font-weight: 600; color: #10B981;">🤖 Otimização & ML Embarcado</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="start-btn-container">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([0.8, 2.4, 0.8])
            with c2:
                if st.button("🚀 INICIAR APLICAÇÃO"):
                    start_app()
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="padding-bottom: 10px;"></div>', unsafe_allow_html=True)

# ==========================================
# 4. DASHBOARD DA APLICAÇÃO PRINCIPAL
# ==========================================
else:
    # Sidebar
    with st.sidebar:
        if st.button("🏠 Voltar para a Tela Inicial"):
            go_home()
            st.rerun()
            
        st.markdown("---")
        st.markdown("## ⚙️ Painel do Material")
        mat_choice = st.selectbox("Selecione o Material", list(MATERIALS.keys()))
        base_lamina = MATERIALS[mat_choice]
        
        with st.expander("🛠️ Personalizar Propriedades"):
            E1 = st.number_input("E1 (GPa)", value=base_lamina.E1 / 1e9, step=1.0) * 1e9
            E2 = st.number_input("E2 (GPa)", value=base_lamina.E2 / 1e9, step=0.5) * 1e9
            G12 = st.number_input("G12 (GPa)", value=base_lamina.G12 / 1e9, step=0.5) * 1e9
            nu12 = st.number_input("ν12", value=base_lamina.nu12, step=0.01)
            t_ply = st.number_input("Espessura por lâmina (mm)", value=base_lamina.t * 1000, step=0.01) / 1000.0
            
            selected_lamina = Lamina(
                E1=E1, E2=E2, G12=G12, nu12=nu12,
                XT=base_lamina.XT, XC=base_lamina.XC, YT=base_lamina.YT, YC=base_lamina.YC, S=base_lamina.S,
                t=t_ply, name="Personalizado"
            )
            
        st.markdown("---")
        st.markdown("### 🔄 Ângulos Double-Double (DD)")
        phi_val = st.slider("Ângulo Φ (graus)", 0.0, 90.0, 22.5, step=0.5)
        psi_val = st.slider("Ângulo Ψ (graus)", 0.0, 90.0, 67.5, step=0.5)
        dd_repeats = st.slider("Repetições (n)", 1, 6, 2)
        
        st.markdown("---")
        st.markdown("### 🔷 Laminado Quad (Referência)")
        quad_type = st.selectbox("Arranjo Quad", ["[0/90/±45]s (Quase-Isotrópico)", "[0/±45/90]s", "[0/0/90/90]s (Ortotrópico Cruzado)"])

    # Construção dos Laminados
    if 'selected_lamina' not in locals():
        selected_lamina = base_lamina

    if quad_type == "[0/90/±45]s (Quase-Isotrópico)":
        angles_quad = [0, 90, 45, -45, -45, 45, 90, 0] * (dd_repeats // 2 if dd_repeats >= 2 else 1)
    elif quad_type == "[0/±45/90]s":
        angles_quad = [0, 45, -45, 90, 90, -45, 45, 0] * (dd_repeats // 2 if dd_repeats >= 2 else 1)
    else:
        angles_quad = [0, 0, 90, 90, 90, 90, 0, 0] * (dd_repeats // 2 if dd_repeats >= 2 else 1)

    sub_dd = [phi_val, -phi_val, psi_val, -psi_val] * dd_repeats
    angles_dd = sub_dd + sub_dd[::-1]

    lam_quad = Laminate(selected_lamina, angles_quad)
    lam_dd = Laminate(selected_lamina, angles_dd)

    # Cabeçalho Principal
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 25px 0;">
        <h1 style="font-size: 2.5rem;" class="gradient-text">Composite Double-Double (DD) vs Quad Expert</h1>
        <p class="subtext" style="max-width: 850px; margin: 0 auto;">
            Análise Avançada da Teoria Clássica de Laminados (CLT), Critério de Falha de Tsai-Wu e Otimização via IA.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Abas da Aplicação
    tab_faq, tab_calc, tab_ml = st.tabs([
        "📚 1. Teoria & FAQ (Fundamentação)",
        "📐 2. Calculadora & Comparador Analítico (CLT)",
        "🤖 3. Otimização & Machine Learning (Surrogate)"
    ])

    # ------------------------------------------
    # ABA 1: TEORIA & FAQ
    # ------------------------------------------
    with tab_faq:
        st.markdown("""
        <div class="glass-card">
            <h2>📖 Fundamentação Técnica e Perguntas Frequentes (FAQ)</h2>
            <p class="subtext">Entenda por que a tecnologia <b>Double-Double (DD)</b> representa um avanço em relação aos laminados convencionais <b>Quad</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            with st.expander("❓ O que é um Laminado Quad e quais são suas limitações?", expanded=True):
                st.markdown(r"""
                Os laminados tradicionais **Quad** são compostos obrigatoriamente por 4 ângulos discretos: **0°, 90°, +45° e -45°**.
                
                **Limitações Principais:**
                - **Descontinuidade de rigidez**: As opções de espessura e orientações mudam em degraus rígidos de 25% (regra dos 10%).
                - **Acoplamento por delaminação**: Para evitar empenamento e acoplamento de flexão-torção, é necessário espelhamento complexo.
                - **Trincamento Térmico/Mecânico**: Camadas de mesmo ângulo agrupadas (ex: 0° espessos) sofrem microtrincamento sob tensões residuais de cura.
                """)

            with st.expander("✨ O que é a Tecnologia Double-Double (DD)?"):
                st.markdown(r"""
                Proposta pelo renomado Prof. **Stephen W. Tsai (Stanford)**, a arquitetura **Double-Double (DD)** substitui o bloco Quad por blocos de 4 camadas compostas por dois pares de ângulos balanceados:
                $$\text{Bloco DD} = [\pm \Phi / \pm \Psi]$$
                
                **Vantagens Fundamentais:**
                1. **Otimização Contínua**: \(\Phi\) e \(\Psi\) variam continuamente de 0° a 90°, permitindo ajustar exatamente as propriedades à carga.
                2. **Homogeneidade e Homogeneização Flexional**: Ao repetir o bloco sub-laminado fino \([\pm \Phi / \pm \Psi]\), a matriz de acoplamento \(B \to 0\) e \(D \propto A\).
                3. **Sem agrupamento espesso**: Evita o microtrincamento e aumenta dramaticamente a resistência à fadiga e ao impacto.
                """)

        with col_t2:
            with st.expander("📖 Manual de Uso do Aplicativo (Guia Passo a Passo)", expanded=False):
                st.markdown(r"""
                **1. Seleção e Personalização do Material (Sidebar)**
                - Escolha entre os materiais padrão (*Carbono AS4, Carbono T300, Vidro E, Kevlar*) ou expanda **🛠️ Personalizar Propriedades** para digitar os seus próprios valores de \(E_1, E_2, G_{12}, \nu_{12}\) e espessura \(t\).
                
                **2. Configuração dos Laminados (Sidebar)**
                - **Double-Double (DD)**: Ajuste os sliders de ângulo \(\Phi\) e \(\Psi\) (0° a 90°) e o número de repetições \(n\) para formar o bloco \([\pm\Phi / \pm\Psi]_n\).
                - **Quad**: Escolha o laminado de referência (\([0/90/\pm 45]_s\), \([0/\pm 45/90]_s\) ou Ortotrópico Cruzado).
                
                **3. Aba 2 - Calculadora & Comparador Analítico (CLT)**
                - **Métricas Rápidas**: Compare os módulos \(E_x, E_y, G_{xy}\) e \(\nu_{xy}\) em tempo real.
                - **Gráfico Polar \(E_x(\theta)\)**: Analise a variação da rigidez direcional de 0° a 360°.
                - **Análise de Falha Tsai-Wu**: Digite as cargas no plano (\(N_x, N_y, N_{xy}\) em kN/m) para calcular a Margem de Segurança e a camada crítica.
                
                **4. Aba 3 - Otimização & Machine Learning**
                - Digite a razão de rigidez desejada (\(E_x / E_y\)) e clique em **🔍 Buscar Ângulos Ótimos** ou **🚀 Treinar/Executar Predição Inteligente via ML** para obter a recomendação instantânea.
                """)

            with st.expander("🧮 Quais são as Equações Fundamentais da CLT?", expanded=False):
                st.markdown(r"""
                A **Teoria Clássica de Laminados (CLT)** relaciona as forças no plano \(\mathbf{N}\) e momentos \(\mathbf{M}\) às deformações do plano médio \(\boldsymbol{\varepsilon}^0\) e curvaturas \(\boldsymbol{\kappa}\):
                
                $$\begin{bmatrix} \mathbf{N} \\ \mathbf{M} \end{bmatrix} = \begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B} & \mathbf{D} \end{bmatrix} \begin{bmatrix} \boldsymbol{\varepsilon}^0 \\ \boldsymbol{\kappa} \end{bmatrix}$$
                
                - **Matriz Extensional (\(\mathbf{A}\))**: \(A_{ij} = \sum_{k=1}^N \bar{Q}_{ij}^{(k)} (z_k - z_{k-1})\)
                - **Matriz de Acoplamento (\(\mathbf{B}\))**: \(B_{ij} = \frac{1}{2} \sum_{k=1}^N \bar{Q}_{ij}^{(k)} (z_k^2 - z_{k-1}^2)\)
                - **Matriz Flexional (\(\mathbf{D}\))**: \(D_{ij} = \frac{1}{3} \sum_{k=1}^N \bar{Q}_{ij}^{(k)} (z_k^3 - z_{k-1}^3)\)
                """)

            with st.expander("📚 Referências Bibliográficas & Citações"):
                st.markdown(r"""
                - **Tsai, S. W.** (2018). *Double-Double: A Breakthrough in Composite Design*. Stanford University Press.
                - **Tsai, S. W., & Melo, J. D. D.** (2020). *Composite Materials Design and Optimization*.
                - **Jones, R. M.** (1999). *Mechanics of Composite Materials*. Taylor & Francis.
                - **Daniel, I. M., & Ishai, O.** (2006). *Engineering Mechanics of Composite Materials*. Oxford University Press.
                """)

    # ------------------------------------------
    # ABA 2: CALCULADORA & COMPARADOR ANALÍTICO
    # ------------------------------------------
    with tab_calc:
        st.markdown("""
        <div class="glass-card">
            <h2>📐 Comparador Analítico Lado a Lado: Quad vs Double-Double</h2>
            <p class="subtext">Análise de propriedades mecânicas equivalentes, matrizes de rigidez e diagrama polar de rigidez.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cards numéricos de alta legibilidade
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Módulo Ex (Quad vs DD)</div>
                <div class="metric-value">{lam_quad.Ex / 1e9:.2f} GPa <span style="font-size:0.9rem; color:#94A3B8;">(Quad)</span></div>
                <div class="metric-value" style="color:#10B981 !important;">{lam_dd.Ex / 1e9:.2f} GPa <span style="font-size:0.9rem; color:#94A3B8;">(DD)</span></div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Módulo Ey (Quad vs DD)</div>
                <div class="metric-value">{lam_quad.Ey / 1e9:.2f} GPa <span style="font-size:0.9rem; color:#94A3B8;">(Quad)</span></div>
                <div class="metric-value" style="color:#10B981 !important;">{lam_dd.Ey / 1e9:.2f} GPa <span style="font-size:0.9rem; color:#94A3B8;">(DD)</span></div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Cisalhamento Gxy (Quad vs DD)</div>
                <div class="metric-value">{lam_quad.Gxy / 1e9:.2f} GPa <span style="font-size:0.9rem; color:#94A3B8;">(Quad)</span></div>
                <div class="metric-value" style="color:#10B981 !important;">{lam_dd.Gxy / 1e9:.2f} GPa <span style="font-size:0.9rem; color:#94A3B8;">(DD)</span></div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Poisson νxy (Quad vs DD)</div>
                <div class="metric-value">{lam_quad.nu_xy:.3f} <span style="font-size:0.9rem; color:#94A3B8;">(Quad)</span></div>
                <div class="metric-value" style="color:#10B981 !important;">{lam_dd.nu_xy:.3f} <span style="font-size:0.9rem; color:#94A3B8;">(DD)</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        col_chart, col_matrices = st.columns([1.3, 1.0])
        
        with col_chart:
            st.subheader("🌐 Diagrama Polar de Rigidez Extensional Ex(θ) [0° a 360°]")
            
            theta_q, Ex_q = lam_quad.get_polar_stiffness()
            theta_dd, Ex_dd = lam_dd.get_polar_stiffness()
            
            fig_polar = go.Figure()
            
            fig_polar.add_trace(go.Scatterpolar(
                r=Ex_q, theta=theta_q, mode='lines', name=f'Quad {quad_type}',
                line=dict(color='#38BDF8', width=3)
            ))
            
            fig_polar.add_trace(go.Scatterpolar(
                r=Ex_dd, theta=theta_dd, mode='lines', name=f'Double-Double [±{phi_val}/±{psi_val}]',
                line=dict(color='#10B981', width=3, dash='dot')
            ))
            
            fig_polar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    radialaxis=dict(visible=True, showticklabels=True, title=dict(text="Ex (GPa)", font=dict(color="#F8FAFC"))),
                    angularaxis=dict(direction="clockwise")
                ),
                legend=dict(orientation="h", y=-0.15),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig_polar, use_container_width=True)

        with col_matrices:
            st.subheader("📊 Matriz de Rigidez Extensional A [MN/m]")
            
            df_A_quad = pd.DataFrame(lam_quad.A / 1e6, columns=['A11', 'A12', 'A16'], index=['A11', 'A12', 'A16'])
            df_A_dd = pd.DataFrame(lam_dd.A / 1e6, columns=['A11', 'A12', 'A16'], index=['A11', 'A12', 'A16'])
            
            st.markdown("**Matriz A - Quad (MN/m):**")
            st.dataframe(df_A_quad.style.format("{:.2f}"), use_container_width=True)
            
            st.markdown(f"**Matriz A - Double-Double [±{phi_val}/±{psi_val}] (MN/m):**")
            st.dataframe(df_A_dd.style.format("{:.2f}"), use_container_width=True)

        st.markdown("---")
        st.subheader("🛡️ Análise de Falha Tsai-Wu Sob Carga no Plano")
        
        col_loads, col_failure = st.columns([1, 1])
        with col_loads:
            st.markdown("##### Definição do Vetor de Cargas (kN/m)")
            Nx_input = st.number_input("Nx (Tração/Compressão x)", value=100.0, step=10.0) * 1000.0
            Ny_input = st.number_input("Ny (Tração/Compressão y)", value=20.0, step=10.0) * 1000.0
            Nxy_input = st.number_input("Nxy (Cisalhamento no plano)", value=10.0, step=5.0) * 1000.0

        with col_failure:
            st.markdown("##### Resultados do Critério de Tsai-Wu")
            fa_q = FailureAnalysis(lam_quad)
            res_q = fa_q.evaluate_laminate_failure(Nx_input, Ny_input, Nxy_input)
            
            fa_dd = FailureAnalysis(lam_dd)
            res_dd = fa_dd.evaluate_laminate_failure(Nx_input, Ny_input, Nxy_input)
            
            st.markdown(f"""
            - **Quad**: 
              - Índice de Falha Tsai-Wu (FI): `{res_q['max_FI']:.4f}` ({'✅ Seguro' if res_q['is_safe'] else '❌ Falha'})
              - Margem de Segurança: `{res_q['min_margin']:.2f}` (Camada Crítica: `{res_q['critical_ply']}`)
            - **Double-Double**: 
              - Índice de Falha Tsai-Wu (FI): `{res_dd['max_FI']:.4f}` ({'✅ Seguro' if res_dd['is_safe'] else '❌ Falha'})
              - Margem de Segurança: `{res_dd['min_margin']:.2f}` (Camada Crítica: `{res_dd['critical_ply']}`)
            """)

    # ------------------------------------------
    # ABA 3: OTIMIZAÇÃO & MACHINE LEARNING
    # ------------------------------------------
    with tab_ml:
        st.markdown("""
        <div class="glass-card">
            <h2>🤖 Otimizador de Ângulos DD & Aprendizado de Máquinas (Surrogate)</h2>
            <p class="subtext">Encontre os ângulos <b>(Φ, Ψ)</b> ótimos para qualquer razão de rigidez ou combinação de cargas instantaneamente.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_opt1, col_opt2 = st.columns([1, 1])
        
        with col_opt1:
            st.markdown("### 🎯 Otimização Contínua Analítica")
            target_ratio = st.slider("Razão de Rigidez Desejada (Ex / Ey)", 0.2, 5.0, 1.5, step=0.1)
            
            if st.button("🔍 Buscar Ângulos Ótimos (Φ, Ψ)"):
                opt = DDOptimizer(selected_lamina, n_repeats=dd_repeats)
                res_opt = opt.optimize_for_stiffness_ratio(target_ratio=target_ratio, step=2.5)
                
                st.success(f"**Ângulos Encontrados:** Φ = {res_opt['phi']}°,  Ψ = {res_opt['psi']}°")
                st.markdown(f"""
                - **Ex resultante:** `{res_opt['Ex']:.2f} GPa`
                - **Ey resultante:** `{res_opt['Ey']:.2f} GPa`
                - **Razão obtida (Ex/Ey):** `{res_opt['ratio']:.3f}` (Alvo: `{target_ratio:.3f}`)
                """)

        with col_opt2:
            st.markdown("### ⚡ Modelo Surrogate de Machine Learning")
            st.info("O modelo surrogate executa a inferência offline sem necessidade de servidor em nuvem.")
            
            ml = MLSurrogate(selected_lamina)
            
            if st.button("🚀 Treinar/Executar Predição Inteligente via ML"):
                with st.spinner("Treinando modelo RandomForest localmente..."):
                    trained = ml.train_surrogate()
                    pred_phi, pred_psi, used_ml = ml.predict_dd(target_ratio)
                    
                if used_ml:
                    st.balloons()
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.25); border: 1px solid #10B981; padding: 18px; border-radius: 12px;">
                        <h4 style="color:#10B981; margin:0;">Modelo ML Recomendou:</h4>
                        <p style="font-size: 1.4rem; font-weight: bold; margin-top: 5px; color:#FFFFFF;">Φ = {pred_phi}°  |  Ψ = {pred_psi}°</p>
                        <small style="color:#CBD5E1;">Inferência realizada via Random Forest Regressor embarcado em milissegundos.</small>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        ### 🌐 Implantação e Publicação Gratuita (PWA / GitHub Pages)
        Este aplicativo foi desenvolvido para rodar sem nenhum custo de servidor. 
        Para hospedar gratuitamente para seus alunos e pesquisadores:
        1. Suba este repositório para o **GitHub**.
        2. Ative o **GitHub Pages** nas configurações do repositório.
        3. O app pode ser compilado como **PWA estático** via **Pyodide** (Python no navegador), garantindo execução 100% no cliente sem mensalidades nem infraestrutura paga.
        """)
