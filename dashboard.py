# ================================================================
#  DASHBOARD INTERACTIVO CON PANEL + PLOTLY
#  Ejecutar con:  panel serve dashboard.py --show
#  O dentro del notebook:  dashboard_pane.servable()
# ================================================================
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint

from numba import njit
import time

pn.extension('plotly')

# ---------------------------------------------------------------
#  Núcleo: simulación determinista y estocástica
# ---------------------------------------------------------------
def gene_ode(y, t, lam, mu, nu, sigma):
    P1, P2, P3 = y
    return [-lam*P1 + mu*P2, lam*P1 - mu*P2, nu*P2 - sigma*P3]


@njit(cache=True)
def gillespie_direct_numba(lam, mu, nu, sigma, t_max, seed):
    np.random.seed(seed)
    max_steps = 2_000_000
    times = np.empty(max_steps, dtype=np.float64)
    p1a = np.empty(max_steps, dtype=np.int64)
    p2a = np.empty(max_steps, dtype=np.int64)
    p3a = np.empty(max_steps, dtype=np.int64)

    P1, P2, P3 = 1, 0, 0
    t = 0.0
    n = 0
    times[0] = 0.0
    p1a[0], p2a[0], p3a[0] = P1, P2, P3

    while n < max_steps - 1:
        a1 = lam*P1; a2 = mu*P2; a3 = nu*P2; a4 = sigma*P3
        a_tot = a1 + a2 + a3 + a4
        if a_tot <= 0.0: break

        r1 = np.random.random(); r2 = np.random.random()
        tau = -np.log(1.0 - r1) / a_tot
        t_new = t + tau
        if t_new > t_max:
            n += 1
            times[n] = t_max; p1a[n] = P1; p2a[n] = P2; p3a[n] = P3
            break

        target = r2 * a_tot
        if   target < a1:                     P1 -= 1; P2 += 1
        elif target < a1 + a2:                P2 -= 1; P1 += 1
        elif target < a1 + a2 + a3:           P3 += 1
        else:                                 P3 -= 1

        t = t_new; n += 1
        times[n] = t; p1a[n] = P1; p2a[n] = P2; p3a[n] = P3

    return times[:n+1], p1a[:n+1], p2a[:n+1], p3a[:n+1]


# ---------------------------------------------------------------
#  Widgets
# ---------------------------------------------------------------
widgets = {
    'lam'   : pn.widgets.FloatSlider(name='λ (activación)',   start=0.2, end=3.0,  value=1.0,  step=0.1),
    'mu'    : pn.widgets.FloatSlider(name='μ (inactivación)', start=1.0, end=10.0, value=5.0,  step=0.5),
    'nu'    : pn.widgets.FloatSlider(name='ν (síntesis)',     start=0.2, end=3.0,  value=1.0,  step=0.1),
    'sigma' : pn.widgets.FloatSlider(name='σ (degradación)',  start=0.005, end=0.1, value=0.02, step=0.005),
    't_max' : pn.widgets.FloatSlider(name='t_max (s)',        start=20,  end=500,  value=100,  step=10),
    'n_cells':pn.widgets.IntSlider(name='Nº células',         start=50,  end=2000, value=300,  step=50),
    'seed'  : pn.widgets.IntSlider(name='Semilla',            start=0,   end=999,  value=42,   step=1),
}

# ---------------------------------------------------------------
#  Función que construye las gráficas
# ---------------------------------------------------------------
def make_plots(lam, mu, nu, sigma, t_max, n_cells, seed):
    np.random.seed(int(seed))

    # ---- Determinista ----
    t_d = np.linspace(0, t_max, 2000)
    sol = odeint(gene_ode, [1, 0, 0], t_d, args=(lam, mu, nu, sigma))

    # ---- Estocástico: una trayectoria ----
    t_s, p1, p2, p3 = gillespie_direct_numba(lam, mu, nu, sigma,
                                             float(t_max), int(seed))

    # ---- Estocástico: población ----
    finals = np.empty((int(n_cells), 3), dtype=np.int64)
    for i in range(int(n_cells)):
        _, _, _, p3i = gillespie_direct_numba(lam, mu, nu, sigma,
                                              float(t_max),
                                              int(seed) + i + 1)
        finals[i] = (0, 0, p3i[-1])

    # ============================================================
    #  FIGURA 1 — Determinista vs Estocástico
    # ============================================================
    fig1 = make_subplots(rows=3, cols=2,
                         shared_xaxes=True,
                         column_titles=['Determinista (ODE)',
                                        'Estocástico (Gillespie)'],
                         vertical_spacing=0.08)

    colors = {'P1': 'royalblue', 'P2': 'purple', 'P3': 'deeppink'}

    # Columna izquierda: determinista
    fig1.add_trace(go.Scatter(x=t_d, y=sol[:, 0], name='P1 (ODE)',
                              line=dict(color=colors['P1'], width=2),
                              legendgroup='det', showlegend=True),
                   row=1, col=1)
    fig1.add_trace(go.Scatter(x=t_d, y=sol[:, 1], name='P2 (ODE)',
                              line=dict(color=colors['P2'], width=2),
                              legendgroup='det'),
                   row=2, col=1)
    fig1.add_trace(go.Scatter(x=t_d, y=sol[:, 2], name='P3 (ODE)',
                              line=dict(color=colors['P3'], width=2.5),
                              legendgroup='det'),
                   row=3, col=1)

    # Columna derecha: estocástico
    fig1.add_trace(go.Scatter(x=t_s, y=p1, name='P1 (Gillespie)',
                              line=dict(color=colors['P1'], width=1.5, shape='hv'),
                              legendgroup='sto'),
                   row=1, col=2)
    fig1.add_trace(go.Scatter(x=t_s, y=p2, name='P2 (Gillespie)',
                              line=dict(color=colors['P2'], width=1.5, shape='hv'),
                              legendgroup='sto'),
                   row=2, col=2)
    fig1.add_trace(go.Scatter(x=t_s, y=p3, name='P3 (Gillespie)',
                              line=dict(color=colors['P3'], width=2, shape='hv'),
                              legendgroup='sto'),
                   row=3, col=2)

    fig1.update_yaxes(title_text='P1', row=1, col=1)
    fig1.update_yaxes(title_text='P2', row=2, col=1)
    fig1.update_yaxes(title_text='P3', row=3, col=1)
    fig1.update_xaxes(title_text='Tiempo (s)', row=3, col=1)
    fig1.update_xaxes(title_text='Tiempo (s)', row=3, col=2)
    fig1.update_layout(height=750, title_text='<b>Determinista vs Estocástico</b>',
                       template='plotly_white', hovermode='x unified')

    # ============================================================
    #  FIGURA 2 — Histograma + CDF de P3
    # ============================================================
    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=['Histograma P3 (final)',
                                         'CDF empírica P3'])

    # Histograma
    fig2.add_trace(go.Histogram(x=finals[:, 2], nbinsx=40,
                                marker_color=colors['P3'],
                                opacity=0.8, name='P3'),
                   row=1, col=1)

    # Línea determinista
    det_value = nu * (lam/(lam+mu)) / sigma
    fig2.add_vline(x=det_value, line_dash='dash', line_color='black',
                   annotation_text=f'Determinista = {det_value:.2f}',
                   annotation_position='top', row=1, col=1)

    # CDF
    sorted_p3 = np.sort(finals[:, 2])
    cdf_vals  = np.arange(1, len(sorted_p3)+1) / len(sorted_p3)
    fig2.add_trace(go.Scatter(x=sorted_p3, y=cdf_vals, mode='lines+markers',
                              line=dict(color=colors['P3'], width=2),
                              marker=dict(size=4),
                              name='CDF P3'),
                   row=1, col=2)

    fig2.update_xaxes(title_text='Nº de moléculas P3', row=1, col=1)
    fig2.update_yaxes(title_text='Frecuencia', row=1, col=1)
    fig2.update_xaxes(title_text='Nº de moléculas P3', row=1, col=2)
    fig2.update_yaxes(title_text='P(X ≤ x)', row=1, col=2)
    fig2.update_layout(height=420, template='plotly_white',
                       showlegend=False,
                       title_text=f'<b>Distribución estacionaria (N = {len(finals)} células)</b>')

    # ============================================================
    #  Métricas numéricas
    # ============================================================
    metrics = {
        'Media P3 (estoc.)'   : f'{finals[:,2].mean():.3f}',
        'Desv. est. P3'       : f'{finals[:,2].std():.3f}',
        'Valor determinista'  : f'{det_value:.3f}',
        'Error relativo'      : f'{abs(finals[:,2].mean() - det_value)/det_value*100:.2f}%',
        'P3 mínimo'           : f'{finals[:,2].min()}',
        'P3 máximo'           : f'{finals[:,2].max()}',
        'Nº células'          : f'{len(finals):,}',
        'Nº eventos (1 tray.)': f'{len(t_s):,}',
    }
    metrics_html = '<br>'.join(
        [f'<b>{k}:</b> {v}' for k, v in metrics.items()]
    )

    return fig1, fig2, metrics_html, finals


# ---------------------------------------------------------------
#  Callback reactivo
# ---------------------------------------------------------------
@pn.depends(**{k: w for k, w in widgets.items()})
def update_dashboard(lam, mu, nu, sigma, t_max, n_cells, seed):
    t0 = time.perf_counter()
    fig1, fig2, metrics_html, finals = make_plots(
        lam, mu, nu, sigma, t_max, n_cells, seed)
    elapsed = time.perf_counter() - t0

    header = pn.pane.Markdown(f"""
    ### 🧬 Dashboard del Modelo Génico de Goss & Peccoud
    **Tiempo de cómputo:** `{elapsed:.2f} s` &nbsp;|&nbsp;
    **Estado estacionario teórico P3*** = ν·λ/((λ+μ)·σ) = `{nu*(lam/(lam+mu))/sigma:.3f}`
    """)

    metrics_pane = pn.pane.HTML(
        f'<div style="font-family: monospace; font-size: 13px; '
        f'background: #f5f5f5; padding: 12px; border-radius: 6px;">'
        f'{metrics_html}</div>'
    )

    # Botón de descarga
    df = pd.DataFrame(finals, columns=['P1', 'P2', 'P3'])
    csv_button = pn.widgets.FileDownload(
        callback=lambda: df.to_csv(index=False).encode(),
        filename='resultados_gillespie.csv',
        button_type='success',
        label='📥 Descargar CSV',
    )

    return pn.Column(
        header,
        pn.Row(
            pn.pane.Plotly(fig1, config={'responsive': True}, sizing_mode='stretch_width'),
        ),
        pn.Row(
            pn.pane.Plotly(fig2, config={'responsive': True}, sizing_mode='stretch_width'),
        ),
        pn.Row(
            pn.Column(pn.pane.Markdown('### 📊 Métricas'), metrics_pane),
            pn.Column(pn.pane.Markdown('### 💾 Exportar'), csv_button),
        ),
    )


# ---------------------------------------------------------------
#  Layout final
# ---------------------------------------------------------------
sidebar = pn.Column(
    pn.pane.Markdown('## 🎛️ Controles'),
    *widgets.values(),
    width=320,
)

dashboard_pane = pn.Row(
    sidebar,
    pn.Column(update_dashboard, sizing_mode='stretch_width'),
    sizing_mode='stretch_width',
).servable(title='Goss-Peccoud Interactive Dashboard')