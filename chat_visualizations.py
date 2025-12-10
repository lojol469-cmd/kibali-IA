"""
Module de visualisations Plotly pour Kibali AI
Crée des graphiques interactifs modernes et futuristes
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import pandas as pd

# Palette de couleurs futuriste Kibali
KIBALI_COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent1': '#00ff88',
    'accent2': '#00d4ff',
    'dark': '#2d3748',
    'light': '#f7fafc'
}

def create_futuristic_theme():
    """Retourne un template Plotly personnalisé futuriste"""
    return {
        'layout': {
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'rgba(247, 250, 252, 0.5)',
            'font': {
                'family': 'Arial, sans-serif',
                'size': 14,
                'color': KIBALI_COLORS['dark']
            },
            'title': {
                'font': {
                    'size': 20,
                    'color': KIBALI_COLORS['primary'],
                    'family': 'Arial, sans-serif'
                }
            },
            'xaxis': {
                'gridcolor': 'rgba(102, 126, 234, 0.1)',
                'showline': True,
                'linewidth': 2,
                'linecolor': KIBALI_COLORS['accent1']
            },
            'yaxis': {
                'gridcolor': 'rgba(102, 126, 234, 0.1)',
                'showline': True,
                'linewidth': 2,
                'linecolor': KIBALI_COLORS['accent2']
            }
        }
    }

def create_stats_card(title: str, value: str, subtitle: str = "", icon: str = "📊"):
    """Crée une carte statistique HTML futuriste"""
    return f"""
    <div style="
        background: linear-gradient(135deg, {KIBALI_COLORS['primary']} 0%, {KIBALI_COLORS['secondary']} 100%);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        color: white;
        margin: 10px 0;
        animation: fadeInUp 0.5s ease-out;
    ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <span style="font-size: 32px;">{icon}</span>
            <h3 style="margin: 0; font-size: 16px; opacity: 0.9;">{title}</h3>
        </div>
        <div style="font-size: 36px; font-weight: 900; margin: 8px 0;">{value}</div>
        {f'<div style="font-size: 14px; opacity: 0.8;">{subtitle}</div>' if subtitle else ''}
    </div>
    """

def create_bar_chart(data: Dict[str, float], title: str, x_label: str, y_label: str):
    """Crée un graphique en barres futuriste avec Plotly"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(data.keys()),
            y=list(data.values()),
            marker=dict(
                color=list(data.values()),
                colorscale=[[0, KIBALI_COLORS['accent1']], [1, KIBALI_COLORS['primary']]],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=KIBALI_COLORS['primary'])),
        xaxis_title=x_label,
        yaxis_title=y_label,
        paper_bgcolor='white',
        plot_bgcolor='rgba(247, 250, 252, 0.5)',
        font=dict(family='Arial', size=14, color=KIBALI_COLORS['dark']),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    return fig

def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Crée un graphique linéaire futuriste avec Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines+markers',
        line=dict(color=KIBALI_COLORS['primary'], width=3),
        marker=dict(
            size=8,
            color=KIBALI_COLORS['accent1'],
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>%{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=KIBALI_COLORS['primary'])),
        paper_bgcolor='white',
        plot_bgcolor='rgba(247, 250, 252, 0.5)',
        font=dict(family='Arial', size=14, color=KIBALI_COLORS['dark']),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        xaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
        yaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)')
    )
    
    return fig

def create_pie_chart(data: Dict[str, float], title: str):
    """Crée un graphique circulaire futuriste avec Plotly"""
    colors = [KIBALI_COLORS['primary'], KIBALI_COLORS['accent1'], 
              KIBALI_COLORS['accent2'], KIBALI_COLORS['secondary']]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=list(data.keys()),
            values=list(data.values()),
            marker=dict(
                colors=colors[:len(data)],
                line=dict(color='white', width=3)
            ),
            hovertemplate='<b>%{label}</b><br>%{value} (%{percent})<extra></extra>',
            textfont=dict(size=16, color='white'),
            hole=0.4
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=KIBALI_COLORS['primary'])),
        paper_bgcolor='white',
        font=dict(family='Arial', size=14, color=KIBALI_COLORS['dark']),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: str = None):
    """Crée un nuage de points futuriste avec Plotly"""
    if color_col and color_col in df.columns:
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col,
            color_continuous_scale=[[0, KIBALI_COLORS['accent1']], [1, KIBALI_COLORS['primary']]]
        )
    else:
        fig = go.Figure(data=[
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='markers',
                marker=dict(
                    size=10,
                    color=KIBALI_COLORS['primary'],
                    line=dict(color='white', width=2),
                    opacity=0.8
                )
            )
        ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=KIBALI_COLORS['primary'])),
        paper_bgcolor='white',
        plot_bgcolor='rgba(247, 250, 252, 0.5)',
        font=dict(family='Arial', size=14, color=KIBALI_COLORS['dark']),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        xaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
        yaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)')
    )
    
    return fig

def create_heatmap(data: List[List[float]], x_labels: List[str], y_labels: List[str], title: str):
    """Crée une heatmap futuriste avec Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, KIBALI_COLORS['accent1']], [0.5, KIBALI_COLORS['accent2']], [1, KIBALI_COLORS['primary']]],
        hovertemplate='X: %{x}<br>Y: %{y}<br>Valeur: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=KIBALI_COLORS['primary'])),
        paper_bgcolor='white',
        font=dict(family='Arial', size=14, color=KIBALI_COLORS['dark']),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    return fig
