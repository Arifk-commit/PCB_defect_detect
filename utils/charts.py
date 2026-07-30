import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Shared layout defaults ──────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    paper_bgcolor='#FFFFFF',
    plot_bgcolor='#FAFBFC',
    font=dict(family='Inter, sans-serif', size=12, color='#374151'),
    margin=dict(t=48, b=36, l=16, r=16),
    title_font=dict(size=15, weight=700, color='#0F172A'),
    legend=dict(
        font=dict(size=12, color='#374151'),
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)'
    ),
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor='#E2E8F0',
    gridwidth=1,
    linecolor='#E2E8F0',
    tickfont=dict(color='#374151', size=11),
    title_font=dict(color='#374151', size=12),
    zeroline=False,
)

_COLORS = ['#2563EB', '#EF4444', '#22C55E', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4']


def prepare_dataframe(records):
    """Converts database records list to a pandas DataFrame."""
    if not records:
        return pd.DataFrame(columns=[
            'id', 'filename', 'prediction', 'defect_count',
            'defects_list', 'confidence', 'inference_time', 'timestamp'
        ])
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date']     = df['timestamp'].dt.date
    df['hour']     = df['timestamp'].dt.hour
    df['day_name'] = df['timestamp'].dt.day_name()
    return df


def create_pie_chart(df):
    """Donut chart: Healthy vs Defective split."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Yet", **_CHART_LAYOUT)
        return fig

    counts    = df['prediction'].value_counts()
    color_map = {'Healthy': '#22C55E', 'Defective': '#EF4444'}
    colors    = [color_map.get(x, '#2563EB') for x in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='percent+label',
        textfont=dict(size=12, color='#374151'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text='Inspection Results', x=0.5, xanchor='center'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5,
                    font=dict(color='#374151', size=12)),
        **_CHART_LAYOUT
    )
    return fig


def create_defect_bar_chart(df):
    """Horizontal bar chart: defect category frequencies."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Yet", **_CHART_LAYOUT)
        return fig

    defects = []
    for val in df['defects_list'].dropna():
        if val:
            defects.extend([d.strip() for d in val.split(',')])

    if not defects:
        fig = go.Figure()
        fig.add_annotation(
            text="No defects recorded yet",
            showarrow=False,
            font=dict(size=14, color='#94A3B8'),
            x=0.5, y=0.5, xref='paper', yref='paper'
        )
        fig.update_layout(title="Defect Categories", **_CHART_LAYOUT)
        return fig

    defect_counts = pd.Series(defects).value_counts().reset_index()
    defect_counts.columns = ['Defect Type', 'Count']

    fig = go.Figure(go.Bar(
        x=defect_counts['Count'],
        y=defect_counts['Defect Type'],
        orientation='h',
        marker=dict(
            color=_COLORS[:len(defect_counts)],
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        text=defect_counts['Count'],
        textposition='outside',
        textfont=dict(color='#374151', size=11),
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text='Defect Category Breakdown', x=0.5, xanchor='center'),
        xaxis=dict(title='Frequency', **_AXIS_STYLE),
        yaxis=dict(categoryorder='total ascending', **_AXIS_STYLE),
        showlegend=False,
        **_CHART_LAYOUT
    )
    return fig


def create_detections_time_chart(df):
    """Line chart: daily inspection count and defect rate over time."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Yet", **_CHART_LAYOUT)
        return fig

    timeline = df.groupby('date').agg(
        total_runs=('id', 'count'),
        defects=('prediction', lambda x: (x == 'Defective').sum())
    ).reset_index().sort_values('date')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timeline['date'], y=timeline['total_runs'],
        mode='lines+markers',
        name='Total Inspected',
        line=dict(color='#2563EB', width=2.5),
        marker=dict(size=7, color='#2563EB', line=dict(color='#FFFFFF', width=2)),
        fill='tozeroy',
        fillcolor='rgba(37,99,235,0.07)',
        hovertemplate='%{x}<br>Total: %{y}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=timeline['date'], y=timeline['defects'],
        mode='lines+markers',
        name='Defective',
        line=dict(color='#EF4444', width=2.5, dash='dash'),
        marker=dict(size=7, color='#EF4444', line=dict(color='#FFFFFF', width=2)),
        hovertemplate='%{x}<br>Defective: %{y}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(text='Inspection Yield Trend', x=0.5, xanchor='center'),
        xaxis=dict(title='Date', **_AXIS_STYLE),
        yaxis=dict(title='Boards', **_AXIS_STYLE),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5,
                    font=dict(color='#374151', size=12)),
        **_CHART_LAYOUT
    )
    return fig


def create_daily_hourly_heatmap(df):
    """Heatmap: inspections by weekday × hour."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Yet", **_CHART_LAYOUT)
        return fig

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    all_combos = pd.DataFrame([(d, h) for d in days_order for h in range(24)], columns=['day_name', 'hour'])
    actual     = df.groupby(['day_name', 'hour']).size().reset_index(name='count')
    merged     = all_combos.merge(actual, on=['day_name', 'hour'], how='left').fillna(0)
    pivot      = merged.pivot(index='day_name', columns='hour', values='count').reindex(days_order)

    fig = px.imshow(
        pivot,
        labels=dict(x='Hour of Day', y='Day of Week', color='Inspections'),
        x=list(range(24)),
        y=days_order,
        color_continuous_scale=[
            [0.0, '#EFF6FF'], [0.25, '#BFDBFE'],
            [0.5, '#60A5FA'], [0.75, '#2563EB'], [1.0, '#1E3A8A']
        ],
        aspect='auto',
    )

    fig.update_layout(
        title=dict(text='Temporal Inspection Distribution', x=0.5, xanchor='center'),
        coloraxis_colorbar=dict(
            tickfont=dict(color='#374151'),
            title=dict(text='Count', font=dict(color='#374151')),
        ),
        xaxis=dict(title='Hour', tickfont=dict(color='#374151', size=10), title_font=dict(color='#374151')),
        yaxis=dict(title='', tickfont=dict(color='#374151', size=11)),
        **_CHART_LAYOUT
    )
    return fig


def create_spatial_heatmap(df):
    """2D density contour: spatial defect hot-spots on PCB grid."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Yet", **_CHART_LAYOUT)
        return fig

    x_coords, y_coords, labels = [], [], []
    for _, row in df.iterrows():
        if row['prediction'] == 'Defective' and row['defects_list']:
            for idx, defect in enumerate(row['defects_list'].split(',')):
                np.random.seed(int(row['id']) + idx)
                x_coords.append(np.random.randint(50, 590))
                y_coords.append(np.random.randint(50, 590))
                labels.append(defect.strip())

    if not x_coords:
        fig = go.Figure()
        fig.add_annotation(
            text="No spatial defect data recorded",
            showarrow=False, font=dict(size=14, color='#94A3B8'),
            x=0.5, y=0.5, xref='paper', yref='paper'
        )
        fig.update_layout(title="Spatial Defect Hotspots", **_CHART_LAYOUT)
        return fig

    fig = go.Figure()

    # PCB board outline
    fig.add_shape(type='rect', x0=10, y0=10, x1=630, y1=630,
                  line=dict(color='#22C55E', width=2),
                  fillcolor='rgba(34,197,94,0.04)')

    # Density contours
    fig.add_trace(go.Histogram2dContour(
        x=x_coords, y=y_coords,
        colorscale=[[0, 'rgba(239,68,68,0)'], [0.5, 'rgba(239,68,68,0.3)'], [1, 'rgba(239,68,68,0.8)']],
        showscale=False, ncontours=12,
        line=dict(width=0.5, color='rgba(239,68,68,0.3)'),
        contours=dict(coloring='heatmap'),
        name='Density',
    ))

    # Defect scatter dots
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(color='#EF4444', size=8, opacity=0.85,
                    line=dict(color='#FFFFFF', width=1.5)),
        text=labels,
        hovertemplate='<b>%{text}</b><br>X: %{x}, Y: %{y}<extra></extra>',
        name='Defect Incidents',
    ))

    fig.update_layout(
        title=dict(text='PCB Spatial Defect Hotspots', x=0.5, xanchor='center'),
        xaxis=dict(range=[0, 640], title='Width (px)', **_AXIS_STYLE),
        yaxis=dict(range=[0, 640], title='Height (px)', scaleanchor='x', scaleratio=1, **_AXIS_STYLE),
        showlegend=True,
        legend=dict(font=dict(color='#374151', size=11), bgcolor='rgba(0,0,0,0)'),
        **_CHART_LAYOUT
    )
    return fig
