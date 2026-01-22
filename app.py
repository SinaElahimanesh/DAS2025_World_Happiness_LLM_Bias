"""
Main web application for World Happiness Analysis
Beautiful interactive dashboard with world map and comprehensive visualizations
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data_loader import load_data, clean_data, add_regions, add_income_levels
from driver_analysis import get_driver_summary, analyze_drivers_by_group
from trend_analysis import (get_global_trends, get_country_trends, 
                           get_regional_trends, get_income_trends, 
                           calculate_trend_statistics, get_volatility_analysis)
from group_comparison import (compare_regions, compare_income_levels, 
                             get_happiness_gap_analysis, get_factor_differences_by_group)
from llm_audit import compare_llm_vs_human, analyze_llm_bias, calculate_correlation

# Import real LLM audit data loader
import sys
import os
HAS_REAL_LLM_DATA = False
get_latest_llm_comparison = None
get_latest_bias_summary = None
get_latest_bias_data = None
get_latest_significant_findings = None

llm_audit_path = os.path.join(os.path.dirname(__file__), 'llm_audit_data')
if os.path.exists(llm_audit_path):
    sys.path.insert(0, llm_audit_path)
    try:
        from load_llm_results import (get_latest_llm_comparison, get_latest_bias_summary, 
                                     get_latest_bias_data, get_latest_significant_findings,
                                     compute_bias_from_comparison, compute_simplified_significance_tests,
                                     prepare_llm_comparison_for_web, prepare_bias_summary_for_web)
        HAS_REAL_LLM_DATA = True
        print("✓ Real LLM audit data loader imported successfully")
    except ImportError as e:
        HAS_REAL_LLM_DATA = False
        print(f"Warning: Could not load real LLM audit data: {e}. Using mock data.")
    except Exception as e:
        HAS_REAL_LLM_DATA = False
        print(f"Warning: Error loading LLM audit module: {e}. Using mock data.")
else:
    print(f"Warning: LLM audit data directory not found at {llm_audit_path}. Using mock data.")

# Load and prepare data
print("Loading data...")
df_raw = load_data()
df = clean_data(df_raw)
df = add_regions(df)
df = add_income_levels(df)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "How Happy is the World? - Saarland University"

# Enhanced CSS for beautiful styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
                background-attachment: fixed;
                color: #333;
            }
            .main-container {
                max-width: 1800px;
                margin: 0 auto;
                padding: 30px 20px;
                min-height: 100vh;
            }
            .header-card {
                background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.95) 100%);
                padding: 40px;
                border-radius: 20px;
                margin-bottom: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                backdrop-filter: blur(10px);
            }
            .header-card h1 {
                margin: 0 0 10px 0;
                color: #1e3c72;
                font-size: 3em;
                font-weight: 700;
                letter-spacing: -1px;
            }
            .header-card .subtitle {
                color: #555;
                font-size: 1.3em;
                margin: 0 0 5px 0;
                font-weight: 400;
            }
            .header-card .university {
                color: #888;
                font-size: 0.95em;
                margin: 0;
            }
            .content-card {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 20px;
                padding: 40px;
                margin-bottom: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.15);
                backdrop-filter: blur(10px);
            }
            .section-title {
                color: #1e3c72;
                margin: 0 0 30px 0;
                font-size: 2.2em;
                font-weight: 600;
                border-bottom: 3px solid #2a5298;
                padding-bottom: 15px;
            }
            .metric-card {
                display: inline-block;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-radius: 15px;
                margin-right: 20px;
                margin-bottom: 20px;
                min-width: 180px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }
            .metric-label {
                font-size: 0.9em;
                color: #666;
                margin: 0 0 8px 0;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .metric-value {
                color: #1e3c72;
                margin: 0;
                font-size: 2em;
                font-weight: 700;
            }
            .tabs-container {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 15px;
                padding: 10px;
                margin-bottom: 30px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1("How Happy is the World?", style={'textAlign': 'center', 'color': 'white', 'margin': '20px 0', 'fontSize': '3.5em', 'fontWeight': '700', 'letterSpacing': '-1px', 'textShadow': '2px 2px 4px rgba(0,0,0,0.2)'}),
            html.P("Aligning Decadal Trends with AI Perceptions", 
                   style={'textAlign': 'center', 'color': 'white', 'fontSize': '1.5em', 'marginBottom': '15px', 'fontWeight': '300', 'textShadow': '1px 1px 2px rgba(0,0,0,0.2)'}),
            html.P("Saarland University - Data and Society Seminar (Winter 2025)", 
                   style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.95)', 'fontSize': '1.1em', 'margin': '0', 'fontWeight': '300'}),
        ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
                  'padding': '50px 40px', 'borderRadius': '20px', 'marginBottom': '30px', 
                  'boxShadow': '0 20px 60px rgba(0,0,0,0.3)', 'backdropFilter': 'blur(10px)'}),
        
        html.Div([
            dcc.Tabs(id="main-tabs", value='overview', children=[
                dcc.Tab(label='World Map', value='map', style={'fontSize': '15px', 'fontWeight': '500', 'padding': '15px'}),
                dcc.Tab(label='Driver Analysis', value='drivers', style={'fontSize': '15px', 'fontWeight': '500', 'padding': '15px'}),
                dcc.Tab(label='Trends', value='trends', style={'fontSize': '15px', 'fontWeight': '500', 'padding': '15px'}),
                dcc.Tab(label='Group Comparisons', value='groups', style={'fontSize': '15px', 'fontWeight': '500', 'padding': '15px'}),
                dcc.Tab(label='LLM Audit', value='llm', style={'fontSize': '15px', 'fontWeight': '500', 'padding': '15px'}),
                dcc.Tab(label='Overview', value='overview', style={'fontSize': '15px', 'fontWeight': '500', 'padding': '15px'}),
            ], style={'fontSize': '16px', 'fontWeight': 'bold'})
        ], style={'background': 'rgba(255, 255, 255, 0.98)', 'borderRadius': '15px', 'padding': '10px', 'marginBottom': '30px', 'boxShadow': '0 8px 30px rgba(0,0,0,0.15)'}),
        
        html.Div(id='tab-content'),
    
    # Modal for country details
    dcc.Store(id='clicked-country-store', data=None),
    dcc.Store(id='selected-year-store', data=df['Year'].max()),
    html.Div([
        html.Div([
            html.Div([
                html.H3(id='modal-country-name', style={'margin': '0 0 20px 0', 'color': '#1e3c72', 'fontSize': '1.8em'}),
                html.Button('×', id='close-modal', n_clicks=0, 
                           style={'position': 'absolute', 'top': '15px', 'right': '15px', 
                                 'background': 'none', 'border': 'none', 'fontSize': '2em', 
                                 'color': '#666', 'cursor': 'pointer', 'padding': '0', 'width': '30px', 'height': '30px'}),
                html.Div(id='modal-country-details', style={'marginTop': '20px'})
            ], style={'position': 'relative', 'padding': '30px'})
        ], style={'background': 'white', 'borderRadius': '15px', 'maxWidth': '900px', 'width': '90%',
                 'maxHeight': '85vh', 'overflowY': 'auto', 'boxShadow': '0 10px 40px rgba(0,0,0,0.3)'})
    ], id='modal-container', 
       style={'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 
             'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)', 'zIndex': '1000', 
             'justifyContent': 'center', 'alignItems': 'center'})
    ], style={'maxWidth': '1800px', 'margin': '0 auto', 'padding': '30px 20px', 'minHeight': '100vh'})
], style={'minHeight': '100vh'})


@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_content(tab):
    try:
        if tab == 'map':
            return render_map_tab()
        elif tab == 'drivers':
            return render_drivers_tab()
        elif tab == 'trends':
            return render_trends_tab()
        elif tab == 'groups':
            return render_groups_tab()
        elif tab == 'llm':
            return render_llm_tab()
        elif tab == 'overview':
            return render_overview_tab()
        else:
            return html.Div("Error: Unknown tab", style={'padding': '20px', 'color': 'red'})
    except Exception as e:
        return html.Div([
            html.H3("Error loading content", style={'color': 'red'}),
            html.P(str(e), style={'color': '#666', 'fontSize': '12px'})
        ], style={'padding': '20px'})


def render_map_tab():
    """Enhanced world map visualization with interactivity"""
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    
    # Create enhanced choropleth map with better styling
    fig = go.Figure(data=go.Choropleth(
        locations=df_latest['country'],
        z=df_latest['happiness_score'],
        locationmode='country names',
        colorscale='RdYlGn',
        reversescale=False,
        text=df_latest.apply(lambda row: f"<b>{row['country']}</b><br>" +
                            f"Happiness Score: {row['happiness_score']:.2f}<br>" +
                            f"GDP: {row['gdp']:.2f}<br>" +
                            f"Social Support: {row['social_support']:.2f}<br>" +
                            f"Life Expectancy: {row['life_expectancy']:.2f}<br>" +
                            f"Freedom: {row['freedom']:.2f}<br>" +
                            f"Generosity: {row['generosity']:.2f}<br>" +
                            f"Corruption: {row['corruption']:.2f}", axis=1),
        hoverinfo='text',
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar=dict(
            title=dict(text="Happiness Score", font=dict(size=14, color='#333')),
            tickfont=dict(size=12, color='#333'),
            len=0.7,
            y=0.5,
            thickness=20,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#ccc',
            borderwidth=1
        )
    ))
    
    fig.update_geos(
        projection_type='natural earth',
        showcoastlines=True,
        coastlinecolor='#ffffff',
        showland=True,
        landcolor='#f5f5f5',
        showocean=True,
        oceancolor='#e8f4f8',
        showlakes=True,
        lakecolor='#e8f4f8',
        showrivers=True,
        rivercolor='#e8f4f8',
        bgcolor='rgba(0,0,0,0)',
        framecolor='#ffffff',
        framewidth=2,
        resolution=110
    )
    
    fig.update_layout(
        title=dict(
            text=f'World Happiness Map ({latest_year})',
            font=dict(size=28, color='#1e3c72', family='Inter'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        height=800,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            showframe=False
        ),
        font=dict(family='Inter', size=12, color='#333'),
        clickmode='event'
    )
    
    # Add year selector
    years = sorted(df['Year'].unique())
    year_options = [{'label': str(year), 'value': year} for year in years]
    
    return html.Div([
        html.Div([
            html.H2("World Happiness Map", style={'color': '#1e3c72', 'margin': '0 0 30px 0', 'fontSize': '2.2em', 'fontWeight': '600', 'borderBottom': '3px solid #2a5298', 'paddingBottom': '15px'}),
            html.Div([
                html.Label("Select Year:", style={'fontSize': '16px', 'fontWeight': '500', 'marginRight': '15px', 'color': '#555'}),
                dcc.Dropdown(
                    id='year-selector',
                    options=year_options,
                    value=latest_year,
                    clearable=False,
                    style={'width': '200px', 'display': 'inline-block'}
                )
            ], style={'marginBottom': '30px', 'display': 'flex', 'alignItems': 'center'}),
            dcc.Graph(id='world-map', figure=fig, style={'height': '800px', 'cursor': 'pointer'}, config={'displayModeBar': False}),
            html.P("Interactive world map showing happiness scores by country. Hover over countries to see quick metrics. Click on countries to view detailed information. Use the year selector to view different time periods.",
                   style={'marginTop': '25px', 'color': '#666', 'fontSize': '14px', 'lineHeight': '1.6', 'fontStyle': 'italic'})
        ], style={'background': 'rgba(255, 255, 255, 0.98)', 'borderRadius': '20px', 'padding': '40px', 'marginBottom': '30px', 'boxShadow': '0 10px 40px rgba(0,0,0,0.15)', 'backdropFilter': 'blur(10px)'})
    ])


@app.callback(
    [Output('world-map', 'figure'),
     Output('selected-year-store', 'data')],
    Input('year-selector', 'value')
)
def update_map(selected_year):
    """Update map based on selected year"""
    try:
        if selected_year is None:
            selected_year = df['Year'].max()
        df_year = df[df['Year'] == selected_year].copy()
        
        if len(df_year) == 0:
            selected_year = df['Year'].max()
            df_year = df[df['Year'] == selected_year].copy()
        
        fig = go.Figure(data=go.Choropleth(
            locations=df_year['country'],
            z=df_year['happiness_score'],
            locationmode='country names',
            colorscale='RdYlGn',
            reversescale=False,
            text=df_year.apply(lambda row: f"<b>{row['country']}</b><br>" +
                                f"Happiness Score: {row['happiness_score']:.2f}<br>" +
                                f"GDP: {row['gdp']:.2f}<br>" +
                                f"Social Support: {row['social_support']:.2f}<br>" +
                                f"Life Expectancy: {row['life_expectancy']:.2f}<br>" +
                                f"Freedom: {row['freedom']:.2f}<br>" +
                                f"Generosity: {row['generosity']:.2f}<br>" +
                                f"Corruption: {row['corruption']:.2f}", axis=1),
            hoverinfo='text',
            marker_line_color='white',
            marker_line_width=0.5,
            colorbar=dict(
                title=dict(text="Happiness Score", font=dict(size=14, color='#333')),
                tickfont=dict(size=12, color='#333'),
                len=0.7,
                y=0.5,
                thickness=20,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#ccc',
                borderwidth=1
            )
        ))
        
        fig.update_geos(
            projection_type='natural earth',
            showcoastlines=True,
            coastlinecolor='#ffffff',
            showland=True,
            landcolor='#f5f5f5',
            showocean=True,
            oceancolor='#e8f4f8',
            showlakes=True,
            lakecolor='#e8f4f8',
            showrivers=True,
            rivercolor='#e8f4f8',
            bgcolor='rgba(0,0,0,0)',
            framecolor='#ffffff',
            framewidth=2,
            resolution=110
        )
        
        fig.update_layout(
            title=dict(
                text=f'World Happiness Map ({selected_year})',
                font=dict(size=28, color='#1e3c72', family='Inter'),
                x=0.5,
                xanchor='center',
                y=0.98,
                yanchor='top'
            ),
            height=800,
            margin=dict(l=0, r=0, t=80, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False),
            font=dict(family='Inter', size=12, color='#333'),
            clickmode='event'
        )
        
        return fig, selected_year
    except Exception as e:
        # Fallback to latest year if error occurs
        latest_year = df['Year'].max()
        df_year = df[df['Year'] == latest_year].copy()
        fig = go.Figure()
        fig.add_annotation(text=f"Error loading map: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig, df['Year'].max()


def render_drivers_tab():
    """Driver analysis visualizations"""
    driver_summary = get_driver_summary(df)
    importance = driver_summary['importance']
    
    # Enhanced bar chart
    fig1 = go.Figure(data=[
        go.Bar(
            x=importance['feature'],
            y=importance['abs_coefficient'],
            marker=dict(
                color=importance['coefficient'],
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title=dict(text="Coefficient", font=dict(size=12)), tickfont=dict(size=10))
            ),
            text=[f'{val:.3f}' for val in importance['abs_coefficient']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Absolute Coefficient: %{y:.3f}<extra></extra>'
        )
    ])
    fig1.update_layout(
        title=dict(text='Factor Importance in Happiness (Weighted Linear Regression)', 
                  font=dict(size=20, color='#1e3c72')),
        height=550,
        xaxis=dict(title=dict(text='Factor', font=dict(size=14)), tickangle=-45, 
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Absolute Coefficient', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    # Regional driver analysis with normalized coefficients
    regional_drivers = analyze_drivers_by_group(df, 'region')
    
    regions_list = list(regional_drivers.keys())[:8]
    fig2_data = []
    for region in regions_list:
        region_importance = regional_drivers[region]
        for _, row in region_importance.iterrows():
            fig2_data.append({
                'region': region,
                'feature': row['feature'],
                'coefficient': row['coefficient']
            })
    
    if fig2_data:
        fig2_df = pd.DataFrame(fig2_data)
        fig2 = px.bar(
            fig2_df,
            x='feature',
            y='coefficient',
            color='region',
            barmode='group',
            title='Normalized Factor Importance by Region',
            labels={'coefficient': 'Normalized Coefficient'}
        )
        fig2.update_layout(
            height=600,
            xaxis=dict(tickangle=-45, title='Factor', 
                      showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
            yaxis=dict(title='Normalized Coefficient',
                      showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12),
            title_font=dict(size=18, color='#1e3c72')
        )
    else:
        fig2 = go.Figure()
        fig2.add_annotation(text="Insufficient data for regional comparison", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
    
    return html.Div([
        html.Div([
            html.H2("Driver Analysis", style={'color': '#1e3c72', 'margin': '0 0 30px 0', 'fontSize': '2.2em', 'fontWeight': '600', 'borderBottom': '3px solid #2a5298', 'paddingBottom': '15px'}),
            html.Div([
                html.Div([
                    html.H3("Overall Factor Importance", style={'fontSize': '20px', 'marginBottom': '15px', 'color': '#1e3c72', 'fontWeight': '600'}),
                    html.Div([
                        html.Div([
                            html.P("R-squared", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                            html.P(f"{driver_summary['r_squared']:.3f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '1.5em', 'fontWeight': '700'})
                        ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'minWidth': '180px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),
                        html.Div([
                            html.P("Top Driver", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                            html.P(driver_summary['top_driver'].replace('_', ' ').title(), style={'color': '#1e3c72', 'margin': '0', 'fontSize': '1.2em', 'fontWeight': '700'})
                        ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'minWidth': '180px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
                    ], style={'marginBottom': '30px'}),
                    dcc.Graph(figure=fig1)
                ], style={'marginBottom': '40px'}),
                html.Div([
                    html.H3("Regional Differences in Factor Importance (Normalized)", 
                           style={'fontSize': '20px', 'marginBottom': '15px', 'color': '#1e3c72', 'fontWeight': '600'}),
                    html.P("Coefficients are normalized to allow fair comparison across regions. Values represent relative importance within each region.",
                          style={'fontSize': '14px', 'color': '#666', 'marginBottom': '20px', 'fontStyle': 'italic'}),
                    dcc.Graph(figure=fig2)
                ])
            ])
        ], style={'background': 'rgba(255, 255, 255, 0.98)', 'borderRadius': '20px', 'padding': '40px', 'marginBottom': '30px', 'boxShadow': '0 10px 40px rgba(0,0,0,0.15)', 'backdropFilter': 'blur(10px)'})
    ])


def render_trends_tab():
    """Trend analysis visualizations"""
    global_trends = get_global_trends(df)
    regional_trends = get_regional_trends(df)
    income_trends = get_income_trends(df)
    country_trends, top_countries = get_country_trends(df, 15)
    trend_stats = calculate_trend_statistics(df)
    
    # Enhanced visualizations
    fig1 = px.line(global_trends, x='Year', y='average_happiness', markers=True)
    fig1.update_traces(line=dict(width=4, color='#1e3c72'), marker=dict(size=10, color='#2a5298'))
    fig1.update_layout(
        title=dict(text=f'Global Average Happiness Trend ({df["Year"].min()}-{df["Year"].max()})', font=dict(size=20, color='#1e3c72')),
        height=450,
        xaxis=dict(title=dict(text='Year', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Average Happiness Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    fig2 = px.line(regional_trends, x='Year', y='happiness_score', color='region', markers=True)
    fig2.update_layout(
        title=dict(text='Happiness Trends by Region', font=dict(size=20, color='#1e3c72')),
        height=550,
        xaxis=dict(title=dict(text='Year', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Happiness Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    fig3 = px.line(income_trends, x='Year', y='happiness_score', color='income_level', markers=True)
    fig3.update_layout(
        title=dict(text='Happiness Trends by Income Level', font=dict(size=20, color='#1e3c72')),
        height=550,
        xaxis=dict(title=dict(text='Year', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Happiness Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    fig4 = px.line(country_trends, x='Year', y='happiness_score', color='country', markers=True)
    fig4.update_layout(
        title=dict(text='Happiness Trends: Top 15 Countries', font=dict(size=20, color='#1e3c72')),
        height=650,
        xaxis=dict(title=dict(text='Year', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Happiness Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=11)
    )
    
    fig5 = px.bar(trend_stats['biggest_improvers'].head(10), x='country', y='change', 
                  color='change', color_continuous_scale='Greens')
    fig5.update_layout(
        title=dict(text=f'Biggest Improvements ({df["Year"].min()}-{df["Year"].max()})', font=dict(size=20, color='#1e3c72')),
        height=450,
        xaxis=dict(title=dict(text='Country', font=dict(size=14)), tickangle=-45,
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Change in Happiness Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    return html.Div([
        html.Div([
            html.H2(f"Trend Analysis ({df['Year'].min()}-{df['Year'].max()})", style={'color': '#1e3c72', 'margin': '0 0 30px 0', 'fontSize': '2.2em', 'fontWeight': '600', 'borderBottom': '3px solid #2a5298', 'paddingBottom': '15px'}),
            html.Div([
                html.Div([
                    html.H3("Key Statistics", style={'fontSize': '20px', 'marginBottom': '20px', 'color': '#1e3c72', 'fontWeight': '600'}),
                    html.Div([
                        html.Div([
                            html.P("Global Change", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                            html.P(f"{trend_stats['global_change']:.2f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'})
                        ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'minWidth': '180px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),
                        html.Div([
                            html.P("Percentage Change", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                            html.P(f"{trend_stats['global_change_pct']:.2f}%", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'})
                        ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'minWidth': '180px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
                    ], style={'marginBottom': '40px'})
                ]),
                html.Div([dcc.Graph(figure=fig1)], style={'marginBottom': '40px'}),
                html.Div([dcc.Graph(figure=fig2)], style={'marginBottom': '40px'}),
                html.Div([dcc.Graph(figure=fig3)], style={'marginBottom': '40px'}),
                html.Div([dcc.Graph(figure=fig4)], style={'marginBottom': '40px'}),
                html.Div([dcc.Graph(figure=fig5)])
            ])
        ], style={'background': 'rgba(255, 255, 255, 0.98)', 'borderRadius': '20px', 'padding': '40px', 'marginBottom': '30px', 'boxShadow': '0 10px 40px rgba(0,0,0,0.15)', 'backdropFilter': 'blur(10px)'})
    ])


def render_groups_tab():
    """Group comparison visualizations"""
    region_stats = compare_regions(df)
    income_stats = compare_income_levels(df)
    gap_analysis = get_happiness_gap_analysis(df)
    factor_diff_region = get_factor_differences_by_group(df, 'region')
    factor_diff_income = get_factor_differences_by_group(df, 'income_level')
    
    fig1 = px.bar(region_stats, x='region', y='avg_happiness', color='avg_happiness', 
                  color_continuous_scale='RdYlGn')
    fig1.update_layout(
        title=dict(text='Average Happiness by Region', font=dict(size=20, color='#1e3c72')),
        height=550,
        xaxis=dict(title=dict(text='Region', font=dict(size=14)), tickangle=-45,
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Average Happiness Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    fig2 = px.bar(income_stats, x='income_level', y='avg_happiness', color='avg_happiness', 
                  color_continuous_scale='RdYlGn')
    fig2.update_layout(
        title=dict(text='Average Happiness by Income Level', font=dict(size=20, color='#1e3c72')),
        height=550,
        xaxis=dict(title=dict(text='Income Level', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Average Happiness Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    fig3 = px.bar(factor_diff_region, x='factor', y='avg_value', color='region', barmode='group')
    fig3.update_layout(
        title=dict(text='Factor Values by Region', font=dict(size=20, color='#1e3c72')),
        height=550,
        xaxis=dict(title=dict(text='Factor', font=dict(size=14)), tickangle=-45,
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Average Factor Value', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    fig4 = px.bar(factor_diff_income, x='factor', y='avg_value', color='income_level', barmode='group')
    fig4.update_layout(
        title=dict(text='Factor Values by Income Level', font=dict(size=20, color='#1e3c72')),
        height=550,
        xaxis=dict(title=dict(text='Factor', font=dict(size=14)), tickangle=-45,
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='Average Factor Value', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    return html.Div([
        html.Div([
            html.H2("Group Comparisons", style={'color': '#1e3c72', 'margin': '0 0 30px 0', 'fontSize': '2.2em', 'fontWeight': '600', 'borderBottom': '3px solid #2a5298', 'paddingBottom': '15px'}),
            html.Div([
                html.H3("Gap Analysis", style={'fontSize': '20px', 'marginBottom': '20px', 'color': '#1e3c72', 'fontWeight': '600'}),
                html.Div([
                    html.Div([
                        html.P("Region Gap", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                        html.P(f"{gap_analysis['region_gap']:.2f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'}),
                        html.P(f"Highest: {gap_analysis['highest_region']}", style={'fontSize': '12px', 'color': '#999', 'margin': '5px 0 0 0'}),
                        html.P(f"Lowest: {gap_analysis['lowest_region']}", style={'fontSize': '12px', 'color': '#999', 'margin': '0'})
                    ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'minWidth': '200px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),
                    html.Div([
                        html.P("Income Gap", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                        html.P(f"{gap_analysis['income_gap']:.2f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'}),
                        html.P(f"Highest: {gap_analysis['highest_income']}", style={'fontSize': '12px', 'color': '#999', 'margin': '5px 0 0 0'}),
                        html.P(f"Lowest: {gap_analysis['lowest_income']}", style={'fontSize': '12px', 'color': '#999', 'margin': '0'})
                    ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'minWidth': '200px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
                ], style={'marginBottom': '40px'}),
                html.Div([dcc.Graph(figure=fig1)], style={'marginBottom': '40px'}),
                html.Div([dcc.Graph(figure=fig2)], style={'marginBottom': '40px'}),
                html.Div([dcc.Graph(figure=fig3)], style={'marginBottom': '40px'}),
                html.Div([dcc.Graph(figure=fig4)])
            ])
        ], style={'background': 'rgba(255, 255, 255, 0.98)', 'borderRadius': '20px', 'padding': '40px', 'marginBottom': '30px', 'boxShadow': '0 10px 40px rgba(0,0,0,0.15)', 'backdropFilter': 'blur(10px)'})
    ])


def render_llm_tab():
    """LLM audit visualizations using REAL data from LLM audit results"""
    
    # Try to load real LLM audit data
    if HAS_REAL_LLM_DATA:
        try:
            comparison_df = get_latest_llm_comparison()
            bias_summary_df = get_latest_bias_summary()
            bias_data_df = get_latest_bias_data()
            significant_df = get_latest_significant_findings()
            
            if comparison_df is not None:
                return render_llm_tab_real_data(comparison_df, bias_summary_df, bias_data_df, significant_df)
        except Exception as e:
            print(f"Error loading real LLM data: {e}")
            # Fall through to mock data
    
    # Fallback to mock data if real data not available
    comparison = compare_llm_vs_human(df, use_mock=True)
    bias_analysis = analyze_llm_bias(comparison)
    correlation = calculate_correlation(comparison)
    
    fig1 = px.scatter(comparison, x='happiness_score', y='llm_score', color='region', 
                     size='abs_difference', hover_name='country')
    fig1.add_trace(go.Scatter(
        x=[comparison['happiness_score'].min(), comparison['happiness_score'].max()],
        y=[comparison['happiness_score'].min(), comparison['happiness_score'].max()],
        mode='lines',
        name='Perfect Agreement',
        line=dict(dash='dash', color='gray', width=2)
    ))
    fig1.update_layout(
        title=dict(text=f'LLM vs Human-Reported Scores (Correlation: {correlation:.3f})', 
                  font=dict(size=20, color='#1e3c72')),
        height=650,
        xaxis=dict(title=dict(text='Human-Reported Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='LLM-Generated Score', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12)
    )
    
    return html.Div([
        html.Div([
            html.H2("LLM Audit", style={'color': '#1e3c72', 'margin': '0 0 30px 0', 'fontSize': '2.2em', 'fontWeight': '600', 'borderBottom': '3px solid #2a5298', 'paddingBottom': '15px'}),
            html.Div([
                html.P("Note: Using mock data. Real LLM audit data not found.", 
                      style={'color': '#ff6b6b', 'fontSize': '14px', 'marginBottom': '20px', 'padding': '10px', 'background': '#fff3cd', 'borderRadius': '5px'}),
                dcc.Graph(figure=fig1)
            ])
        ], style={'background': 'rgba(255, 255, 255, 0.98)', 'borderRadius': '20px', 'padding': '40px', 'marginBottom': '30px', 'boxShadow': '0 10px 40px rgba(0,0,0,0.15)', 'backdropFilter': 'blur(10px)'})
    ])


def render_llm_tab_real_data(comparison_df, bias_summary_df, bias_data_df, significant_df):
    """Render LLM audit tab with REAL data from actual LLM audit results"""
    
    # Calculate overall statistics
    overall_bias = comparison_df['diff_overall_happiness'].mean()
    overall_correlation = comparison_df['llm_overall_happiness'].corr(comparison_df['real_overall_happiness'])
    
    # Get approach information if available
    approaches_used = []
    if 'approach' in comparison_df.columns:
        approaches_list = comparison_df['approach'].unique().tolist()
        approach_names = {
            'initial': 'Initial Approach',
            'few_shot': 'Few-Shot Approach',
            'single_question': 'Single Question Gallup Approach'
        }
        approaches_used = [approach_names.get(a, a) for a in approaches_list]
    else:
        approaches_used = ['All three approaches']
    
    # Prepare data for storage (convert NaN to None for JSON serialization)
    comparison_dict = comparison_df.replace({np.nan: None}).to_dict('records')
    bias_summary_dict = bias_summary_df.replace({np.nan: None}).to_dict('records') if bias_summary_df is not None and len(bias_summary_df) > 0 else None
    bias_data_dict = bias_data_df.replace({np.nan: None}).to_dict('records') if bias_data_df is not None and len(bias_data_df) > 0 else None
    significant_dict = significant_df.replace({np.nan: None}).to_dict('records') if significant_df is not None and len(significant_df) > 0 else None
    
    # Create sub-tabs for different views
    llm_subtabs = dcc.Tabs(id='llm-subtabs', value='overview', children=[
        dcc.Tab(label='Overview', value='overview', style={'fontSize': '14px', 'fontWeight': '500', 'padding': '10px'}),
        dcc.Tab(label='Key Findings', value='findings', style={'fontSize': '14px', 'fontWeight': '500', 'padding': '10px'}),
        dcc.Tab(label='Bias by Groups', value='groups', style={'fontSize': '14px', 'fontWeight': '500', 'padding': '10px'}),
        dcc.Tab(label='Statistical Significance', value='significance', style={'fontSize': '14px', 'fontWeight': '500', 'padding': '10px'}),
    ], style={'marginBottom': '30px'})
    
    # Initial content (overview) - will be filtered by callback
    initial_content = render_llm_overview(comparison_df, bias_summary_df)
    
    return html.Div([
        html.Div([
            html.H2("LLM Audit - Real Data Analysis", style={'color': '#1e3c72', 'margin': '0 0 10px 0', 'fontSize': '2.2em', 'fontWeight': '600', 'borderBottom': '3px solid #2a5298', 'paddingBottom': '15px'}),
            html.Div([
                html.Label("Filter by Approach:", 
                          style={'fontSize': '14px', 'fontWeight': '600', 'color': '#1e3c72', 'marginRight': '10px', 'display': 'inline-block'}),
                dcc.Dropdown(
                    id='llm-approach-filter',
                    options=[
                        {'label': 'All Approaches', 'value': 'all'},
                        {'label': 'Initial Approach', 'value': 'initial'},
                        {'label': 'Few-Shot Approach', 'value': 'few_shot'},
                        {'label': 'Single Question Gallup Approach', 'value': 'single_question'}
                    ],
                    value='all',
                    clearable=False,
                    style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}
                ),
                html.Span(id='llm-filter-count', 
                         style={'fontSize': '13px', 'color': '#666', 'fontStyle': 'italic'})
            ], style={'marginBottom': '30px', 'padding': '15px', 'background': '#f8f9fa', 'borderRadius': '10px'}),
            # Store the full comparison data for filtering
            html.Div([
                dcc.Store(id='llm-comparison-data', data=comparison_dict),
                dcc.Store(id='llm-bias-summary-data', data=bias_summary_dict),
                dcc.Store(id='llm-bias-data', data=bias_data_dict),
                dcc.Store(id='llm-significant-data', data=significant_dict),
            ], style={'display': 'none'}),
            llm_subtabs,
            html.Div(id='llm-subtab-content', children=initial_content)
        ], style={'background': 'rgba(255, 255, 255, 0.98)', 'borderRadius': '20px', 'padding': '40px', 'marginBottom': '30px', 'boxShadow': '0 10px 40px rgba(0,0,0,0.15)', 'backdropFilter': 'blur(10px)'})
    ])


def render_llm_overview(comparison_df, bias_summary_df, is_filtered=False):
    """Render LLM audit overview with key statistics"""
    
    # Overall statistics
    overall_bias = comparison_df['diff_overall_happiness'].mean()
    overall_correlation = comparison_df['llm_overall_happiness'].corr(comparison_df['real_overall_happiness'])
    rmse = np.sqrt((comparison_df['diff_overall_happiness']**2).mean())
    
    # Scatter plot: LLM vs Real Overall Happiness
    # Add color by approach if multiple approaches are shown
    if 'approach' in comparison_df.columns and comparison_df['approach'].nunique() > 1:
        color_col = 'approach'
        title_suffix = ' (All Approaches)'
    else:
        color_col = None
        title_suffix = ''
    
    fig1 = px.scatter(comparison_df, x='real_overall_happiness', y='llm_overall_happiness', 
                     hover_name='country', hover_data=['diff_overall_happiness'] + (['approach'] if color_col else []),
                     color=color_col,
                     labels={'real_overall_happiness': 'Real Data (from data.xlsx)', 
                            'llm_overall_happiness': 'LLM Prediction',
                            'approach': 'Approach'},
                     title=f'LLM vs Real Data: Overall Happiness{title_suffix}')
    fig1.add_trace(go.Scatter(
        x=[comparison_df['real_overall_happiness'].min(), comparison_df['real_overall_happiness'].max()],
        y=[comparison_df['real_overall_happiness'].min(), comparison_df['real_overall_happiness'].max()],
        mode='lines',
        name='Perfect Agreement',
        line=dict(dash='dash', color='gray', width=2)
    ))
    fig1.update_layout(
        height=600,
        xaxis=dict(title=dict(text='Real Overall Happiness', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        yaxis=dict(title=dict(text='LLM Overall Happiness', font=dict(size=14)),
                  showgrid=True, gridcolor='#e8e8e8', gridwidth=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12),
        title_font=dict(size=18, color='#1e3c72')
    )
    
    # Top overestimated/underestimated countries
    # If multiple approaches exist, aggregate by country first (average bias across approaches)
    if 'approach' in comparison_df.columns and comparison_df['approach'].nunique() > 1:
        # Aggregate by country: take average bias across all approaches
        country_agg = comparison_df.groupby('country')['diff_overall_happiness'].mean().reset_index()
        top_over = country_agg.nlargest(10, 'diff_overall_happiness')
        top_under = country_agg.nsmallest(10, 'diff_overall_happiness')
    else:
        # Single approach: use data directly
        top_over = comparison_df.nlargest(10, 'diff_overall_happiness')
        top_under = comparison_df.nsmallest(10, 'diff_overall_happiness')
    
    # Ensure no country appears in both lists
    top_over_countries = set(top_over['country'].tolist())
    top_under_countries = set(top_under['country'].tolist())
    
    # Remove duplicates: if a country is in both, keep it only in the list where it has larger absolute bias
    common_countries = top_over_countries & top_under_countries
    for country in common_countries:
        over_bias = top_over[top_over['country'] == country]['diff_overall_happiness'].iloc[0]
        under_bias = top_under[top_under['country'] == country]['diff_overall_happiness'].iloc[0]
        # Keep in the list where absolute bias is larger
        if abs(over_bias) >= abs(under_bias):
            top_under = top_under[top_under['country'] != country]
        else:
            top_over = top_over[top_over['country'] != country]
    
    fig2_data = pd.concat([
        top_over[['country', 'diff_overall_happiness']].assign(type='Overestimated'),
        top_under[['country', 'diff_overall_happiness']].assign(type='Underestimated')
    ])
    
    fig2 = px.bar(fig2_data, x='country', y='diff_overall_happiness', color='type',
                  color_discrete_map={'Overestimated': '#ff6b6b', 'Underestimated': '#4ecdc4'},
                  labels={'diff_overall_happiness': 'Bias (LLM - Real)'})
    fig2.update_layout(
        title='Top 10 Countries with Largest Bias',
        height=500,
        xaxis=dict(title='Country', tickangle=-45, showgrid=True, gridcolor='#e8e8e8'),
        yaxis=dict(title='Bias', showgrid=True, gridcolor='#e8e8e8'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12),
        title_font=dict(size=18, color='#1e3c72')
    )
    
    # Bias by metric
    metrics = ['diff_overall_happiness', 'diff_gdp', 'diff_social_support', 
               'diff_health', 'diff_freedom', 'diff_generosity', 'diff_corruption']
    metric_names = ['Overall Happiness', 'GDP', 'Social Support', 'Health', 
                   'Freedom', 'Generosity', 'Corruption']
    metric_biases = [comparison_df[m].mean() for m in metrics]
    
    fig3 = px.bar(x=metric_names, y=metric_biases, 
                  labels={'x': 'Metric', 'y': 'Mean Bias (LLM - Real)'},
                  title='Mean Bias by Metric')
    fig3.update_traces(marker_color=['#ff6b6b' if b > 0 else '#4ecdc4' for b in metric_biases])
    fig3.update_layout(
        height=500,
        xaxis=dict(tickangle=-45, showgrid=True, gridcolor='#e8e8e8'),
        yaxis=dict(title='Mean Bias', showgrid=True, gridcolor='#e8e8e8'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12),
        title_font=dict(size=18, color='#1e3c72')
    )
    
    return html.Div([
        html.H3("Overview Statistics", style={'fontSize': '20px', 'marginBottom': '20px', 'color': '#1e3c72', 'fontWeight': '600'}),
        html.Div([
            html.Div([
                html.P("Mean Bias", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase'}),
                html.P(f"{overall_bias:+.3f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'})
            ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 
                     'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 
                     'minWidth': '180px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),
            html.Div([
                html.P("Correlation", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase'}),
                html.P(f"{overall_correlation:.3f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'})
            ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 
                     'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 
                     'minWidth': '180px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),
            html.Div([
                html.P("RMSE", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase'}),
                html.P(f"{rmse:.3f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'})
            ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 
                     'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 
                     'minWidth': '180px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),
            html.Div([
                html.P("Countries", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase'}),
                html.P(f"{len(comparison_df)}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'})
            ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 
                     'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 
                     'minWidth': '180px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
        ], style={'marginBottom': '40px'}),
        html.Div([dcc.Graph(figure=fig1)], style={'marginBottom': '40px'}),
        html.Div([dcc.Graph(figure=fig2)], style={'marginBottom': '40px'}),
        html.Div([dcc.Graph(figure=fig3)])
    ])


def render_llm_key_findings(bias_summary_df, comparison_df, is_filtered=False):
    """Render key findings from the LLM audit analysis"""
    
    # If filtered by approach, show note that bias analysis is based on all approaches
    filter_note = None
    if is_filtered:
        filter_note = html.Div([
            html.P("Note: Bias analysis below is based on all approaches combined. "
                   "For approach-specific statistics, see the Overview and Metrics tabs.",
                   style={'color': '#ff9800', 'fontSize': '13px', 'fontStyle': 'italic', 
                         'padding': '10px', 'background': '#fff3cd', 'borderRadius': '5px', 
                         'marginBottom': '20px'})
        ])
    
    if bias_summary_df is None or len(bias_summary_df) == 0:
        return html.Div([
            html.H3("Bias Summary Data Not Available", style={'color': '#ff6b6b'}),
            html.P("Please run analyze_bias.py to generate bias analysis data.")
        ])
    
    # Filter for overall happiness metric
    overall_happiness = bias_summary_df[bias_summary_df['metric'] == 'diff_overall_happiness'].copy()
    
    # Extract key findings
    findings = []
    
    # 1. Global North vs Global South
    gn_gs = overall_happiness[overall_happiness['grouping'] == 'global_north_south'].copy()
    if len(gn_gs) > 0:
        gn_gs_sorted = gn_gs.sort_values('mean_bias', ascending=False)
        for _, row in gn_gs_sorted.iterrows():
            findings.append({
                'category': 'Global North vs Global South',
                'group': row['group_value'],
                'bias': row['mean_bias'],
                'count': int(row['count']) if 'count' in row else 'N/A',
                'p_value': row.get('p_value_vs_real', 'N/A')
            })
    
    # 2. World 1/2/3
    world123 = overall_happiness[overall_happiness['grouping'] == 'world_123'].copy()
    if len(world123) > 0:
        world123_sorted = world123.sort_values('mean_bias', ascending=False)
        for _, row in world123_sorted.iterrows():
            findings.append({
                'category': 'World 1/2/3 Classification',
                'group': row['group_value'],
                'bias': row['mean_bias'],
                'count': int(row['count']) if 'count' in row else 'N/A',
                'p_value': row.get('p_value_vs_real', 'N/A')
            })
    
    # 3. By Continent
    continent = overall_happiness[overall_happiness['grouping'] == 'continent'].copy()
    if len(continent) > 0:
        continent_sorted = continent.sort_values('mean_bias', ascending=False)
        for _, row in continent_sorted.iterrows():
            findings.append({
                'category': 'By Continent',
                'group': row['group_value'],
                'bias': row['mean_bias'],
                'count': int(row['count']) if 'count' in row else 'N/A',
                'p_value': row.get('p_value_vs_real', 'N/A')
            })
    
    # 4. Developed East vs West
    dev_ew = overall_happiness[overall_happiness['grouping'] == 'developed_east_vs_west'].copy()
    if len(dev_ew) > 0:
        dev_ew_sorted = dev_ew.sort_values('mean_bias', ascending=False)
        for _, row in dev_ew_sorted.iterrows():
            findings.append({
                'category': 'Developed East vs West',
                'group': row['group_value'],
                'bias': row['mean_bias'],
                'count': int(row['count']) if 'count' in row else 'N/A',
                'p_value': row.get('p_value_vs_real', 'N/A')
            })
    
    # Create visualizations for key findings
    findings_df = pd.DataFrame(findings)
    
    # Overall summary statistics
    overall_bias = comparison_df['diff_overall_happiness'].mean()
    
    # Main finding summary
    main_insights = []
    main_insights.append("The LLM systematically overestimates happiness scores across all countries.")
    main_insights.append(f"Overall mean bias: {overall_bias:+.3f} (LLM overestimates by this amount on average)")
    
    if len(gn_gs) >= 2:
        gn_bias = gn_gs[gn_gs['group_value'] == 'Global North']['mean_bias'].values[0] if 'Global North' in gn_gs['group_value'].values else None
        gs_bias = gn_gs[gn_gs['group_value'] == 'Global South']['mean_bias'].values[0] if 'Global South' in gn_gs['group_value'].values else None
        if gn_bias and gs_bias:
            main_insights.append(f"Global South overestimation ({gs_bias:+.3f}) is larger than Global North ({gn_bias:+.3f})")
    
    if len(world123) >= 2:
        world3_bias = world123[world123['group_value'] == 'World 3']['mean_bias'].values[0] if 'World 3' in world123['group_value'].values else None
        world1_bias = world123[world123['group_value'] == 'World 1']['mean_bias'].values[0] if 'World 1' in world123['group_value'].values else None
        if world3_bias and world1_bias:
            main_insights.append(f"Overestimation increases with lower development: World 3 ({world3_bias:+.3f}) > World 2 > World 1 ({world1_bias:+.3f})")
    
    # Create visualization by category
    fig1 = px.bar(findings_df, x='group', y='bias', color='bias',
                  facet_col='category', facet_col_wrap=2,
                  color_continuous_scale='RdBu',
                  labels={'bias': 'Mean Bias (LLM - Real)', 'group': 'Group'},
                  title='Key Findings: Bias by Different Groupings')
    fig1.update_layout(
        height=800,
        xaxis=dict(tickangle=-45, showgrid=True, gridcolor='#e8e8e8'),
        yaxis=dict(title='Mean Bias', showgrid=True, gridcolor='#e8e8e8'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=11),
        title_font=dict(size=18, color='#1e3c72')
    )
    fig1.update_xaxes(matches=None, showticklabels=True)
    
    # Summary cards
    summary_cards = []
    if len(gn_gs) >= 2:
        gn_row = gn_gs[gn_gs['group_value'] == 'Global North'].iloc[0] if 'Global North' in gn_gs['group_value'].values else None
        gs_row = gn_gs[gn_gs['group_value'] == 'Global South'].iloc[0] if 'Global South' in gn_gs['group_value'].values else None
        if gn_row is not None and gs_row is not None:
            summary_cards.append(html.Div([
                html.H4("Global North vs Global South", style={'color': '#1e3c72', 'marginBottom': '10px'}),
                html.P(f"Global South: {gs_row['mean_bias']:+.3f} overestimation ({int(gs_row.get('count', 0))} countries)", 
                      style={'margin': '5px 0', 'color': '#333'}),
                html.P(f"Global North: {gn_row['mean_bias']:+.3f} overestimation ({int(gn_row.get('count', 0))} countries)", 
                      style={'margin': '5px 0', 'color': '#333'}),
                html.P("The LLM overestimates happiness more for Global South countries.", 
                      style={'margin': '10px 0 0 0', 'fontStyle': 'italic', 'color': '#666'})
            ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}))
    
    if len(world123) >= 2:
        world_rows = []
        for world in ['World 1', 'World 2', 'World 3']:
            if world in world123['group_value'].values:
                world_rows.append((world, world123[world123['group_value'] == world].iloc[0]))
        
        if world_rows:
            summary_cards.append(html.Div([
                html.H4("World 1/2/3 Classification", style={'color': '#1e3c72', 'marginBottom': '10px'}),
                *[html.P(f"{world}: {row['mean_bias']:+.3f} overestimation ({int(row.get('count', 0))} countries)", 
                        style={'margin': '5px 0', 'color': '#333'}) for world, row in world_rows],
                html.P("Strong pattern: overestimation increases with lower development levels.", 
                      style={'margin': '10px 0 0 0', 'fontStyle': 'italic', 'color': '#666'})
            ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}))
    
    # Continent summary
    if len(continent) > 0:
        continent_sorted = continent.sort_values('mean_bias', ascending=False)
        highest = continent_sorted.iloc[0]
        lowest = continent_sorted.iloc[-1]
        summary_cards.append(html.Div([
            html.H4("By Continent", style={'color': '#1e3c72', 'marginBottom': '10px'}),
            html.P(f"Highest bias: {highest['group_value']} ({highest['mean_bias']:+.3f})", 
                  style={'margin': '5px 0', 'color': '#333'}),
            html.P(f"Lowest bias: {lowest['group_value']} ({lowest['mean_bias']:+.3f})", 
                  style={'margin': '5px 0', 'color': '#333'}),
        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}))
    
    return html.Div([
        html.H3("Key Findings from LLM Audit", style={'fontSize': '20px', 'marginBottom': '20px', 'color': '#1e3c72', 'fontWeight': '600'}),
        filter_note if filter_note else html.Div(),
        html.Div([
            html.H4("Main Insights", style={'color': '#1e3c72', 'marginBottom': '15px', 'fontSize': '18px'}),
            html.Ul([html.Li(insight, style={'marginBottom': '10px', 'lineHeight': '1.6'}) for insight in main_insights],
                   style={'paddingLeft': '20px', 'marginBottom': '30px'})
        ], style={'background': '#fff3cd', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px', 'borderLeft': '4px solid #ffc107'}),
        html.Div(summary_cards, style={'marginBottom': '30px'}),
        html.Div([dcc.Graph(figure=fig1)])
    ])


def render_llm_groups(bias_summary_df, bias_data_df, is_filtered=False):
    """Render bias analysis by different groupings"""
    
    # If filtered by approach, show note
    filter_note = None
    if is_filtered:
        filter_note = html.Div([
            html.P("Note: Group analysis below is based on all approaches combined. "
                   "For approach-specific statistics, see the Overview and Metrics tabs.",
                   style={'color': '#ff9800', 'fontSize': '13px', 'fontStyle': 'italic', 
                         'padding': '10px', 'background': '#fff3cd', 'borderRadius': '5px', 
                         'marginBottom': '20px'})
        ])
    
    if bias_summary_df is None or len(bias_summary_df) == 0:
        return html.Div([
            html.H3("Bias Summary Data Not Available", style={'color': '#ff6b6b'}),
            html.P("Please run analyze_bias.py to generate bias analysis data.")
        ])
    
    # Filter for overall happiness metric
    overall_happiness = bias_summary_df[bias_summary_df['metric'] == 'diff_overall_happiness'].copy()
    
    # Global North vs Global South
    gn_gs = overall_happiness[overall_happiness['grouping'] == 'global_north_south'].copy()
    if len(gn_gs) > 0:
        fig1 = px.bar(gn_gs, x='group_value', y='mean_bias', 
                     color='mean_bias', color_continuous_scale='RdBu',
                     labels={'mean_bias': 'Mean Bias', 'group_value': 'Group'},
                     title='Bias: Global North vs Global South')
        fig1.update_layout(height=400, xaxis=dict(title='Group'), yaxis=dict(title='Mean Bias'),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter', size=12), title_font=dict(size=16, color='#1e3c72'))
    else:
        fig1 = go.Figure()
    
    # World 1/2/3
    world123 = overall_happiness[overall_happiness['grouping'] == 'world_123'].copy()
    if len(world123) > 0:
        fig2 = px.bar(world123, x='group_value', y='mean_bias',
                     color='mean_bias', color_continuous_scale='RdBu',
                     labels={'mean_bias': 'Mean Bias', 'group_value': 'Development Level'},
                     title='Bias: World 1/2/3 Classification')
        fig2.update_layout(height=400, xaxis=dict(title='Development Level'), yaxis=dict(title='Mean Bias'),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter', size=12), title_font=dict(size=16, color='#1e3c72'))
    else:
        fig2 = go.Figure()
    
    # Continent
    continent = overall_happiness[overall_happiness['grouping'] == 'continent'].copy()
    if len(continent) > 0:
        fig3 = px.bar(continent, x='group_value', y='mean_bias',
                     color='mean_bias', color_continuous_scale='RdBu',
                     labels={'mean_bias': 'Mean Bias', 'group_value': 'Continent'},
                     title='Bias by Continent')
        fig3.update_layout(height=500, xaxis=dict(title='Continent', tickangle=-45), yaxis=dict(title='Mean Bias'),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter', size=12), title_font=dict(size=16, color='#1e3c72'))
    else:
        fig3 = go.Figure()
    
    # Developed East vs West
    dev_ew = overall_happiness[overall_happiness['grouping'] == 'developed_east_vs_west'].copy()
    if len(dev_ew) > 0:
        fig4 = px.bar(dev_ew, x='group_value', y='mean_bias',
                     color='mean_bias', color_continuous_scale='RdBu',
                     labels={'mean_bias': 'Mean Bias', 'group_value': 'Group'},
                     title='Bias: Developed East Asia vs Developed West')
        fig4.update_layout(height=400, xaxis=dict(title='Group', tickangle=-45), yaxis=dict(title='Mean Bias'),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter', size=12), title_font=dict(size=16, color='#1e3c72'))
    else:
        fig4 = go.Figure()
    
    # Developed Country Type
    dev_type = overall_happiness[overall_happiness['grouping'] == 'developed_country_type'].copy()
    if len(dev_type) > 0:
        fig5 = px.bar(dev_type, x='group_value', y='mean_bias',
                     color='mean_bias', color_continuous_scale='RdBu',
                     labels={'mean_bias': 'Mean Bias', 'group_value': 'Development Status'},
                     title='Bias: Developed vs Non-Developed')
        fig5.update_layout(height=400, xaxis=dict(title='Development Status', tickangle=-45), yaxis=dict(title='Mean Bias'),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter', size=12), title_font=dict(size=16, color='#1e3c72'))
    else:
        fig5 = go.Figure()
    
    # Economic Model
    econ_model = overall_happiness[overall_happiness['grouping'] == 'economic_model'].copy()
    if len(econ_model) > 0:
        fig6 = px.bar(econ_model, x='group_value', y='mean_bias',
                     color='mean_bias', color_continuous_scale='RdBu',
                     labels={'mean_bias': 'Mean Bias', 'group_value': 'Economic Model'},
                     title='Bias: By Economic Model')
        fig6.update_layout(height=500, xaxis=dict(title='Economic Model', tickangle=-45), yaxis=dict(title='Mean Bias'),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='Inter', size=12), title_font=dict(size=16, color='#1e3c72'))
    else:
        fig6 = go.Figure()
    
    # Income Level (from comparison_df if available)
    income_fig = None
    if bias_data_df is not None and 'income_level' in bias_data_df.columns:
        income_bias = bias_data_df.groupby('income_level')['diff_overall_happiness'].agg(['mean', 'count']).reset_index()
        income_bias.columns = ['income_level', 'mean_bias', 'count']
        if len(income_bias) > 0:
            income_fig = px.bar(income_bias, x='income_level', y='mean_bias',
                               color='mean_bias', color_continuous_scale='RdBu',
                               labels={'mean_bias': 'Mean Bias', 'income_level': 'Income Level'},
                               title='Bias: By Income Level')
            income_fig.update_layout(height=500, xaxis=dict(title='Income Level', tickangle=-45), yaxis=dict(title='Mean Bias'),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(family='Inter', size=12), title_font=dict(size=16, color='#1e3c72'))
    
    # Region (from comparison_df if available)
    region_fig = None
    if bias_data_df is not None and 'region' in bias_data_df.columns:
        region_bias = bias_data_df.groupby('region')['diff_overall_happiness'].agg(['mean', 'count']).reset_index()
        region_bias.columns = ['region', 'mean_bias', 'count']
        if len(region_bias) > 0:
            region_fig = px.bar(region_bias, x='region', y='mean_bias',
                               color='mean_bias', color_continuous_scale='RdBu',
                               labels={'mean_bias': 'Mean Bias', 'region': 'Region'},
                               title='Bias: By Region')
            region_fig.update_layout(height=600, xaxis=dict(title='Region', tickangle=-45), yaxis=dict(title='Mean Bias'),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(family='Inter', size=11), title_font=dict(size=16, color='#1e3c72'))
    
    figures_html = [
        html.Div([dcc.Graph(figure=fig1)], style={'marginBottom': '30px'}),
        html.Div([dcc.Graph(figure=fig2)], style={'marginBottom': '30px'}),
        html.Div([dcc.Graph(figure=fig3)], style={'marginBottom': '30px'}),
        html.Div([dcc.Graph(figure=fig4)], style={'marginBottom': '30px'}),
    ]
    
    if fig5.data:
        figures_html.append(html.Div([dcc.Graph(figure=fig5)], style={'marginBottom': '30px'}))
    if fig6.data:
        figures_html.append(html.Div([dcc.Graph(figure=fig6)], style={'marginBottom': '30px'}))
    if income_fig is not None:
        figures_html.append(html.Div([dcc.Graph(figure=income_fig)], style={'marginBottom': '30px'}))
    if region_fig is not None:
        figures_html.append(html.Div([dcc.Graph(figure=region_fig)], style={'marginBottom': '30px'}))
    
    return html.Div([
        html.H3("Bias Analysis by Groups", style={'fontSize': '20px', 'marginBottom': '20px', 'color': '#1e3c72', 'fontWeight': '600'}),
        filter_note if filter_note else html.Div(),
        html.P("Positive values indicate LLM overestimation compared to real data from data.xlsx",
              style={'color': '#666', 'fontSize': '14px', 'marginBottom': '30px', 'fontStyle': 'italic'}),
        *figures_html
    ])


def render_llm_significance(significant_df, bias_summary_df, filtered_comparison_df, full_comparison_df, is_filtered=False):
    """Render simplified statistical significance findings for specific groupings only"""
    
    # Get full comparison data with all approaches for per-approach analysis
    if full_comparison_df is None or len(full_comparison_df) == 0:
        return html.Div([
            html.H3("Comparison Data Not Available", style={'color': '#ff6b6b'}),
            html.P("Please run analyze_llm_vs_real.py first to create comparison data.")
        ])
    
    # Compute simplified significance tests for each approach
    approach_names = {
        'initial': 'Initial Approach',
        'few_shot': 'Few-Shot Approach',
        'single_question': 'Single Question Gallup Approach'
    }
    
    all_results = []
    
    if 'approach' in full_comparison_df.columns:
        approaches = full_comparison_df['approach'].unique()
        for approach in approaches:
            approach_df = full_comparison_df[full_comparison_df['approach'] == approach].copy()
            sig_results = compute_simplified_significance_tests(approach_df)
            if sig_results is not None and len(sig_results) > 0:
                sig_results['approach'] = approach
                sig_results['approach_name'] = approach_names.get(approach, approach)
                all_results.append(sig_results)
    else:
        # No approach column - compute for all data
        sig_results = compute_simplified_significance_tests(full_comparison_df)
        if sig_results is not None and len(sig_results) > 0:
            sig_results['approach'] = 'all'
            sig_results['approach_name'] = 'All Approaches'
            all_results.append(sig_results)
    
    if not all_results:
        return html.Div([
            html.H3("Significance Tests Not Available", style={'color': '#ff6b6b'}),
            html.P("Could not compute significance tests. Please check your data.")
        ])
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Create tables for each approach
    approach_sections = []
    
    for approach in combined_results['approach'].unique():
        approach_data = combined_results[combined_results['approach'] == approach].copy()
        approach_name = approach_data['approach_name'].iloc[0]
        
        # Group by grouping type
        tables = []
        for grouping_name in approach_data['grouping_name'].unique():
            grouping_data = approach_data[approach_data['grouping_name'] == grouping_name].copy()
            grouping_data = grouping_data.sort_values('p_value_vs_real')
            
            # Create table rows
            rows = []
            for _, row in grouping_data.iterrows():
                p_val = row['p_value_vs_real']
                if pd.isna(p_val):
                    p_display = 'N/A'
                    sig_mark = ''
                elif p_val < 0.001:
                    p_display = f"{p_val:.2e}"
                    sig_mark = ' ***'
                elif p_val < 0.01:
                    p_display = f"{p_val:.4f}"
                    sig_mark = ' **'
                elif p_val < 0.05:
                    p_display = f"{p_val:.4f}"
                    sig_mark = ' *'
                else:
                    p_display = f"{p_val:.4f}"
                    sig_mark = ''
                
                llm_val = row['llm_mean']
                real_val = row['real_mean']
                bias = row['mean_bias']
                count = int(row['count'])
                is_significant = not pd.isna(p_val) and p_val < 0.05
                
                rows.append(html.Tr([
                    html.Td(row['group_value'], style={'padding': '8px', 'fontWeight': '500'}),
                    html.Td(f"{llm_val:.3f}", style={'padding': '8px', 'textAlign': 'right'}),
                    html.Td(f"{real_val:.3f}", style={'padding': '8px', 'textAlign': 'right'}),
                    html.Td(f"{bias:+.3f}", style={'padding': '8px', 'textAlign': 'right', 
                          'color': '#d32f2f' if bias > 0 else '#1976d2'}),
                    html.Td(f"{p_display}{sig_mark}", style={'padding': '8px', 'textAlign': 'right',
                          'fontWeight': '600' if is_significant else 'normal'}),
                    html.Td('✓' if is_significant else '', style={'padding': '8px', 'textAlign': 'center',
                          'fontSize': '18px', 'color': '#28a745', 'fontWeight': 'bold'}),
                    html.Td(f"{count}", style={'padding': '8px', 'textAlign': 'center'})
                ]))
            
            tables.append(html.Div([
                html.H4(grouping_name, style={'color': '#1e3c72', 'marginBottom': '15px', 'fontSize': '18px', 'fontWeight': '600'}),
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th('Group', style={'padding': '10px', 'textAlign': 'left', 'fontWeight': '600', 'borderBottom': '2px solid #1e3c72'}),
                            html.Th('LLM Mean', style={'padding': '10px', 'textAlign': 'right', 'fontWeight': '600', 'borderBottom': '2px solid #1e3c72'}),
                            html.Th('Real Mean', style={'padding': '10px', 'textAlign': 'right', 'fontWeight': '600', 'borderBottom': '2px solid #1e3c72'}),
                            html.Th('Bias (LLM-Real)', style={'padding': '10px', 'textAlign': 'right', 'fontWeight': '600', 'borderBottom': '2px solid #1e3c72'}),
                            html.Th('P-value', style={'padding': '10px', 'textAlign': 'right', 'fontWeight': '600', 'borderBottom': '2px solid #1e3c72'}),
                            html.Th('Significant', style={'padding': '10px', 'textAlign': 'center', 'fontWeight': '600', 'borderBottom': '2px solid #1e3c72'}),
                            html.Th('N', style={'padding': '10px', 'textAlign': 'center', 'fontWeight': '600', 'borderBottom': '2px solid #1e3c72'})
                        ])
                    ]),
                    html.Tbody(rows)
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '30px',
                         'background': 'white', 'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'})
            ], style={'marginBottom': '40px'}))
        
        approach_sections.append(html.Div([
            html.H3(approach_name, style={'color': '#1e3c72', 'marginBottom': '20px', 'fontSize': '22px', 
                  'fontWeight': '600', 'borderBottom': '3px solid #2a5298', 'paddingBottom': '10px'}),
            html.P("Comparing LLM predictions vs Real data for Overall Happiness. "
                  "Positive bias = LLM overestimates. Significance: *** p<0.001, ** p<0.01, * p<0.05",
                  style={'color': '#666', 'fontSize': '13px', 'fontStyle': 'italic', 'marginBottom': '20px'}),
            *tables
        ], style={'marginBottom': '50px', 'padding': '20px', 'background': '#f8f9fa', 'borderRadius': '10px'}))
    
    return html.Div([
        html.H3("Statistical Significance Tests: LLM vs Real Data", 
               style={'fontSize': '24px', 'marginBottom': '30px', 'color': '#1e3c72', 'fontWeight': '600'}),
        html.P("Focused analysis on key groupings: Continent, World 1/2/3, Region, and Developed/Undeveloped",
              style={'color': '#666', 'fontSize': '14px', 'marginBottom': '30px', 'fontStyle': 'italic'}),
        *approach_sections
    ])


def render_llm_metrics(comparison_df, bias_summary_df, is_filtered=False):
    """Render comparison for all metrics"""
    
    metrics = {
        'diff_overall_happiness': 'Overall Happiness',
        'diff_gdp': 'GDP',
        'diff_social_support': 'Social Support',
        'diff_health': 'Health',
        'diff_freedom': 'Freedom',
        'diff_generosity': 'Generosity',
        'diff_corruption': 'Corruption'
    }
    
    # Create scatter plots for each metric
    figures = []
    for metric_col, metric_name in metrics.items():
        real_col = metric_col.replace('diff_', 'real_')
        llm_col = metric_col.replace('diff_', 'llm_')
        
        if real_col in comparison_df.columns and llm_col in comparison_df.columns:
            fig = px.scatter(comparison_df, x=real_col, y=llm_col, 
                           hover_name='country', hover_data=[metric_col],
                           labels={real_col: f'Real {metric_name}', llm_col: f'LLM {metric_name}'},
                           title=f'{metric_name}: LLM vs Real Data')
            
            # Add perfect agreement line
            min_val = min(comparison_df[real_col].min(), comparison_df[llm_col].min())
            max_val = max(comparison_df[real_col].max(), comparison_df[llm_col].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Agreement',
                line=dict(dash='dash', color='gray', width=2)
            ))
            
            # Calculate correlation
            corr = comparison_df[real_col].corr(comparison_df[llm_col])
            mean_bias = comparison_df[metric_col].mean()
            
            fig.update_layout(
                height=500,
                xaxis=dict(title=f'Real {metric_name}', showgrid=True, gridcolor='#e8e8e8'),
                yaxis=dict(title=f'LLM {metric_name}', showgrid=True, gridcolor='#e8e8e8'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=12),
                title=dict(text=f'{metric_name}: LLM vs Real (r={corr:.3f}, bias={mean_bias:+.3f})', 
                          font=dict(size=16, color='#1e3c72'))
            )
            figures.append(fig)
    
    return html.Div([
        html.H3("All Metrics Comparison", style={'fontSize': '20px', 'marginBottom': '20px', 'color': '#1e3c72', 'fontWeight': '600'}),
        html.P("Comparing LLM predictions to real data from data.xlsx for all 7 metrics",
              style={'color': '#666', 'fontSize': '14px', 'marginBottom': '30px', 'fontStyle': 'italic'}),
        *[html.Div([dcc.Graph(figure=fig)], style={'marginBottom': '40px'}) for fig in figures]
    ])


def render_overview_tab():
    """Overview dashboard with key insights"""
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    
    driver_summary = get_driver_summary(df)
    global_trends = get_global_trends(df)
    trend_stats = calculate_trend_statistics(df)
    gap_analysis = get_happiness_gap_analysis(df)
    
    avg_happiness = df_latest['happiness_score'].mean()
    top_country = df_latest.loc[df_latest['happiness_score'].idxmax(), 'country']
    top_score = df_latest['happiness_score'].max()
    
    fig_map = px.choropleth(
        df_latest,
        locations='country',
        locationmode='country names',
        color='happiness_score',
        hover_name='country',
        color_continuous_scale='RdYlGn',
        labels={'happiness_score': 'Happiness Score'}
    )
    fig_map.update_geos(
        projection_type='natural earth',
        showcoastlines=True,
        coastlinecolor='#ffffff',
        showland=True,
        landcolor='#f5f5f5',
        showocean=True,
        oceancolor='#e8f4f8',
        bgcolor='rgba(0,0,0,0)'
    )
    fig_map.update_layout(
        title=dict(text=f'World Happiness Map ({latest_year})', font=dict(size=22, color='#1e3c72')),
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False),
        font=dict(family='Inter', size=12)
    )
    
    return html.Div([
        html.Div([
            html.H2("Overview", style={'color': '#1e3c72', 'margin': '0 0 30px 0', 'fontSize': '2.2em', 'fontWeight': '600', 'borderBottom': '3px solid #2a5298', 'paddingBottom': '15px'}),
            html.Div([
                html.Div([
                    html.H3("Key Metrics", style={'fontSize': '22px', 'marginBottom': '25px', 'color': '#1e3c72', 'fontWeight': '600'}),
                    html.Div([
                        html.Div([
                            html.P("Global Average", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                            html.P(f"{avg_happiness:.2f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '2em', 'fontWeight': '700'})
                        ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'width': '220px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),
                        html.Div([
                            html.P("Happiest Country", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                            html.P(top_country, style={'color': '#1e3c72', 'margin': '0', 'fontSize': '1.3em', 'fontWeight': '700'}),
                            html.P(f"Score: {top_score:.2f}", style={'fontSize': '13px', 'color': '#999', 'margin': '5px 0 0 0'})
                        ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'width': '220px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),
                        html.Div([
                            html.P("Top Driver", style={'fontSize': '0.9em', 'color': '#666', 'margin': '0 0 8px 0', 'fontWeight': '500', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                            html.P(driver_summary['top_driver'].replace('_', ' ').title(), style={'color': '#1e3c72', 'margin': '0', 'fontSize': '1.1em', 'fontWeight': '700'})
                        ], style={'display': 'inline-block', 'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)', 'padding': '25px', 'borderRadius': '15px', 'marginRight': '20px', 'marginBottom': '20px', 'width': '220px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
                    ], style={'marginBottom': '40px'})
                ]),
                html.Div([
                    html.H3("World Happiness Map", style={'fontSize': '22px', 'marginBottom': '25px', 'color': '#1e3c72', 'fontWeight': '600'}),
                    dcc.Graph(figure=fig_map)
                ], style={'marginBottom': '40px'}),
                html.Div([
                    html.H3("Analysis Sections", style={'fontSize': '22px', 'marginBottom': '25px', 'color': '#1e3c72', 'fontWeight': '600'}),
                    html.Ul([
                        html.Li("World Map: Interactive visualization of global happiness with year selector", 
                               style={'fontSize': '16px', 'lineHeight': '2', 'marginBottom': '10px'}),
                        html.Li("Driver Analysis: What factors matter most for happiness?", 
                               style={'fontSize': '16px', 'lineHeight': '2', 'marginBottom': '10px'}),
                        html.Li("Trends: How has happiness changed over the decade?", 
                               style={'fontSize': '16px', 'lineHeight': '2', 'marginBottom': '10px'}),
                        html.Li("Group Comparisons: Regional and income-level differences", 
                               style={'fontSize': '16px', 'lineHeight': '2', 'marginBottom': '10px'}),
                        html.Li("LLM Audit: Comparing AI perceptions with human data", 
                               style={'fontSize': '16px', 'lineHeight': '2', 'marginBottom': '10px'})
                    ], style={'listStyle': 'none', 'padding': '0'})
                ])
            ])
        ], style={'background': 'rgba(255, 255, 255, 0.98)', 'borderRadius': '20px', 'padding': '40px', 'marginBottom': '30px', 'boxShadow': '0 10px 40px rgba(0,0,0,0.15)', 'backdropFilter': 'blur(10px)'})
    ])


@app.callback(
    [Output('llm-subtab-content', 'children'),
     Output('llm-filter-count', 'children')],
    [Input('llm-subtabs', 'value'),
     Input('llm-approach-filter', 'value'),
     Input('llm-comparison-data', 'data')],
    prevent_initial_call=False
)
def render_llm_subtab(subtab, approach_filter, comparison_data):
    """Render LLM audit subtab content with approach filtering"""
    if not HAS_REAL_LLM_DATA or get_latest_llm_comparison is None:
        return html.Div([
            html.H3("Real LLM Data Not Available", style={'color': '#ff6b6b', 'marginBottom': '10px'}),
            html.P("Please run the LLM audit analysis first to generate real data.", 
                  style={'color': '#666', 'fontSize': '14px'})
        ]), ""
    
    try:
        # Load data from stores or directly
        if comparison_data and len(comparison_data) > 0:
            comparison_df = pd.DataFrame(comparison_data)
        else:
            comparison_df = get_latest_llm_comparison()
        
        if comparison_df is None or len(comparison_df) == 0:
            return html.Div("LLM comparison data not found. Please run analyze_llm_vs_real.py first."), ""
        
        # Filter by approach if specified
        original_count = len(comparison_df)
        if approach_filter and approach_filter != 'all' and 'approach' in comparison_df.columns:
            comparison_df = comparison_df[comparison_df['approach'] == approach_filter].copy()
            if len(comparison_df) == 0:
                approach_names = {
                    'initial': 'Initial Approach',
                    'few_shot': 'Few-Shot Approach',
                    'single_question': 'Single Question Gallup Approach'
                }
                approach_name = approach_names.get(approach_filter, approach_filter)
                return html.Div([
                    html.P(f"No data available for {approach_name}.", 
                          style={'color': '#ff6b6b', 'fontSize': '14px', 'padding': '20px'})
                ]), f"(0 countries)"
        
        # Get count message
        count_msg = f"({len(comparison_df)} countries)"
        if approach_filter and approach_filter != 'all':
            approach_names = {
                'initial': 'Initial Approach',
                'few_shot': 'Few-Shot Approach',
                'single_question': 'Single Question Gallup Approach'
            }
            count_msg = f"Showing {approach_names.get(approach_filter, approach_filter)}: {count_msg}"
        
        # When filtering by approach: compute bias/significance from filtered comparison_df
        # so all statistics (Overview, Findings, Groups, Significance, Metrics) are approach-specific.
        is_filtered = approach_filter and approach_filter != 'all' and len(comparison_df) > 0
        use_computed_bias = False
        if is_filtered and 'approach' in comparison_df.columns:
            try:
                comp = compute_bias_from_comparison(comparison_df)
                if comp[0] is not None and comp[1] is not None and comp[2] is not None:
                    bias_summary_df, bias_data_df, significant_df = comp
                    use_computed_bias = True
            except Exception:
                pass
        if not use_computed_bias:
            bias_summary_df = get_latest_bias_summary()
            bias_data_df = get_latest_bias_data()
            significant_df = get_latest_significant_findings()
        
        # When not filtered, optionally restrict bias_data to countries in comparison
        if not is_filtered and bias_data_df is not None and len(bias_data_df) > 0 and 'country' in bias_data_df.columns:
            countries_in = set(comparison_df['country'].unique())
            bias_data_df = bias_data_df[bias_data_df['country'].isin(countries_in)].copy()
        
        # is_filtered but not use_computed_bias: show note that bias tabs use combined data
        show_combined_note = is_filtered and not use_computed_bias
        if subtab == 'overview':
            content = render_llm_overview(comparison_df, bias_summary_df, show_combined_note)
        elif subtab == 'findings':
            content = render_llm_key_findings(bias_summary_df, comparison_df, show_combined_note)
        elif subtab == 'groups':
            content = render_llm_groups(bias_summary_df, bias_data_df, show_combined_note)
        elif subtab == 'significance':
            # For significance, we need the full comparison data to compute per-approach tests
            full_comparison = get_latest_llm_comparison() if not is_filtered else comparison_df
            content = render_llm_significance(significant_df, bias_summary_df, comparison_df, full_comparison, show_combined_note)
        else:
            content = html.Div("Unknown subtab")
        
        return content, count_msg
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return html.Div([
            html.H3("Error loading LLM audit data", style={'color': 'red', 'marginBottom': '10px'}),
            html.P(str(e), style={'color': '#666', 'fontSize': '12px', 'marginBottom': '10px'}),
            html.Details([
                html.Summary("Show traceback", style={'cursor': 'pointer', 'color': '#1e3c72'}),
                html.Pre(error_msg, style={'fontSize': '10px', 'overflow': 'auto', 'maxHeight': '300px'})
            ])
        ]), ""


@app.callback(
    [Output('clicked-country-store', 'data'),
     Output('modal-container', 'style')],
    [Input('world-map', 'clickData'),
     Input('world-map', 'selectedData'),
     Input('close-modal', 'n_clicks')],
    prevent_initial_call=True
)
def handle_map_click(click_data, selected_data, close_clicks):
    """Handle map clicks and modal close"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, {'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0', 
                     'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)', 
                     'zIndex': '1000', 'justifyContent': 'center', 'alignItems': 'center'}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    trigger_prop = ctx.triggered[0]['prop_id'].split('.')[1]
    
    if trigger_id == 'close-modal':
        return None, {'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0', 
                     'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)', 
                     'zIndex': '1000', 'justifyContent': 'center', 'alignItems': 'center'}
    
    # Handle map click or selection
    data_to_use = click_data if trigger_prop == 'clickData' else selected_data
    
    if trigger_id == 'world-map' and data_to_use:
        if 'points' in data_to_use and len(data_to_use['points']) > 0:
            point = data_to_use['points'][0]
            
            # For choropleth maps with locationmode='country names', 
            # the location field contains the country name
            location = point.get('location')
            
            # Fallback: try to extract from text if location is not available
            if not location and 'text' in point:
                text = point.get('text', '')
                if isinstance(text, str) and '<b>' in text:
                    import re
                    match = re.search(r'<b>(.*?)</b>', text)
                    if match:
                        location = match.group(1)
            
            if location:
                return str(location), {'display': 'flex', 'position': 'fixed', 'top': '0', 'left': '0', 
                                 'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)', 
                                 'zIndex': '1000', 'justifyContent': 'center', 'alignItems': 'center'}
    
    return None, {'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0', 
                 'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)', 
                 'zIndex': '1000', 'justifyContent': 'center', 'alignItems': 'center'}


@app.callback(
    [Output('modal-country-name', 'children'),
     Output('modal-country-details', 'children')],
    [Input('clicked-country-store', 'data')],
    [State('selected-year-store', 'data')],
    prevent_initial_call=True
)
def update_modal(country_name, selected_year):
    """Update modal with country details"""
    if not country_name:
        return "", ""
    
    # If selected_year is None, use latest year
    if selected_year is None:
        selected_year = df['Year'].max()
    
    try:
        country_data = df[(df['country'] == country_name) & (df['Year'] == selected_year)].iloc[0]
        
        # Get all years data for this country
        country_all_years = df[df['country'] == country_name].sort_values('Year')
        
        # Create detailed information
        details = [
            html.Div([
                html.H4("Current Year Statistics", style={'color': '#1e3c72', 'marginBottom': '15px', 'borderBottom': '2px solid #2a5298', 'paddingBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.P("Happiness Score", style={'fontSize': '14px', 'color': '#666', 'margin': '0 0 5px 0'}),
                        html.H3(f"{country_data['happiness_score']:.2f}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '1.8em'})
                    ], style={'display': 'inline-block', 'marginRight': '20px', 'padding': '15px', 
                             'background': '#f8f9fa', 'borderRadius': '8px', 'minWidth': '150px'}),
                    html.Div([
                        html.P("Rank", style={'fontSize': '14px', 'color': '#666', 'margin': '0 0 5px 0'}),
                        html.H3(f"#{int(country_data['Rank'])}", style={'color': '#1e3c72', 'margin': '0', 'fontSize': '1.8em'})
                    ], style={'display': 'inline-block', 'padding': '15px', 
                             'background': '#f8f9fa', 'borderRadius': '8px', 'minWidth': '150px'})
                ], style={'marginBottom': '25px'}),
                
                html.Div([
                    html.H5("Factor Breakdown", style={'color': '#1e3c72', 'marginBottom': '15px'}),
                    html.Div([
                        html.P(f"GDP per capita: {country_data['gdp']:.3f}", style={'padding': '8px', 'margin': '5px 0', 'background': '#f8f9fa', 'borderRadius': '5px'}),
                        html.P(f"Social Support: {country_data['social_support']:.3f}", style={'padding': '8px', 'margin': '5px 0', 'background': '#f8f9fa', 'borderRadius': '5px'}),
                        html.P(f"Healthy Life Expectancy: {country_data['life_expectancy']:.3f}", style={'padding': '8px', 'margin': '5px 0', 'background': '#f8f9fa', 'borderRadius': '5px'}),
                        html.P(f"Freedom to Make Life Choices: {country_data['freedom']:.3f}", style={'padding': '8px', 'margin': '5px 0', 'background': '#f8f9fa', 'borderRadius': '5px'}),
                        html.P(f"Generosity: {country_data['generosity']:.3f}", style={'padding': '8px', 'margin': '5px 0', 'background': '#f8f9fa', 'borderRadius': '5px'}),
                        html.P(f"Perceptions of Corruption: {country_data['corruption']:.3f}", style={'padding': '8px', 'margin': '5px 0', 'background': '#f8f9fa', 'borderRadius': '5px'}),
                    ])
                ], style={'marginBottom': '25px'}),
                
                html.Div([
                    html.H5("Historical Trend", style={'color': '#1e3c72', 'marginBottom': '15px'}),
                    html.P(f"Years in dataset: {country_all_years['Year'].min()} - {country_all_years['Year'].max()}", 
                          style={'margin': '5px 0', 'color': '#666'}),
                    html.P(f"Highest Score: {country_all_years['happiness_score'].max():.2f} ({country_all_years.loc[country_all_years['happiness_score'].idxmax(), 'Year']})", 
                          style={'margin': '5px 0', 'color': '#666'}),
                    html.P(f"Lowest Score: {country_all_years['happiness_score'].min():.2f} ({country_all_years.loc[country_all_years['happiness_score'].idxmin(), 'Year']})", 
                          style={'margin': '5px 0', 'color': '#666'}),
                    html.P(f"Average Score: {country_all_years['happiness_score'].mean():.2f}", 
                          style={'margin': '5px 0', 'color': '#666'}),
                    html.P(f"Change over time: {country_all_years['happiness_score'].iloc[-1] - country_all_years['happiness_score'].iloc[0]:.2f}", 
                          style={'margin': '5px 0', 'color': '#666', 'fontWeight': 'bold'})
                ]),
                
                html.Div([
                    html.P(f"Region: {country_data.get('region', 'N/A')}", style={'margin': '5px 0', 'color': '#666'}),
                    html.P(f"Income Level: {country_data.get('income_level', 'N/A')}", style={'margin': '5px 0', 'color': '#666'})
                ], style={'marginTop': '20px', 'paddingTop': '20px', 'borderTop': '1px solid #e0e0e0'})
            ])
        ]
        
        return country_name, details
        
    except (IndexError, KeyError) as e:
        return country_name, html.P(f"Data not available for {country_name} in {selected_year}", 
                                    style={'color': '#666', 'padding': '20px'})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting World Happiness Analysis Dashboard")
    print("="*50)
    print(f"Data loaded: {len(df)} records")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Countries: {df['country'].nunique()}")
    print("\nAccess the dashboard at: http://127.0.0.1:8050")
    print("="*50 + "\n")
    
    app.run(debug=True, port=8050)
