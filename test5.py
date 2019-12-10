import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go

app = dash.Dash(__name__)
app.title = 'S100B'

layout = go.Layout(showlegend=True, title='S100B', yaxis=dict(title='S-100 Values', rangemode='nonnegative',
                                                                        autorange=False, range=[0, 0.17], zeroline=True,
                                                                        showline=True,
                                                                        mirror='ticks'),
                   xaxis=dict(title='Months', rangemode='nonnegative',
                              autorange=True, zeroline=True, showline=True, mirror='ticks'))

params = [
    'S-100 value', 'Date'
]

app.layout = html.Div([
    dash_table.DataTable(
        id='table-editing-simple',
        columns=(
            [{'id': p, 'name': p} for p in params]
        ),
        data=[
        ],
        editable=True,
        row_deletable=True
    ),
    html.Button('Add Row', id='editing-rows-button', n_clicks=0),
    html.Div(id='output-state'),

])


@app.callback(
    Output('table-editing-simple', 'data'),
    [Input('editing-rows-button', 'n_clicks')],
    [State('table-editing-simple', 'data'),
     State('table-editing-simple', 'columns')])
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: '' for c in columns})
    return rows


def calculate_score_points(x, y, distance):
    x_aux = x[distance:]
    y_aux = y[distance:]

    y_aux = np.clip(y_aux, a_min=0.05, a_max=None)

    score = ((y_aux[-1] * 100 / y_aux[distance]) - 100) / (x_aux[-1] - x_aux[distance])
    return score


@app.callback(
    # Output('table-editing-simple-output', 'figure'),
    Output('output-state', 'children'),
    [Input('table-editing-simple', 'data'),
     Input('table-editing-simple', 'columns')])
def display_output(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    df['S-100 value'] = pd.to_numeric(df['S-100 value'], errors='coerce')
    df = df.dropna()
    df = df[(df.T != 0).any()]

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values('Date')

        x = df['Date']
        x = (x - x.iloc[0]).map(lambda x: x.days / 30)
    except Exception:
        import traceback
        traceback.print_exc()
        return

    x = x.values
    y_orig = df['S-100 value'].values
    y = np.clip(y_orig, 0.05, None)

    score_2 = -1
    score_3 = -1

    color_2 = ('rgb(24, 205, 12)')
    color_3 = ('rgb(24, 205, 12)')

    alert = False
    if len(x) == 2:
        score_2 = calculate_score_points(x, y, -2)

        if score_2 > 20:
            color_2 = ('rgb(205, 12, 24)')

    elif len(x) >= 2:
        score_2 = calculate_score_points(x, y, -2)
        score_3 = calculate_score_points(x, y, -3)

        if score_2 > 20:
            color_2 = ('rgb(205, 12, 24)')

        if score_3 > 4:
            color_3 = ('rgb(205, 12, 24)')

    lines = []

    new_data = go.Scatter(x=x, y=y, mode='markers', marker=dict(
        color='rgb(84, 86, 89)',
        size=12, symbol='circle'), showlegend=False, hoverinfo='none')
    lines.append(new_data)

    try:
        curve_2_pos = go.Scatter(x=x.take([-2, -1]), y=y.take([-2, -1]), mode='lines',
                                 line=dict(color=color_2, dash='dash'), showlegend=True,
                                 name='Score_2_points = {} %'.format('{:.1f}'.format(score_2)))
        lines.append(curve_2_pos)
    except Exception:
        import traceback
        traceback.print_exc()
    try:
        curve_3_pos = go.Scatter(x=x.take([-3, -1]), y=y.take([-3, -1]), mode='lines',
                                 line=dict(color=color_3, dash='dash'), showlegend=True,
                                 name='Score_3_points = {} %'.format('{:.1f}'.format(score_3)))
        lines.append(curve_3_pos)
    except Exception:
        import traceback
        traceback.print_exc()

    threshold_value = go.Scatter(x=x, y=np.ones_like(y) * 0.05, mode='lines', line=dict(color=('rgb(24, 12, 205)')),
                                 showlegend=False, hoverinfo='none')
    lines.append(threshold_value)

    figure = go.Figure(
        data=lines,
        layout=layout)

    return [dcc.Graph(id='table-editing-simple-output', figure=figure)]


if __name__ == '__main__':
    app.run_server(debug=True)
