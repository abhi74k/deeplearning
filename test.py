import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Sample data
data_heatmap = np.random.rand(10, 20)
data_lineplot = np.random.rand(20)

app = dash.Dash(__name__)

# Initial selected column
selected_column = 0

app.layout = html.Div([
    dcc.Graph(id='heatmap-lineplot', style={"display": "inline-block"},
              config={'staticPlot': False, 'displayModeBar': True}),
    dcc.Graph(id='bar-plot', style={"display": "inline-block"}),

    html.Button('Previous', id='btn-prev', style={"display": "inline-block"}),
    html.Button('Next', id='btn-next', style={"display": "inline-block"}),

    # Hidden div storing selected column
    html.Div(id='selected-column', style={'display': 'none'}, children=selected_column)
])

@app.callback(
    [Output('heatmap-lineplot', 'figure'),
     Output('bar-plot', 'figure'),
     Output('selected-column', 'children')],
    [Input('btn-prev', 'n_clicks'),
     Input('btn-next', 'n_clicks'),
     Input('heatmap-lineplot', 'clickData'),
     Input('selected-column', 'children')]
)
def update_plots(prev_clicks, next_clicks, clickData, selected_column):
    selected_column = int(selected_column)

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-prev':
        selected_column = max(0, selected_column - 1)
    elif button_id == 'btn-next':
        selected_column = min(data_heatmap.shape[1] - 1, selected_column + 1)
    elif button_id == 'heatmap-lineplot' and clickData:
        # Clicked x-coordinate to determine the column
        clicked_column = int(clickData['points'][0]['x'])
        selected_column = clicked_column

    heatmap = {
        'data': [{
            'z': data_heatmap,
            'type': 'heatmap',
            'colorscale': 'Blues',
            'showscale': False
        },
        {
            'x': np.arange(20),
            'y': data_lineplot * 10,
            'type': 'scatter',
            'line': {'color': 'red'}
        }],
        'layout': {
            'shapes': [{
                'type': 'line',
                'x0': selected_column,
                'x1': selected_column,
                'y0': 0,
                'y1': 10,
                'line': {
                    'color': 'yellow',
                    'width': 2,
                    'dash': 'dash'
                }
            }]
        }
    }

    selected_values = data_heatmap[:, selected_column]
    max_value = np.max(data_heatmap[:, selected_column])
    inverse_values = max_value - selected_values

    # Stacked bar plot
    bar_plot = {
        'data': [
            {
                'x': selected_values,
                'y': list(range(10)),
                'type': 'bar',
                'orientation': 'h',
                'name': 'Value'
            },
            {
                'x': inverse_values,
                'y': list(range(10)),
                'type': 'bar',
                'orientation': 'h',
                'name': 'Inverse Value'
            }
        ],
        'layout': {
            'yaxis': {'tickvals': list(range(10))},
            'barmode': 'stack'
        }
    }

    return heatmap, bar_plot, selected_column


if __name__ == '__main__':
    app.run_server(debug=True)
