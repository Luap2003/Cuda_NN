import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import os
import glob
from dash.exceptions import PreventUpdate
import numpy as np

def get_log_folders(main_dir):
    """
    Retrieve all subdirectories in the main logs directory.
    
    Args:
        main_dir (str): Path to the main logs directory.
        
    Returns:
        list: List of subdirectory names.
    """
    if not os.path.exists(main_dir):
        raise FileNotFoundError(f"Main logs directory not found at path: {main_dir}")
    
    # List all subdirectories in the main logs directory
    folders = [name for name in os.listdir(main_dir)
               if os.path.isdir(os.path.join(main_dir, name))]
    return folders

def process_logs(directory, framework):
    """
    Read and concatenate all CSV log files for a given framework within a directory.
    
    Args:
        directory (str): Path to the log directory.
        framework (str): Framework identifier ('cuda' or 'tf').
        
    Returns:
        pd.DataFrame: Concatenated DataFrame of all logs for the specified framework.
    """
    logs = []
    # Use glob to match patterns for flexibility
    pattern = os.path.join(directory, f"training_log_{framework}*.csv")
    for file in glob.glob(pattern):
        try:
            df = pd.read_csv(file)
            df['Epoch'] = df['Epoch'].astype(int)
            df['Framework'] = framework.upper() if framework.lower() == 'cuda' else 'TensorFlow'
            logs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if logs:
        return pd.concat(logs, ignore_index=True)
    else:
        return pd.DataFrame()

def create_empty_fig(message):
    """
    Create an empty Plotly figure with a centered message.
    
    Args:
        message (str): Message to display.
        
    Returns:
        plotly.graph_objs.Figure: Empty figure with annotation.
    """
    return {
        'data': [],
        'layout': go.Layout(
            title=message,
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[
                dict(
                    text=message,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(size=20)
                )
            ]
        )
    }


# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For deployment

# Define the main logs directory
MAIN_LOGS_DIR = '../logs'  # Ensure this path is correct relative to app.py

# Retrieve available log folders
available_log_folders = get_log_folders(MAIN_LOGS_DIR)

if not available_log_folders:
    raise ValueError(f"No log folders found in the main logs directory: {MAIN_LOGS_DIR}")


app.layout = html.Div([
    html.H1("Training Logs Dashboard", style={'textAlign': 'center'}),
    
    # Dropdown for selecting Log Folder Setup
    html.Div([
        html.Label("Select Log Setup:"),
        dcc.Dropdown(
            id='log-setup-dropdown',
            options=[{'label': folder, 'value': folder} for folder in available_log_folders],
            value=available_log_folders[0],  # Default to the first available folder
            multi=False,
            clearable=False
        )
    ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),
    
    # Dropdown for selecting Framework
    html.Div([
        html.Label("Select Framework:"),
        dcc.Dropdown(
            id='framework-dropdown',
            options=[
                {'label': 'CUDA', 'value': 'cuda'},
                {'label': 'TensorFlow', 'value': 'tf'},
                {'label': 'Both', 'value': 'Both'}
            ],
            value='Both',
            multi=False,
            clearable=False
        )
    ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),
    
    # Graphs
    html.Div([
        # First Row
        html.Div([
            dcc.Graph(id='loss-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='accuracy-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        # Second Row
        html.Div([
            dcc.Graph(id='epoch-time-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='batches-per-sec-graph')
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    html.Div([
        # New Weights Graph
    html.Div([
        dcc.Graph(id='weights-graph')
    ], style={'width': '50%', 'display': 'inline-block'}),
        # New Biases Graph
    html.Div([
        dcc.Graph(id='biases-graph')
    ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
])

@app.callback(
    [
        Output('loss-graph', 'figure'),
        Output('accuracy-graph', 'figure'),
        Output('epoch-time-graph', 'figure'),
        Output('batches-per-sec-graph', 'figure')
    ],
    [
        Input('log-setup-dropdown', 'value'),
        Input('framework-dropdown', 'value')
    ]
)
def update_graphs(selected_log_setup, selected_framework):
    """
    Update all four graphs based on the selected log setup and framework.
    
    Args:
        selected_log_setup (str): Selected log folder.
        selected_framework (str): Selected framework ('cuda', 'tf', 'Both').
        
    Returns:
        tuple: Four Plotly figures.
    """
    log_dir = os.path.join(MAIN_LOGS_DIR, selected_log_setup)
    
    # Determine frameworks to process
    frameworks = []
    if selected_framework.lower() == 'both':
        frameworks = ['cuda', 'tf']
    else:
        frameworks = [selected_framework.lower()]
    
    # Initialize empty DataFrame
    combined_df = pd.DataFrame()
    
    # Process each selected framework
    for fw in frameworks:
        df = process_logs(log_dir, fw)
        if not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    if combined_df.empty:
        message = "No data available for the selected options."
        empty_fig = create_empty_fig(message)
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # Ensure DataFrame is sorted by Framework and Epoch
    combined_df = combined_df.sort_values(['Framework', 'Epoch'])
    
    # Group by Framework and Epoch to compute averages and standard deviations
    grouped = combined_df.groupby(['Framework', 'Epoch']).agg({
        'Loss': ['mean', 'std'],
        'Accuracy%': ['mean', 'std'],
        'Epoch Time(s)': ['mean', 'std'],
        'Batches/s': ['mean', 'std']
    }).reset_index()
    
    # Flatten column multi-index
    grouped.columns = ['Framework', 'Epoch',
                       'Loss_mean', 'Loss_std',
                       'Accuracy_mean', 'Accuracy_std',
                       'EpochTime_mean', 'EpochTime_std',
                       'Batches_mean', 'Batches_std']
    
    # Filter based on selected framework
    if selected_framework.lower() != 'both':
        df_grouped = grouped[grouped['Framework'] == selected_framework.upper() if selected_framework.lower() == 'cuda' else 'TensorFlow']
    else:
        df_grouped = grouped.copy()
    
    if df_grouped.empty:
        message = "No data available after filtering."
        empty_fig = create_empty_fig(message)
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # Initialize Plotly figures
    loss_fig = go.Figure()
    accuracy_fig = go.Figure()
    epoch_time_fig = go.Figure()
    batches_sec_fig = go.Figure()
    
    # Helper function to add traces with error bands
    def add_trace_with_error(fig, df, y_mean, y_std, framework_label, marker_symbol):
        fig.add_trace(go.Scatter(
            x=df['Epoch'],
            y=df[y_mean],
            mode='lines+markers',
            name=f'{framework_label} Mean',
            marker=dict(symbol=marker_symbol),
            line=dict(shape='linear')
        ))
        fig.add_trace(go.Scatter(
            x=df['Epoch'],
            y=df[y_mean] - df[y_std],
            mode='lines',
            name=f'{framework_label} Lower',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df['Epoch'],
            y=df[y_mean] + df[y_std],
            mode='lines',
            name=f'{framework_label} Upper',
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(width=0),
            showlegend=False
        ))
    
    # Plot 1: Training Loss vs Epoch
    for framework in df_grouped['Framework'].unique():
        fw_data = df_grouped[df_grouped['Framework'] == framework]
        marker = 'circle' if framework == 'CUDA' else 'square'
        add_trace_with_error(loss_fig, fw_data, 'Loss_mean', 'Loss_std', framework, marker)
    
    loss_fig.update_layout(
        title='Training Loss vs Epoch',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified'
    )
    
    
    # Plot 2: Training Accuracy vs Epoch
    for framework in df_grouped['Framework'].unique():
        fw_data = df_grouped[df_grouped['Framework'] == framework]
        marker = 'square' if framework == 'CUDA' else 'diamond'
        add_trace_with_error(accuracy_fig, fw_data, 'Accuracy_mean', 'Accuracy_std', framework, marker)
    
    accuracy_fig.update_layout(
        title='Training Accuracy vs Epoch',
        xaxis_title='Epoch',
        yaxis_title='Accuracy (%)',
        hovermode='x unified'
    )
    
    # Plot 3: Epoch Time vs Epoch
    for framework in df_grouped['Framework'].unique():
        fw_data = df_grouped[df_grouped['Framework'] == framework]
        marker = 'triangle-up' if framework == 'CUDA' else 'triangle-down'
        add_trace_with_error(epoch_time_fig, fw_data, 'EpochTime_mean', 'EpochTime_std', framework, marker)
    
    epoch_time_fig.update_layout(
        title='Epoch Time vs Epoch',
        xaxis_title='Epoch',
        yaxis_title='Epoch Time (s)',
        hovermode='x unified'
    )
    
    # Plot 4: Batches per Second vs Epoch
    for framework in df_grouped['Framework'].unique():
        fw_data = df_grouped[df_grouped['Framework'] == framework]
        marker = 'diamond' if framework == 'CUDA' else 'hexagon'
        add_trace_with_error(batches_sec_fig, fw_data, 'Batches_mean', 'Batches_std', framework, marker)
    
    batches_sec_fig.update_layout(
        title='Batches per Second vs Epoch',
        xaxis_title='Epoch',
        yaxis_title='Batches/s',
        hovermode='x unified'
    )
    
    return loss_fig, accuracy_fig, epoch_time_fig, batches_sec_fig


def process_weights(directory, framework):
    """
    Read and parse the weights file for a given framework within a directory.
    
    Args:
        directory (str): Path to the log directory.
        framework (str): Framework identifier ('cuda' or 'tf').
        
    Returns:
        pd.DataFrame: DataFrame with Framework, Epoch, Weight_Index, and Weight_Value.
    """
    weights_data = []
    # Define the weights file pattern; adjust as per your naming convention
    # Example: weights_cuda.txt or weights_tf.txt
    if framework.lower() == 'both':
        frameworks = ['cuda', 'tf']
    else:
        frameworks = [framework.lower()]
    
    for fw in frameworks:
        # Adjust the filename pattern as needed
        weights_file_pattern = os.path.join(directory, f"weights_log_{fw}.txt")
        if not os.path.exists(weights_file_pattern):
            print(f"Weights file not found: {weights_file_pattern}")
            continue
        try:
            with open(weights_file_pattern, 'r') as f:
                next(f)  # Skip the header line
                for line_number, line in enumerate(f, start=2):  # start=2 considering header is line 1
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        print(f"Malformed line at {weights_file_pattern}:{line_number}")
                        continue  # Skip malformed lines
                    epoch_str = parts[0].strip()
                    if not epoch_str.isdigit():
                        print(f"Invalid epoch value '{epoch_str}' at {weights_file_pattern}:{line_number}")
                        continue
                    epoch = int(epoch_str)
                    weights = []
                    for idx, weight_str in enumerate(parts[1:], start=0):
                        weight_str = weight_str.strip()
                        if weight_str == '':
                            continue  # Skip empty strings resulting from trailing commas
                        try:
                            weight = float(weight_str)
                            weights.append(weight)
                        except ValueError:
                            print(f"Invalid weight value '{weight_str}' at {weights_file_pattern}:{line_number}, index {idx}")
                            continue
                    for idx, weight in enumerate(weights):
                        weights_data.append({
                            'Framework': fw.upper() if fw.lower() == 'cuda' else 'TensorFlow',
                            'Epoch': epoch,
                            'Weight_Index': idx,
                            'Weight_Value': weight
                        })
        except Exception as e:
            print(f"Error reading weights file {weights_file_pattern}: {e}")
    
    if weights_data:
        return pd.DataFrame(weights_data)
    else:
        return pd.DataFrame()

# Callback to update the weights graph
@app.callback(
    Output('weights-graph', 'figure'),
    [
        Input('log-setup-dropdown', 'value'),
        Input('framework-dropdown', 'value')
    ]
)
def update_weights_graph(selected_log_setup, selected_framework):
    """
    Update the 3D Weights graph to show the distribution of weight values over epochs.
    
    The x-axis represents binned weight values, 
    the y-axis represents epochs, and 
    the z-axis represents the frequency of weights in each bin.
    
    Args:
        selected_log_setup (str): Selected log folder.
        selected_framework (str): Selected framework ('cuda', 'tf', 'Both').
        
    Returns:
        plotly.graph_objs.Figure: 3D surface plot of weight distributions.
    """
    log_dir = os.path.join(MAIN_LOGS_DIR, selected_log_setup)
    
    # Process weights into a DataFrame
    weights_df = process_weights(log_dir, selected_framework)
    
    if weights_df.empty:
        message = "No weights data available for the selected options."
        return create_empty_fig(message)
    
    # Determine histogram bins based on all weight values across epochs
    all_weights = weights_df['Weight_Value'].dropna().values
    # Using 'auto' to determine an optimal number of bins; adjust as needed
    bins = np.histogram_bin_edges(all_weights, bins='auto')
    
    # Get sorted unique epochs
    epochs = sorted(weights_df['Epoch'].unique())
    
    # Initialize list to store histogram counts for each epoch
    hist_data = []
    
    # Compute histogram for each epoch
    for epoch in epochs:
        epoch_weights = weights_df.loc[weights_df['Epoch'] == epoch, 'Weight_Value'].dropna().values
        # Compute histogram counts for current epoch using predefined bins
        hist_counts, _ = np.histogram(epoch_weights, bins=bins)
        hist_data.append(hist_counts)
    
    # Convert list of histograms to a 2D array where each row corresponds to an epoch
    hist_array = np.array(hist_data)  # Shape: (number_of_epochs, number_of_bins)
    
    # Compute bin centers for the x-axis
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Create the 3D Surface Plot
    weights_fig = go.Figure(data=[
        go.Surface(
            x=bin_centers,   # Weight value bins (x-axis)
            y=epochs,        # Epochs (y-axis)
            z=hist_array,    # Frequency counts (z-axis)
            colorscale='Viridis',
            colorbar=dict(title='Frequency')
        )
    ])
    
    weights_fig.update_layout(
        title='Distribution of Weight Values Over Epochs',
        scene=dict(
            xaxis_title='Weight Value',
            yaxis_title='Epoch',
            zaxis_title='Frequency'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return weights_fig

def process_biases(directory, framework):
    """
    Read and parse the biases file for a given framework within a directory.
    
    Args:
        directory (str): Path to the log directory.
        framework (str): Framework identifier ('cuda', 'tf', or 'Both').
        
    Returns:
        pd.DataFrame: DataFrame with Framework, Epoch, Bias_Index, and Bias_Value.
    """
    biases_data = []
    if framework.lower() == 'both':
        frameworks = ['cuda', 'tf']
    else:
        frameworks = [framework.lower()]
    
    for fw in frameworks:
        biases_file_pattern = os.path.join(directory, f"biases_log_{fw}.txt")
        if not os.path.exists(biases_file_pattern):
            print(f"Biases file not found: {biases_file_pattern}")
            continue
        try:
            with open(biases_file_pattern, 'r') as f:
                next(f)  # Skip header
                for line_number, line in enumerate(f, start=2):
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        print(f"Malformed line at {biases_file_pattern}:{line_number}")
                        continue
                    epoch_str = parts[0].strip()
                    if not epoch_str.isdigit():
                        print(f"Invalid epoch value '{epoch_str}' at {biases_file_pattern}:{line_number}")
                        continue
                    epoch = int(epoch_str)
                    biases = []
                    for idx, bias_str in enumerate(parts[1:], start=0):
                        bias_str = bias_str.strip()
                        if bias_str == '':
                            continue
                        try:
                            bias = float(bias_str)
                            biases.append(bias)
                        except ValueError:
                            print(f"Invalid bias value '{bias_str}' at {biases_file_pattern}:{line_number}, index {idx}")
                            continue
                    for idx, bias in enumerate(biases):
                        biases_data.append({
                            'Framework': fw.upper() if fw.lower() == 'cuda' else 'TensorFlow',
                            'Epoch': epoch,
                            'Bias_Index': idx,
                            'Bias_Value': bias
                        })
        except Exception as e:
            print(f"Error reading biases file {biases_file_pattern}: {e}")
    
    if biases_data:
        return pd.DataFrame(biases_data)
    else:
        return pd.DataFrame()

@app.callback(
    Output('biases-graph', 'figure'),
    [
        Input('log-setup-dropdown', 'value'),
        Input('framework-dropdown', 'value')
    ]
)
def update_biases_graph(selected_log_setup, selected_framework):
    """
    Update the 3D Biases graph to show the distribution of bias values over epochs.
    
    Args:
        selected_log_setup (str): Selected log folder.
        selected_framework (str): Selected framework ('cuda', 'tf', 'Both').
        
    Returns:
        plotly.graph_objs.Figure: 3D surface plot of bias distributions.
    """
    log_dir = os.path.join(MAIN_LOGS_DIR, selected_log_setup)
    
    # Process biases into a DataFrame
    biases_df = process_biases(log_dir, selected_framework)
    
    if biases_df.empty:
        message = "No biases data available for the selected options."
        return create_empty_fig(message)
    
    # Determine histogram bins based on all bias values across epochs
    all_biases = biases_df['Bias_Value'].dropna().values
    bins = np.histogram_bin_edges(all_biases, bins='auto')
    
    # Get sorted unique epochs
    epochs = sorted(biases_df['Epoch'].unique())
    
    # Compute histogram for each epoch
    hist_data = []
    for epoch in epochs:
        epoch_biases = biases_df.loc[biases_df['Epoch'] == epoch, 'Bias_Value'].dropna().values
        hist_counts, _ = np.histogram(epoch_biases, bins=bins)
        hist_data.append(hist_counts)
    
    # Convert list of histograms to a 2D array where each row corresponds to an epoch
    hist_array = np.array(hist_data)
    
    # Compute bin centers for the x-axis
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Create the 3D Surface Plot for biases
    biases_fig = go.Figure(data=[
        go.Surface(
            x=bin_centers,   # Bias value bins (x-axis)
            y=epochs,        # Epochs (y-axis)
            z=hist_array,    # Frequency counts (z-axis)
            colorscale='Viridis',
            colorbar=dict(title='Frequency')
        )
    ])
    
    biases_fig.update_layout(
        title='Distribution of Bias Values Over Epochs',
        scene=dict(
            xaxis_title='Bias Value',
            yaxis_title='Epoch',
            zaxis_title='Frequency'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return biases_fig

if __name__ == '__main__':
    app.run_server(debug=True)
