import pandas as pd
import plotly.express as px
import numpy as np


COLOR_COL = "degree_of_uncertainty"
BAR_COL = "speaker_classes_of_cluster"
BAR_SEGMENT_COL = "author"


def plot_diarization(df: pd.DataFrame):
    # Aggregating data to count the number of rows per 'bar' and 'bar_segment'
    # and to calculate the mean 'color' per 'bar_segment'.
    agg_df = df.groupby([BAR_COL, BAR_SEGMENT_COL]).agg(
        num_utterances=(BAR_COL, 'size'),
        mean_uncertainty=(COLOR_COL, 'mean')
    ).reset_index()

    # Creating the color scale for the bars where 0 is green and 1 is red.
    color_scale = agg_df['mean_uncertainty'].apply(lambda x: (1 - x, x, 0))

    # Plotting the interactive bar chart using Plotly
    fig = px.bar(
        agg_df, 
        x=BAR_COL, 
        y="num_utterances", 
        color="mean_uncertainty",
        color_continuous_scale=["green", "red"], 
        text=BAR_SEGMENT_COL,
    )

    # Customizing hover data
    fig.update_traces(
        hovertemplate='<b>%{text}</b><br>Num Utterances: %{y}<br>Mean Uncertainty: %{marker.color:.2f}'
    )

    # Customizing the layout to wrap the x-axis text and to adjust for up to 18 bars
    fig.update_layout(
        title='Identified Speaker Clusters Split By Book Author Regions',
        yaxis_title='Number of Utterances',
        xaxis_title="Identified Speaker Cluster",
        xaxis={'tickangle':-45},
        xaxis_tickangle=-45,
        xaxis_tickmode='array',
        xaxis_tickvals=agg_df[BAR_COL],
        xaxis_ticktext=[label.replace('\n', '<br>') for label in agg_df[BAR_COL]],
        coloraxis_showscale=False,
        margin=dict(b=100)  # Bottom margin to ensure x-axis labels are visible
    )


    # Updating the width to fit up to 18 bars. Assuming 100px per bar.
    fig.update_layout(width=1350)

    # Show the figure
    fig.show()
