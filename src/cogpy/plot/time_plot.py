from cogpy.utils.imports import import_optional
import_optional("plotly")
from plotly.graph_objs import Layout, Scatter
from plotly.graph_objs.layout import YAxis, Annotation


def design_trace_layout(times, data, ch_names):
    """
    Parameters
    ----------
    data: array (channels, samples)
    times: array (samples, )
    ch_names: list of str (channels,)

    Returns
    ------
    traces
    layout
    """
    n_channels = data.shape[0]
    step = 1.0 / n_channels
    kwargs = dict(
        domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False
    )

    # create objects for layout and traces; give the data for the first channel
    layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
    traces = [Scatter(x=times, y=data[0])]

    # loop over the rest of the channels
    for ii in range(1, n_channels):
        kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
        layout.update({"yaxis%d" % (ii + 1): YAxis(kwargs), "showlegend": False})
        traces.append(Scatter(x=times, y=data[ii], yaxis="y%d" % (ii + 1)))

    # # add channel names using Annotations
    annotations = [
        Annotation(
            x=-0.06,
            y=0,
            xref="paper",
            yref="y%d" % (ii + 1),
            text=ch_name,
            showarrow=False,
        )
        for ii, ch_name in enumerate(ch_names)
    ]

    # Add range slider
    layout.update(
        xaxis=dict(rangeslider=dict(visible=True)), yaxis=dict(fixedrange=False)
    )

    # annotations = go.Annotations(ant)
    layout.update(annotations=annotations)

    # set the size of the figure and plot it
    layout.update(autosize=False, width=1000, height=1000)
    return traces, layout
