import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_title(title):
    title = dict(
        text = title,
        xref = 'paper',
        x = 0
    )
    return title

def make_axis(title):
    axis = dict(
        title = title,
        tickmode = 'array',
        nticks = 10,
        titlefont = dict(color = 'black'),
        showticklabels = True,
        showline = True,
        tickangle = 0,
        tickfont = dict(color = 'black')
    )
    return axis

def make_updatemenus(menu_type, buttons):
    updatemenus = [dict(
        type = menu_type,
        font = dict(color = 'black'),
        bgcolor = 'white',
        direction = 'down',
        active = 0,
        xanchor = 'left',
        x = -0.2,
        yanchor = 'top',
        y = 1.0,
        buttons = buttons
    )]
    return updatemenus
    
def make_layout(menu_type, buttons, xtitle, ytitle, title):
    layout = go.Layout(
        template = 'plotly_white',
        title = make_title(title),
        xaxis = make_axis(xtitle),
        yaxis = make_axis(ytitle),
        updatemenus = make_updatemenus(menu_type, buttons),
        autosize = True,
        width = 1600,
        height = 800,
        hovermode = 'closest'
    )
    return layout