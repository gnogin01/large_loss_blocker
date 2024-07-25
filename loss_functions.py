import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly_helper_functions

def generate_loss_savings_dataframe(df, thresholds, model_name = None):
    results_dictionary = {}
    #thresholds = [round(x, 3) for x in np.arange(0.9, 1, 0.001)]

    # Perform a for loop to generate data
    for i in range(0, len(thresholds)):
        # Separate policies into two dataframes. Policies that would have been removed/kept in the book.
        df['blocked_and_bad'] = (df['prediction_percentile'] > thresholds[i]) & (df['underwriting_blocker'])
        df_removed_policies = df.loc[df['blocked_and_bad']]
        df_kept_poliices = df.loc[~df['blocked_and_bad']]

        # Claim counts for the kept policies
        claim_count = (df_kept_poliices['total_inc_loss'] > 0).sum()
        non_claim_count = (df_kept_poliices['total_inc_loss'] == 0).sum()
        claim_non_claim_ratio = claim_count / non_claim_count

        # Premium and losses for the removed policies
        earned_premium_lost = (df_removed_policies['written_aop_premium'] * df_removed_policies['sum_earned_exposure']).sum()
        losses_avoided = df_removed_policies['total_inc_loss'].sum()
        net_avoidance = losses_avoided - earned_premium_lost
        
        # Premium and losses for the kept policies
        counterfactual_earned_premium = (df_kept_poliices['written_aop_premium'] * df_kept_poliices['sum_earned_exposure']).sum()
        counterfactual_losses = df_kept_poliices['total_inc_loss'].sum()
        counterfactual_loss_ratio = counterfactual_losses / counterfactual_earned_premium
        
        # Percent of policies removed
        percent_policies_removed = np.round(df_removed_policies.shape[0]/df.shape[0], 2)
        
        # Save to dictionary
        results_dictionary[f'Run{i}'] = [
            thresholds[i], 
            earned_premium_lost, 
            losses_avoided, 
            net_avoidance, 
            counterfactual_earned_premium,  
            counterfactual_losses, 
            counterfactual_loss_ratio,
            percent_policies_removed, 
            claim_count, 
            non_claim_count, 
            claim_non_claim_ratio
        ] 
        
    # Convert dictionary to pandas dataframe
    df_summary = pd.DataFrame.from_dict(results_dictionary, orient = 'index', 
                                        columns = ['threshold', 
                                                   'earned_premium_lost', 'losses_avoided', 'net_avoidance', 
                                                   'counterfactual_earned_premium',  'counterfactual_losses', 'counterfactual_loss_ratio', 
                                                   'percent_policies_removed',
                                                   'claim_count', 'non_claim_count', 'claim_non_claim_ratio'])

    # Actual loss ratio for all policies
    df_summary['actual_loss_ratio'] = df['total_inc_loss'].sum() / (df['written_aop_premium'] * df['sum_earned_exposure']).sum()
    
    # Model name
    df_summary['model_name'] = model_name

    return df_summary


def generate_traces(fig, df, plot_type, visible):
    num_traces = 0
    
    models = df['model_name'].drop_duplicates().tolist()
    df_models = df
    color_list = px.colors.qualitative.Prism
    
    for i in range(0, len(models)):
        df = df_models[df_models['model_name'] == models[i]]
        
        ## Claims Plots ##
        if plot_type == 'Claims':
            num_traces = num_traces + 1
            trace1 = go.Scattergl(
                x = df['threshold'],
                y = df['claim_non_claim_ratio'],
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Ratio of Claims to Non-Claims: ' + round(df['claim_non_claim_ratio'], 3).astype(str) + '<br>' + '<br>' + \
                             'Number of Claims: ' + df['claim_count'].astype(str) + '<br>' + \
                             'Number of Non-Claims: ' + df['non_claim_count'].astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Ratio of Claims <br>' + '  to Non-Claims'
            )
            fig.add_trace(trace1, row = 1, col = 1)

            num_traces = num_traces + 1
            trace2 = go.Scattergl(
                x = df['threshold'],
                y = df['claim_count'],
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Number of Claims: ' + df['claim_count'].astype(str) + '<br>' + '<br>' + \
                             'Number of Non-Claims: ' + df['non_claim_count'].astype(str) + '<br>' + \
                             'Ratio of Claims to Non-Claims: ' + round(df['claim_non_claim_ratio'], 3).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Claim Count'
            )
            fig.add_trace(trace2, row = 2, col = 1)

            num_traces = num_traces + 1
            trace3 = go.Scattergl(
                x = df['threshold'],
                y = df['non_claim_count'],
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Number of Non-Claims: ' + df['non_claim_count'].astype(str) + '<br>' + '<br>' + \
                             'Number of Claims: ' + df['claim_count'].astype(str) + '<br>' + \
                             'Ratio of Claims to Non-Claims: ' + round(df['claim_non_claim_ratio'], 3).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Non-Claim Count'
            )
            fig.add_trace(trace3, row = 2, col = 2)
            
        ## Avoidance Plots ##
        if plot_type == 'Avoidance':
            num_traces = num_traces + 1
            trace1 = go.Scattergl(
                x = df['threshold'],
                y = df['net_avoidance'],
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Net Avoidance: ' + df['net_avoidance'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + '<br>' + \
                             'Loss Avoided: ' + df['losses_avoided'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + \
                             'Premium Lost: ' + df['earned_premium_lost'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Net Avoidance'
            )
            fig.add_trace(trace1, row = 1, col = 1)

            num_traces = num_traces + 1
            trace2 = go.Scattergl(
                x = df['threshold'],
                y = df['losses_avoided'],
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Loss Avoided: ' + df['losses_avoided'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + '<br>' + \
                             'Net Avoidance: ' + df['net_avoidance'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + \
                             'Premium Lost: ' + df['earned_premium_lost'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Losses Avoided'
            )
            fig.add_trace(trace2, row = 2, col = 1)

            num_traces = num_traces + 1
            trace3 = go.Scattergl(
                x = df['threshold'],
                y = df['earned_premium_lost'],
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Premium Lost: ' + df['earned_premium_lost'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + '<br>' + \
                             'Loss Avoided: ' + df['losses_avoided'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + \
                             'Net Avoidance: ' +  df['net_avoidance'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Premium Lost'
            )
            fig.add_trace(trace3, row = 2, col = 2)

        ## Counterfactual Book ##
        if plot_type == 'Counterfactual <br> Book':
            num_traces = num_traces + 1
            trace1 = go.Scattergl(
                x = df['threshold'],
                y = df['counterfactual_earned_premium'],
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Counterfactual Earned Premium: ' + df['counterfactual_earned_premium'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + '<br>' + \
                             'Counterfactual Losses: ' + df['counterfactual_losses'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + \
                             'Percent Policies Removed: ' + round(df['percent_policies_removed'] * 100, 3).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Counterfactual <br>' + '  Earned Premium'
            )
            fig.add_trace(trace1, row = 1, col = 1)

            num_traces = num_traces + 1
            trace2 = go.Scattergl(
                x = df['threshold'],
                y = df['counterfactual_losses'],
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Counterfactual Losses: ' + df['counterfactual_losses'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + '<br>' + \
                             'Counterfactual Earned Premium: ' + df['counterfactual_earned_premium'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + \
                             'Percent Policies Removed: ' + round(df['percent_policies_removed'] * 100, 3).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Counterfactual <br>' + '  Losses'
            )
            fig.add_trace(trace2, row = 2, col = 1)

            num_traces = num_traces + 1
            trace3 = go.Scattergl(
                x = df['threshold'],
                y = df['percent_policies_removed'] * 100,
                hovertext = ('Model: ' + df['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df['threshold'].astype(str) + '<br>' + \
                             'Percent Policies Removed: ' + round(df['percent_policies_removed'] * 100, 3).astype(str) + '<br>' + '<br>' + \
                             'Counterfactual Earned Premium: ' + df['counterfactual_earned_premium'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str) + '<br>' + \
                             'Counterfactual Losses: ' + df['counterfactual_losses'].apply(lambda x: "${:.3f}M".format((x/1000000))).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 4, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = visible,
                legendgroup = models[i],
                name = models[i] + ': <br>' + '  Percent of <br>' + '  Policies Removed'
            )
            fig.add_trace(trace3, row = 2, col = 2)
    
    return num_traces, fig


def make_multi_plot(df_loss):
    # Initialize figure
    fig = make_subplots(rows = 2, cols = 2, shared_xaxes = True, vertical_spacing = 0.2,
                        specs=[[{"rowspan": 1, "colspan": 2}, None],[{}, {}]],
                        subplot_titles = ('<b>Ratio of Claims to Non-Claims</b>', '<b>Number of Claims</b>', '<b>Number of Non-Claims</b>'))
    
    # Generate traces for plots
    total_traces = 0
    num_traces_vector = []
    
    plot_types = ['Claims', 'Avoidance', 'Counterfactual <br> Book']
    titles = ['<b>Claims Summary</b>', '<b>Avoidance Summary</b>', '<b>Summary of Counterfactual Book</b>']
    subplot_titles = [['<b>Ratio of Claims to Non-Claims</b>', '<b>Number of Claims</b>', '<b>Number of Non-Claims</b>'],
                      ['<b>Net Avoidance</b>', '<b>Losses Avoided</b>', '<b>Premium Lost</b>'],
                      ['<b>Counterfactual Losses</b>', '<b>Counterfactual Earned Premium</b>', '<b>Percent of Policies Removed</b>']]
    xaxis_titles = [['<b>Threshold</b>', '<b>Threshold</b>', '<b>Threshold</b>'], 
                    ['<b>Threshold</b>', '<b>Threshold</b>', '<b>Threshold</b>'], 
                    ['<b>Threshold</b>', '<b>Threshold</b>', '<b>Threshold</b>']] 
    yaxis_titles = [['<b>Ratio</b>', '<b>Number of Claims</b>', '<b>Number of Non-Claims</b>'], 
                    ['<b>Net Avoidance ($)</b>', '<b> Losses Avoided ($)</b>', '<b>Premium Lost ($)</b>'], 
                    ['<b>Losses ($)</b>', '<b>Premium Earned ($)</b>', '<b>Percent (%)</b>']]
    
    for i in range(0, len(plot_types)):
        visible = True if i == 0 else False
        num_traces, fig = generate_traces(fig, df_loss, plot_types[i], visible)
        
        total_traces = total_traces + num_traces
        num_traces_vector.append(num_traces)
        
    # Buttons
    start = 0
    end = 0
    buttons = []
    for i in range(0, len(plot_types)):
        # Mapping the traces to the buttons
        end = end + num_traces_vector[i]
        id_on = [False] * total_traces
        id_on[start:end] = [True] * (end - start)
        start = start + num_traces_vector[i]
        
        buttons.append(dict(
            label = plot_types[i],
            method = 'update',
            args = [{'visible': id_on},
                    {'title': titles[i],
                     'xaxis.title.text': xaxis_titles[i][0],
                     'xaxis2.title.text': xaxis_titles[i][1],
                     'xaxis3.title.text': xaxis_titles[i][2],
                     'yaxis.title.text': yaxis_titles[i][0],
                     'yaxis2.title.text': yaxis_titles[i][1],
                     'yaxis3.title.text': yaxis_titles[i][2],
                     'annotations': [dict(xref = 'paper', yref = 'paper', showarrow = False, x = 0.5, y = 1.037, font = dict(size = 16), text = subplot_titles[i][0]),
                                     dict(xref = 'paper', yref = 'paper', showarrow = False, x = 0.07, y = 0.418, font = dict(size = 16), text = subplot_titles[i][1]),
                                     dict(xref = 'paper', yref = 'paper', showarrow = False, x = 0.9, y = 0.418, font = dict(size = 16), text = subplot_titles[i][2])
                                    ]}]
        ))
        
    # Plot layout
    menu_type = 'buttons' if len(plot_types) <= 10 else 'dropdown'
    
    layout = plotly_helper_functions.make_layout(menu_type = menu_type, buttons = buttons, xtitle = xaxis_titles[0][0], ytitle = yaxis_titles[0][0], title = titles[0])
    fig.update_layout(layout)
    fig.update_layout(hovermode = 'x unified', 
                      xaxis2 = plotly_helper_functions.make_axis(title = xaxis_titles[0][1]), xaxis3 = plotly_helper_functions.make_axis(title = xaxis_titles[0][2]),
                      yaxis2 = plotly_helper_functions.make_axis(title = yaxis_titles[0][1]), yaxis3 = plotly_helper_functions.make_axis(title = yaxis_titles[0][2]))
    
    fig.update_xaxes(matches = 'x')
    fig.update_yaxes(rangemode = 'tozero')
    
    return fig


def make_loss_ratio_plot(df):
    # Initialize figure
    fig = go.Figure()

    # Create traces
    models = df['model_name'].drop_duplicates().tolist()
    color_list = px.colors.qualitative.Prism
    
    for i in range(0, len(models)):
        df_model = df[df['model_name'] == models[i]]
        
        trace1 = go.Scattergl(
                x = df_model['threshold'],
                y = df_model['counterfactual_loss_ratio'],
                hovertext = ('Model: ' + df_model['model_name'].astype(str) + '<br>' + \
                             'Threshold: ' + df_model['threshold'].astype(str) + '<br>' + \
                             'Counterfactual Loss Ratio: ' + round(df_model['counterfactual_loss_ratio'], 3).astype(str)).tolist(),
                hoverinfo = 'text',
                mode = 'lines+markers',
                marker_size = 6, 
                marker_color = color_list[i],
                line_color = color_list[i],
                opacity = 0.65,
                visible = True,
                legendgroup = models[i],
                name = models[i]
            )
        fig.add_trace(trace1)
    
    # Plot layout
    layout = plotly_helper_functions.make_layout(menu_type = 'buttons', buttons = [], 
                                                 xtitle = '<b>Threshold</b>', ytitle = '<b>Loss Ratio</b>', title = '<b>Loss Ratio for Various Models</b>')
    fig.update_layout(layout)
    
    return fig