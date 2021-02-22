import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff



#SL Survival Status
# 'Prepaid' coded as 1
# 'Default'/'Delinquent'/'Late' coded as 2
# 'Current' coded as 0
def one_two_or_zero(status):
    if (status == 'Fully Paid'):
        return 1
    elif ((status == 'Charged Off') | (status == 'Not Current')):
        return 2
    else:
        return 0
# 'Default'/'Delinquent'/'Late' coded as 1
# 'Current' coded as 0

def one_or_zero(status):
    if (status == 'Charged Off'): # Only defaulted loans are coded as event = 1
        return 1
    else:
        return 0 # Everything else coded as 0 (even prepaid)
#

def percent_good(df_, col):
    ret_string = []
    ret_data = []
    # Column Header for Data that follows in the for loop
    ret_string.append('{:22s}  {:>7s}  {:>7s}  {:>7s}  {:>13s}  {:>7s}\n'
            .format('Category', 'Good %', 'Bad %', 'Col %', 'Total', 'Int %'))
    val_ = df_[col].unique()
    #val_.sort()
    for item_ in np.sort(val_):                               # Item is the unique category item from the column (col)
        tot_ = df_[col].count()                                   # total loans with data for this column
        t_ = df_[df_[col]==item_][col].count()                     # Count of loans matching the category for this column
        c_ = df_[(df_[col]==item_) & (df_['good'])][col].count()  # Count of Good Loans
        b_ = t_ - c_                                              # Count of bad loans (total - good)
        ret_string.append(' {:24s}  {:8.1%}  {:8.1%}  {:8.1%}  {:14,.0f}  {:8.1%}\n'
              .format(str(item_), c_/t_, b_/t_, t_/tot_, t_,
                      df_[df_[col]==item_]['int_rate'].mean()/100))
        ret_data.append([str(item_), round(c_/t_*100,1), round(b_/t_*100,1), round(t_/tot_*100,1), t_,
                round(df_[df_[col]==item_]['int_rate'].mean(),1)])
    return ret_string, ret_data

def return_survival(df_,grade_items_value):
    total_count = df_[df_['months_of_pay']<41]['balance_b100'].count()
    y2 = 1-df_[df_['months_of_pay']<41].groupby(['months_of_pay'])['good'].count().cumsum()/total_count
    df_by_month = df_[df_['months_of_pay']<41].groupby(['grade', 'months_of_pay'])['good'].count().reset_index()
    fig = go.Figure()
    fig.add_scatter(x=y2.index, y=y2, mode='lines', name='All')
    fig.update_layout(showlegend=True)
    for  yy in grade_items_value:
        total_count = df_[(df_['months_of_pay']<41) & (df_['grade']==yy)]['balance_b100'].count()
        y=1-df_by_month[df_by_month['grade']==yy]['good'].cumsum()/total_count
        fig.add_scatter(x=df_by_month['months_of_pay'], y=y,mode='lines',name=yy)
    fig.update_layout({'plot_bgcolor': 'lightgrey',
                        'paper_bgcolor':'grey',
                        'title_x': 0.5,
                        'font': {'size': 20},
                        'title': 'Survival Plot'})
    return fig


def file_save(data, file):
    data.to_csv(file, mode = 'a', header = False)


def return_control(features, y_features, z_features):
    controls = dbc.FormGroup(
        [html.P('Year Selector', style={
                'textAlign': 'center'
            }),
            dcc.RangeSlider(
                id='year_slider',
                min=2009,
                max=2016,
                step=1,
                value=[2009, 2016],
                marks={i:i for i in range(2009,2016, 2)}
            ),

            html.Br(),
            html.P('Loan Grade', style={
                'textAlign': 'center'
            }),
            dbc.Card([dbc.Checklist(
                id='grade_items',
                options=[{
                    'label': 'Grade - A',
                    'value': 'A'
                },
                    {
                        'label': 'Grade - B',
                        'value': 'B'
                    },
                    {
                        'label': 'Grade - C',
                        'value': 'C'
                    },
                    {
                        'label': 'Grade - D',
                        'value': 'D'
                    },
                    {
                        'label': 'Grade - E',
                        'value': 'E'
                    },
                    {
                        'label': 'Grade - F',
                        'value': 'F'
                    },
                    {
                        'label': 'Grade - G',
                        'value': 'G'
                    }
                ],
                value=['B', 'C', 'D', 'E'],
                inline=False,
                style={
                    'margin': 'auto'
                }
            )]),
            html.Br(),
            html.P('Loan Term', style={
                'textAlign': 'center'
            }),
            dbc.Card([dbc.Checklist(
                id='term_items',
                options=[{
                    'label': 'Term - 36',
                    'value': 0
                },
                    {
                        'label': 'Term - 60',
                        'value': 1
                    }
                ],
                value=[0],
                inline=False,
                style={
                    'margin': 'auto'
                }
            )]
            ),
            html.Br(),
            html.P('X Axis - feature', style={
                'textAlign': 'center'
            }),

            dcc.Dropdown(
                        id='xaxis',
                        options=[{'label': i, 'value': i} for i in features],
                        value='sub_grade'
            ),
            html.Br(),
            html.P('Y Axis - feature', style={
                'textAlign': 'center'
            }),

            dcc.Dropdown(
                        id='yaxis',
                        options=[{'label': i, 'value': i} for i in y_features],
                        value='months_of_pay'
            ),
            html.Br(),
            html.P('Z Axis - feature', style={
                'textAlign': 'center'
            }),

            dcc.Dropdown(
                        id='zaxis',
                        options=[{'label': i, 'value': i} for i in z_features],
                        value='good'
            ),

            html.Br(),


            html.P('Top Pane', style={
                'textAlign': 'center'
            }),

            dbc.Card([dbc.RadioItems(
                id='radio_t',
                options=[{
                    'label': 'Bar Plot',
                    'value': 'BC_rd'
                },
                    {
                        'label': 'Density Plot',
                        'value': 'DC_rd'
                    },
                    {
                        'label': 'Scatter Plot',
                        'value': 'SC_rd'
                    },
                    {
                        'label': 'Stacked KDE Plot',
                        'value': 'KDE_rd'
                    },
                    {
                        'label': "Don't do it!",
                        'value': 'SSC_rd'
                    }
                ],
                value='BC_rd',
                inline=False
            )]),
            html.Br(),

            html.P('Left Pane', style={
                'textAlign': 'center'
            }),

            dbc.Card([dbc.RadioItems(
                id='radio_l',
                options=[{
                    'label': 'DataFrame',
                    'value': 'DF_rd'
                },
                    {
                        'label': 'Chart Of Awesomeness',
                        'value': 'CT_rd'
                    },
                    {
                        'label': 'Notes For This Page',
                        'value': 'NOTE_rd'
                    }
                ],
                value='DF_rd',
                inline=False
            )]),
            html.Br(),
            html.P('Right Pane', style={
                'textAlign': 'center'
            }),
            dbc.Card([dbc.RadioItems(
                id='radio_r',
                options=[{
                    'label': 'Sunburst Chart',
                    'value': 'sb_rd'
                },
                    {
                        'label': 'Yield Curve',
                        'value': 'yc_rd'
                    },
                    {
                        'label': 'Survival Curve',
                        'value': 'sc_rd'
                    }
                ],
                value='sb_rd',
                inline=False
            )]),
            html.Br(),
            dbc.Button(
                id='submit_button',
                n_clicks=0,
                children='Submit',
                color='primary',
                block=True
            ),
            #
            html.Br(),
            html.P('Selected Items:', style={
                'textAlign': 'center'

            }),
            dbc.Card([dbc.Label(
                id='years_lb',
                children='Under Construction',
                style={
                    'margin': 'auto',
                    'inline': False
                }
            ),
            dbc.Label(
                id='grade_lb',
                children='In for repairs',
                style={
                    'margin': 'auto',
                    'inline': False
                }
            )])####
        ]
    )
    return controls
###
