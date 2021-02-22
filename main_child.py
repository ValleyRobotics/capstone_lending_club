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
###
# New Layout sidebar ######
#css
# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'font-size':'140%'#,
    #'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970',
    'font-size':'140%'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

CARD_TEXT_STYLE2 = {
    'textAlign': 'left',
    'color': 'white',
    'font-size':'145%'
}
TITLE_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9',
    'font-size':'200%'
}
TAB_STYLE = {
    'textAlign': 'center',
    'color': 'green',
    'background-color':'yellow',
    'font-size':'150%'
}
###
def main_child():
    return [
        html.H1('Capstone Project', style=TITLE_TEXT_STYLE),
        html.H2('Lending Club Business Model - FinTech'),
        dbc.Card([html.P(
            'The Lending Club  (LC) offers personal loans up to $40,000.  They fund \
            these loans through investors that sign up on the same site to invest in loans.  The investers\
            can select loans from 6 different grades (A-G), and invest from $25 into any of the loans they\
            want to fund. The grades are further divided by sub grades (1-5, ie A1, A2, ..., G5).\
            The initial grade is based mostly on credit report information while uses a point system to account for\
            more loan specific risks factors (ie Loan Amount, Number of months, etc...).\
            the LC takes a cut off the top (origination fee) and a 1% service fee on all monthly loan payments.  They\
            also charge a collection fee on all deliquent loans they need to collect. Thier main service\
             is matching those wishing to borrow loans (more of B and C borrowers) with those wishing to invest \
             in these types of loans.'
        , style=CARD_TEXT_STYLE2),
        ]),
        html.Br(),
        dbc.Card([dcc.Markdown('''
            __Project Objectives__
            * Tools: Python, R, Dash plotly, pandas, gitHub, OneDrive, Powerpoint, Excel
            * Exploratory Data Analysis (EDA): Profit Implications
            * Build an interactive Dashboard for presenting and showing the data
            * Survival Analysis: Identify timeline of risky events
                 * Competing Risk Analysis
            * Simulator to test findings from model prediction and survival analysis
                 * Monte Carlo Simulator
            * Takeaways 
            '''
        , style=CARD_TEXT_STYLE2),
        ]),
        html.Br(),
        dbc.Card([dcc.Markdown('''
            __About This App__
            * Main Tab -
                * Project Objectives
                * Data Assumptions
                * Dashboard Layout
            * EDA
                * Interactive Dashboard for selecting data to look for opportunities and to understand the data!
                * Some of the plots can be tricky **Write up preferred critera**
                * fix all charts so hover text is better
                * show percent where needed
            * Time Dependent EDA with Survival Analysis
                * Survival Analysis allows a glimpse to time-to-event (prepaid or default in our case)
                * Is widespread in business analytics (customer churn), duration analysis (economics), 
                  reliability analysis (engineering)
                * Allows better customer segmentation to predict default or prepaid behavior
            * Sim
                * The LC Sim, allows the future investor to see the expected range of different investments
                    based on different criteria
                * Demo Mode, will demonstrate some of the features without taking the full time to run the App
            * Conclusion
                * The default/delinquent borrowers are less profitable than the ones that prepay (unsurprisingly)
                * The profitability of 36 month borrowers more predictable and marginally more profitable than 60 month 
                * The time-to-event for prepayment 
            * Hosting
                * this is still a wish list item.
 dataframe options
            * Conclusion 
              * Importance of diversification (risky loans are more profitability and more risky)
              * Investors want choice: analysis and dashboard provide snapshots that provide customization for 
                time, risk, and profit preferences
            * hosting
                * set it up

            '''
        , style=CARD_TEXT_STYLE2),
        ]),
    ]
