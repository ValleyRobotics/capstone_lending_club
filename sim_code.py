# regular imports
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

%matplotlib inline

# display related imports
from IPython.display import display, Image, clear_output, HTML, IFrame

# Widgets
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# to save dataframe as an image
import dataframe_image as dfi

# hide warnings
import warnings
warnings.filterwarnings('ignore')

# Forest
from sklearn.tree import DecisionTreeClassifier

# Time Series
from datetime import datetime, timedelta
from dateutil.parser import parse
from pandas.tseries.offsets import DateOffset

df = pd.read_csv('data/df_for_filter.csv')
