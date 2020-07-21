import pandas as pd
import numpy as np



        
# Import csv file with raw data
data=pd.read_csv('cr_loan2.csv')



# Create function constructing a descriptive statstics table
# The table can be used to identify variables with missing values and
# obvious outliers 
def descriptive(df):
    # set screen parameters to faicilitate displaying descriptive statistics
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.options.display.float_format = "{:,.1f}".format
    
    # Construct descriptive statistics table
    des=pd.DataFrame()
    des['Type']=df.dtypes
    des['N']=df.count()
    des['Missing']=df.isnull().sum()
    des['Avg']=df.mean(numeric_only=True)
    des['Med']=df.median(numeric_only=True)
    des['Min']=df.min(numeric_only=True)
    des['Max']=df.max(numeric_only=True)
    
    des = des.replace(np.nan, '', regex=True)
    
    return (des)


