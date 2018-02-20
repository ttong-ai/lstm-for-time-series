# This module is for preparing the original data.
# Original data is not clean and has also NaN values.
# We also added hand-crafted features to boost the correlation
# In deep learning, one of the goals is to do feature engineering automatically
# without using any hand-crafted features. We suggest to use auto-encoders for feature engineering.
# We left using auto-encoders as exercise.

import pandas as pd 
import numpy as np

def createNewFeatures(df_expanded):
    # This function is for the feature engineering. Basically, hand-crafted features are being created.
    # Usually, an area expert analyze the data and come up with this kind of feature creation decision.
    # The code below is written based on an expert's analysis.
    
    df_expanded['y_lagged'] = df_expanded[['id', 'y']].groupby('id').shift(periods=1)
    df_expanded['technical_diff'] = df_expanded['technical_20'] - df_expanded['technical_30']
    timediffCols = ['technical_diff', 'technical_20', 'technical_30', 'technical_40']
    # 8 new features will be created
    for thisColumn in timediffCols:
        kernel = thisColumn.replace('technical_','krnl')
        periodicity = thisColumn.replace('technical_','delta5')
        grped = df_expanded[['id', thisColumn]].groupby('id')
        df_expanded[kernel] = 12.5*(df_expanded[thisColumn] - 0.92 * grped[thisColumn].shift(periods=1))
        df_expanded[periodicity] = df_expanded[thisColumn] - grped[thisColumn].shift(periods=5)
    
    for thisColumn in ['fundamental_29']:
        crossSectional = thisColumn.replace('fundamental_', 'fmod').replace('technical_', 'tmod')
        tmp = df_expanded[['timestamp', thisColumn]].groupby('timestamp')[thisColumn].mean()
        df_expanded[crossSectional] = tmp[df_expanded['timestamp']].values
    
    return df_expanded

def fillNaNs(df): 
    # After the extreme data is removed, the data is prepared
    COLs = [c for c in df.columns if c not in ['timestamp', 'y']]     
    # Calculating median without NaN
    COLs_mean = df[COLs].dropna().median() 
    print(COLs_mean.head()) 
    
    # Replaces the NaNs with median.
    df = df.fillna(COLs_mean)
    
    return df 

def removeExtremeValues(df, insampleCutoffTimestamp):
    # Data cleaning part of the code.
    # Truncating (clipping) extreme values
    # utilizing robust statistics (quartiles) because some values
    # are extreme, e.g. derived_1 has a value of 1.06845e+16

    secondQuartile = df[df.timestamp < insampleCutoffTimestamp].quantile(0.25) # Gets the value of the 25% for each column
    fourthQuartile = df[df.timestamp < insampleCutoffTimestamp].quantile(0.75) # Gets the value of the 75% for each column

    # Getting upper and lower level
    twoQuartileRange = fourthQuartile - secondQuartile
    allowedFrom = secondQuartile - 9 * twoQuartileRange 
    allowedTo = fourthQuartile + 9 * twoQuartileRange

    nRows, _ = df.shape

    for thisColumn in df.columns: 
        if thisColumn not in ('id', 'timestamp', 'y', 'CntNs'): 
            # Truncate unusual values
            indexTooLow = df[thisColumn] < allowedFrom[thisColumn]
            indexTooHigh = df[thisColumn] > allowedTo[thisColumn]
            maxValue = df[thisColumn].max()
            minValue = df[thisColumn].min()
            print('Truncating %s: TooLow %s (%.1f%%), TooHigh %s (%.1f%%), range: %s to %s' \
                  % ( thisColumn, sum(indexTooLow), 100*sum(indexTooLow)/nRows, \
                                  sum(indexTooHigh), 100*sum(indexTooHigh)/nRows,
                                  minValue, maxValue))

            # Gets the part of the data where data makes sense
            df[thisColumn] = df[thisColumn].clip(allowedFrom[thisColumn], allowedTo[thisColumn])

            # If the resulted column does not have any variance, it is removed.
            if abs(df[thisColumn].std()) < 0.0000001:
                print('variance = ', df[thisColumn].std(), ' dropping it')
                df.drop([thisColumn], axis = 1, inplace = True)
            #else:
            #    df[thisColumn] = (df[thisColumn] - df[thisColumn].mean()) / df[thisColumn].std()
        else:
            print('skipping ', thisColumn) 
            
    return df