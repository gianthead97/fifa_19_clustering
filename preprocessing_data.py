#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def fix_value(x: str) -> str:
    if (x.find('.') == -1):
        return x
    else:
        return x[:len(x)-1].replace('.', '')


def convert_feet_to_centimeters(x : str) -> float:
    if (x == 'nan'):
        return x
    digits = x.split('\'')
    inches = int(digits[1]) + 12 * int(digits[0])
    return (inches * 2.54)



def main():

    df = pd.read_csv('data.csv')
    df = df.drop(df.columns[0], axis=1)
    df.set_index('ID', inplace=True)
    df = df.drop(['Name', 'Photo', 'Nationality', 'Club', 'Club Logo',
        'Flag', 'Special', 'Jersey Number',
        'Contract Valid Until', 'Loaned From',
        'Joined', 'Work Rate', 'Body Type', 'Real Face',
        'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 
        'LAM', 'CAM', 'RAM', 'LM', 'LCM',
        'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 
        'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'], axis=1)


    df.replace({
        'Left': 0,
        'Right': 1
    }, inplace=True)


    df['Wage'] = df[['Wage']].replace(
        {'€':'', 
        'K': '00',
        'M': '00000'
        }, regex=True).convert_objects(convert_numeric=True)

    df['Release Clause'] = df[['Release Clause']].fillna('nan')
    df['Height'] = df[['Height']].fillna('nan')

    df['Release Clause'] = df['Release Clause'].replace({
        '€': '',
        'M': '00000',
        'K': '000',
    }, regex=True).apply(fix_value).convert_objects(convert_numeric=True)

    df['Value'] = df['Value'].replace({
        '€': '',
        'M': '000000',
        'K': '000'
    }, regex=True).map(fix_value).convert_objects(convert_numeric=True)


    df['Position'].replace({
        'GK': 0.0,
        'CB': 1.0,
        'LCB': 1.0,
        'RCB': 1.0,
        'LB': 1.5,
        'RB': 1.5,
        'RWB': 1.9,
        'LWB': 1.9,
        'CM': 2,
        'LCM': 2,
        'RCM': 2,
        'CDM': 1.5,
        'LDM': 1.5,
        'RDM': 1.5,
        'LM': 2.5,
        'RM': 2.5,
        'RAM': 3,
        'CAM': 3,
        'LAM': 3,
        'LW': 3.5,
        'RW': 3.5,
        'CF': 3.8,
        'LF': 3.8,
        'RF': 3.8,
        'LS': 3.9,
        'RS': 3.9,
        'ST': 4.2
    }, inplace=True)

    df['Height'] = df['Height'].map(convert_feet_to_centimeters)

    df['Weight'].replace('lbs', '', regex=True, inplace=True)

    df_num_scaled = pd.DataFrame(MinMaxScaler().fit_transform(df))

    df_num_scaled.columns = df.columns
    print(df_num_scaled.interpolate().head(30))

    df_num_scaled.to_csv('preprocessed_data.csv')




if __name__ == '__main__':
    main()
