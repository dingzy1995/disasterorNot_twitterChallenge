# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:28:32 2020

@author: zding
"""
import argparse
import pandas as pd

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-a','--address', default='train.csv', help='the address of the file to be process')
args = args.parse_args()

def read_data(address):
    with open(address, 'r') as file:
        df=pd.read_csv(file)
        df_without_null=df.dropna()
        df_without_null.to_csv(address+'_without_null.csv')

def main():
    read_data(args.address)

if __name__ == "__main__":
    main()