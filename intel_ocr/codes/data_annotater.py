# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:19:33 2019

@author: BIG1KOR
"""
#%%
import pandas as pd

img_name = input('Please enter name of image : ')
boxes = int(input('Please enter number of workspaces : '))

box_num = []
line_name = []
char = []
exp =[]

for i in range(boxes):
    lines = input('Please enter lines of workspace %d separated by ; (semicolon)\n' %(i+1) )
    lines = lines.split(';')
    for j,line in enumerate(lines):
        flag = 0
        for k in range(len(line)):
            if(line[k] == '^'):
                flag = 1
                continue
            box_num.append(i)
            line_name.append('%d%d' %(i,j))
            char.append(line[k])
            exp.append(flag)
            flag = 0

data_dict = {'box_num': box_num, 'line_name': line_name, 'char': char, 'exp':exp}  
df_chars = pd.DataFrame(data=data_dict, index=None)           

try:
    filename = 'data/%s.h5' %img_name
    df_chars.to_hdf(filename, key='df_chars', mode='w')
    print('\nImage annotated file saved as %s' %filename)

except:
    print('\nFile could not be generated')
#%%
    
