# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:19:33 2019

@author: BIG1KOR
"""
#%%
import pandas as pd
import ast
import operator as op
import re
'''
Evaluatore new
'''
# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

def eval_expr(expr):
    """
    >>> eval_expr('2^6')
    4
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    """
    return eval_(ast.parse(expr, mode='eval').body)

def eval_(node):
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)

img_name = input('Please enter name of image : ')
boxes = int(input('Please enter number of workspaces : '))

box_num = []
line_name = []
char = []
exp =[]
line_val = []

for i in range(boxes):
    lines = input('Please enter lines of workspace %d separated by ; (semicolon)\n' %(i+1) )
    lines = lines.split(';')
    for j,line in enumerate(lines):

        flag = 0
        k = 0
        pred = line
        try:
            val = eval_expr(pred)
        except:
            #This except block is fired when brackets are un necessarily used 
            #while writing the answerscripts and in strings
            matches_left = re.findall(r'\d\(\d', pred)
            matches_right = re.findall(r'\d\)\d', pred)
            
            for s in matches_left:
                sn = s.split('(')
                snew = sn[0]+'*('+sn[1]
                pred = pred.replace(s,snew)    
                
            for s in matches_right:
                sn = s.split(')')
                snew = sn[0]+')*'+sn[1]
                pred = pred.replace(s,snew) 
                

            val = eval_expr(pred)
            
        while( k < len(line) ):
            if(line[k] == '*'):
                if(line[k+1] == '*'):
                    flag = 1
                    k = k+2
                    continue
            box_num.append(i)
            line_name.append('%d%d' %(i,j))
            char.append(line[k])
            exp.append(flag)
            line_val.append(val)
            flag = 0
            k = k+1
        
data_dict = {'box_num': box_num, 'line_name': line_name, 'char': char, 'exp':exp, 'line_val':line_val}  
df_chars = pd.DataFrame(data=data_dict, index=None)           

try:
    filename = 'data/kk_images/%s.h5' %img_name
    df_chars.to_hdf(filename, key='df_chars', mode='w')
    print('\nImage annotated file saved as %s' %filename)

except:
    print('\nFile could not be generated')
#%%
    
