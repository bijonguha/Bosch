# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:19:43 2019

@author: BIG1KOR
"""
import cv2

def take_image(img_path):
    
    img = cv2.imread(img_path)
    print('Image loaded successfulyy')
    return img

def inputs(lits):
    print('Inputs Received',lits)

def main(img_path, input_list):
    image = take_image(img_path)
    inputs(input_list)
    return [image, image, image]
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Welcome to Intel OCR GUI, Equation : Ax^2+By')

    parser.add_argument('image', metavar='IMG', help='Image Path', type=str )
    parser.add_argument('list', metavar='VARs' , help='Comma separated list input - A,B,x,y', type=str) 

    args = parser.parse_args()
    
    img_path = args.image
    input_list = [int(item) for item in args.list.split(',')]
    
    main(img_path, input_list)
               