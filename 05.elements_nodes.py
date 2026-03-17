#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:37:43 2025

@author: chingchen
"""






def equ(level,order):
    element = 2**(level)
    node = 2**(level)*(order+1)
    return element, node


for ii in range(2,7):
    for jj in range(1,5):
        print(ii,jj,equ(ii,jj))