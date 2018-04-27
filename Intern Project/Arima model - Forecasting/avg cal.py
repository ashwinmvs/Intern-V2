# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:24:43 2018

@author: ashwin.monpur
"""

n=int(input("Enter the number of stocks you own: "))

b_price=float(input("Enter the average price at which you bought the stocks: "))
p_price=float(input("Enter the present price of the Stock: "))
print("Note: Average price should be greater than present price")
avg_price=float(input("Enter what average price:  "))

portf=n*b_price
print("Present portifolio value {}".format(portf))

m=0

while round(b_price,2)>=avg_price:
    new_price=(portf+(p_price*m))/(m+n)
    b_price=round(new_price,2)
    #print('123')
    #print(b_price)
    if new_price<=avg_price:
        print('Number of stocks to be bought calculate {}'.format(m))
        print("Amount to be invested: {}".format(p_price*m))
        print("Present Portifolio value : {}".format(portf+(p_price*m)))
        break
    else:
        m=m+1
        continue
        
