#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random


# # Simulation Project - Newsdealer's Problem
# 

# ## Define function to run simulation

# In[2]:


# newsdealer simulation function
def simulate_newspaper(cost_price, selling_price, scrap_price, order_quantity, ndays, newstype_dist, demand_dist, random_numbers):
        
    # initialize empty dataframe
    df = pd.DataFrame(columns=['Day','RN Newsday Type','Newsday Type','RN Demand','Demand','Revenue','Lost Profit','Scrap Sale','Daily Profit'])
    
    # index to interate over random numbers
    idx = 0
    
    # calculate cost of newspapers
    cost_of_newspapers = order_quantity * cost_price
    
    # repeat the following from day 1 to n number of days
    for i in range(1,ndays+1):
        
        # get newstype rn then newstype
        rn_newstype = random_numbers[idx]
        idx = idx + 1
        newstype = newstype_dist["newsday type"][newstype_dist["cprob"] >= rn_newstype].tolist()[0]
        
        # get demand rn then demand
        rn_demand = random_numbers[idx]
        idx = idx + 1
        demand = demand_dist["demand"][demand_dist[f"{newstype} cprob"] >= rn_demand].tolist()[0]
        
        # calculate revenue, lost profit and scrap
        if (demand <= order_quantity):
            revenue = selling_price*demand
            lost_profit = 0
            scrap = (order_quantity - demand) * scrap_price
        else:
            revenue = selling_price*order_quantity
            lost_profit = (demand - order_quantity) * (selling_price - cost_price)
            scrap = 0
        
        # calculate day profit
        profit = revenue - cost_of_newspapers - lost_profit + scrap
        
        # create new row for this day and add it to dataframe
        new_row = {
            "Day": i,
            "RN Newsday Type": rn_newstype,
            "Newsday Type": newstype,
            "RN Demand": rn_demand,
            "Demand": demand,
            "Revenue": round(revenue * 0.01, 2),
            "Lost Profit": round(lost_profit * 0.01, 2),
            "Scrap Sale": round(scrap * 0.01, 2),
            "Daily Profit": round(profit * 0.01, 2)
        }
        df.loc[i] = new_row
    
    # return the dataframe with all the calculations and the total profit variable
    total_profit = sum(df["Daily Profit"])
    return df, total_profit


# ## Take Parameters as user input

# In[3]:


cost_price = int(input('The price at which the dealer buys the newspapers (in cent): '))


# In[4]:


selling_price = int(input('The price at which the dealer sells the newspapers (in cent): '))


# In[5]:


scrap_price = int(input('Scrap price (in cent): '))


# In[6]:


order_quantity = int(input('Order quantity (in bundles of 10): '))


# In[7]:


ndays = int(input('Number of days to run the simulation: '))


# In[8]:


newsday_type = ['good','fair','poor']
newsday_prob = []

print('Enter probabilities for Newsday Types good, fair, poor respectively:')
print()
for i in range(0, 3):
        newsday_prob.append(float(input()))

print()

# add probabilities to dataframe and create cumulative probability column
newstype_dist = pd.DataFrame({
    "newsday type": newsday_type,
    "prob": newsday_prob
})

newstype_dist["cprob"] = newstype_dist["prob"].cumsum()

print(newstype_dist)


# In[9]:


demand_values = [40, 50, 60, 70, 80, 90, 100]
demand_dist = pd.DataFrame({
    "demand": demand_values
})

print('Enter probabilities for demand values from 40 t 100 (7 numbers for each newsday type):')
print()
for type in newsday_type:
    demand_prob = []
    print(f'Demand probailities for {type} newsdays:')
    print()
    for i in range(0, 7):
        demand_prob.append(float(input()))
    print()
    demand_dist[f'{type} prob'] = demand_prob

# add probabilities to dataframe and create cumulative probability columns
demand_dist["good cprob"] = demand_dist["good prob"].cumsum()
demand_dist["fair cprob"] = demand_dist["fair prob"].cumsum()
demand_dist["poor cprob"] = demand_dist["poor prob"].cumsum()

print(demand_dist)


# In[10]:


random_numbers = []
if(input('Would you like to enter custom Random Numbers? (T/F) ') == 'T'):
    
    print(f'Enter {ndays*2} numbers between 1 and 100:\n')
    for i in range(0, ndays*2):
        n = input()
        random_numbers.append(int(n))
else:
    random_numbers = random.sample(range(1,100), ndays * 2)
    print(f'Generated numbers: {random_numbers}')

# scale numbers [1,100] ~ [0,1] to match cprob column
random_numbers = [round(n * 0.01, 2) for n in random_numbers]
print(f'\nScaled numbers: {random_numbers}')


# ## Call function to run simulation

# In[11]:


simulation, profit = simulate_newspaper(cost_price, selling_price, scrap_price, order_quantity, ndays, newstype_dist, demand_dist, random_numbers)

display(simulation)
print(f'\nTotal Profit: {profit}')

