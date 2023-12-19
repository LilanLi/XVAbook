import numpy as np
from scipy.stats import f
import pandas as pd
import time as time
#################################
from numba import jit
start = time.time()

df = pd.read_csv(r"D:\Users\data1.csv")
@jit(nopython=True)
def f_test(source_table,value_to_check):
    size_1 = 0 
    size_2 = 0 
    m1_1 = 0.0 
    xi2_1 = 0.0 
    m1_2 = 0.0 
    xi2_2 = 0.0 
    tol = 1.0e-9      
    #for i in range(n_row): 
    for field in source_table:
        if not np.isnan(field): 
            vij = float(field) 
            m1_1 += vij 
            xi2_1 += vij * vij 
            if np.abs(vij - value_to_check) > tol: 
                m1_2 += vij 
                xi2_2 += vij * vij 
                size_2 += 1 
            size_1 += 1 
    if size_2 == size_1: 
        raise ValueError("Value being checked is not in the input range") 
    if size_2 < 2: 
        raise ValueError("Insufficient range size") 
    m1_1 /= size_1 
    m1_2 /= size_2 
    xi2_1 = xi2_1 / size_1 - m1_1 * m1_1 
    xi2_2 = xi2_2 / size_2 - m1_2 * m1_2 
        # Calculate F-statistic 
    df1 = size_1 - 1 
    df2 = size_2 - 1
    f_stat = (xi2_2 / df2) / (xi2_1 / df1)
    return f_stat, df1, df2


#main
results = {'Tile_Num': [], 'Tile_SequenceNum':[],'Submission': [], 'F_Stat': [], 'P_Value': []}

for category in df['Tile_Num'].unique().tolist(): # Extract the subset of data for the category 
    category_data = df[df['Tile_Num'] == category] 
    category_values = category_data['Submission'].tolist()
    category_sequence_num = category_data['Tile_SequenceNum'].tolist()
    # Loop through the values in the subset 
    for value, sequence_num in zip(category_values, category_sequence_num): 
 # Calculate the f-test and p-test for the samples 
        #result = xlMathStat_FstatIncremental(category_values, value) #append the results to the dictionary  
        try:
            if np.isnan(value): 
               raise ValueError("Error: empty field to check") 
            f_test_result = f_test(category_values, value)
            f_stat= f_test_result[0]    
            p_value = f.cdf(f_stat, f_test_result[1], f_test_result[2])  
        except Exception as e:
            print (str(e))
        results['Tile_Num'].append(category) 
        results['Tile_SequenceNum'].append(sequence_num) # Add Tile_SequenceNum 
        results['Submission'].append(value) 
        results['F_Stat'].append(f_stat) 
        results['P_Value'].append(p_value) 

results_df = pd.DataFrame(results)
print(results_df)

end = time.time()
print("Elapsed (with compilation) = %s" % (end-start))
