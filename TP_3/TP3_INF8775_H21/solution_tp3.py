#!/usr/bin/env python
# coding: utf-8

# # General Initialization

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time

random_seed = 1


# In[2]:


import os

"""
Launch the script to generate the data.
"""
def generate_data(number_blocs_vertical, number_blocs_horizontal):
    number_blocs_vertical = str(number_blocs_vertical)
    number_blocs_horizontal = str(number_blocs_horizontal)
    status = os.system('inst_gen.py -s {} {}'.format(number_blocs_vertical, number_blocs_horizontal))
    if status == 0 : 
        print('SUCCESS : Data generation with ' + number_blocs_vertical + ' by ' + number_blocs_horizontal + ' cities')
        return 'N' + number_blocs_vertical + "_M" + number_blocs_horizontal
    else : 
        print('FAILURE : Data generation with ' + number_blocs_vertical + ' by ' + number_blocs_horizontal + ' cities')
        return ''

"""
Import the data.
"""
def import_data(directory):
    return np.genfromtxt(directory, dtype=np.int64, skip_header=1)

"""
Merge value and costs.
"""
def merge_value_cost(imported_data):
    blocs_data_height = imported_data.shape[0]
    value_data = blocs_data[:int(blocs_data_height/2)]
    cost_data = blocs_data[int(blocs_data_height/2):]
    result_data = value_data - cost_data
    return result_data


# # Data Generation

# In[3]:


generated_file_name = generate_data(number_blocs_vertical=100, number_blocs_horizontal=100)
blocs_data = import_data(generated_file_name)


# In[4]:


result_data = merge_value_cost(blocs_data)
result_data


# # Developped Algorithm (Algorithme Développé)

# ## Definitions

# In[5]:


def display_solution(holes_list):
    for y in range(holes_list.shape[0]):
        for x in range(holes_list.shape[1]):
            if holes_list[y][x]:
                print(y, x)
    print("")

def remove_hole(y_position, x_position, matrix):
    
    # Bottom left
    try:
        if matrix[y_position + 1][x_position - 1] == True:
            remove_hole(y_position + 1, x_position - 1, matrix)
    except:
        pass
    
    # Bottom
    try:
        if matrix[y_position + 1][x_position    ] == True:
            remove_hole(y_position + 1, x_position, matrix)
    except:
        pass
        
    # Bottom right
    try:
        if matrix[y_position + 1][x_position + 1] == True:
            remove_hole(y_position + 1, x_position + 1, matrix)
    except:
        pass
    
    # Remove hole
    matrix[y_position][x_position] = False
        


# In[6]:


def solve(input_data, must_compute_time=True, must_display_path=True, min_row_start_display=200, row_interval_must_display=50):
    
    # Time probe
    if must_compute_time :
        start_time = time.time()
    
    
    # Deep copy the data
    dynamic_data = np.array(input_data)

    # Assignation of the indexes to use
    # 0     X --> X --> X --> X --> LAST  0
    # 0     X --> X --> X --> X --> X     0
    # 0     X --> X --> X --> X --> X     0
    # 0     X --> X --> X --> X --> X     0
    # 0     FIRST X --> X --> X --> X     0
    # 0     0     0     0     0     0     0
    # 0     0     0     0     0     0     0
    data_height = dynamic_data.shape[0]
    data_width = dynamic_data.shape[1]
    
    print(data_height, data_width)
    print(input_data)
    
    lateral_buffer_size = 2
    vertical_buffer_size = 2
    
    hozrizontal_first_index = lateral_buffer_size
    vertical_first_index = dynamic_data.shape[0] - 1

    hozrizontal_last_index = dynamic_data.shape[1] + lateral_buffer_size - 1
    vertical_last_index = 0

    # Add three buffer rows at the bottom
    bottom_buffer = np.zeros((vertical_buffer_size, dynamic_data.shape[1]))
    dynamic_data = np.vstack((dynamic_data, bottom_buffer))
    input_data = np.vstack((input_data, bottom_buffer))

    # Add a buffer column at the left and right side
    latteral_buffer = np.zeros((dynamic_data.shape[0], lateral_buffer_size))
    dynamic_data = np.hstack((latteral_buffer, dynamic_data, latteral_buffer))

    # Compute initial solution
    current_solution = input_data.sum()
    
    # Keep track of hte wholes to make
    holes_list = np.full((data_height, data_width), True, dtype=bool)
    
    # For every row
    has_better_solution_to_display = True
    for index_row in range(vertical_first_index, vertical_last_index - 1, - 1):
        for index_bloc in range(hozrizontal_first_index, hozrizontal_last_index + 1):

            # Hole to inspect
            current_hole = (index_row, index_bloc - lateral_buffer_size)
            
            # Compute the impact of not mining a bloc
            bloc_below = 0
            try:
                bloc_below = input_data[index_row + 1][index_bloc - lateral_buffer_size] * int(holes_list[current_hole[0] + 1][current_hole[1]])
            except:
                pass
            bloc_value = dynamic_data[index_row][index_bloc] + bloc_below + dynamic_data[index_row + 1][index_bloc - 1]  + dynamic_data[index_row + 1][index_bloc + 1] - dynamic_data[index_row + 2][index_bloc]

            # If the bloc has a positive impact on the total value acquired, we assign the value to the dynamix table
            if bloc_value >= 0:
                dynamic_data[index_row][index_bloc] = bloc_value

            # If the bloc has a negative impact on the total value acquired, we do not mine it
            # As a result, we set the value of all the unmined blocs resulting to 0 
            # (only for 2 levels of depth as we will not reuse values that are deeper)
            else:
                
                # Bloc itself
                dynamic_data[index_row][index_bloc] = 0

                # First level of depth
                dynamic_data[index_row + 1][index_bloc] = 0
                dynamic_data[index_row + 1][index_bloc - 1] = 0
                dynamic_data[index_row + 1][index_bloc + 1] = 0

                # Second level of depth
                dynamic_data[index_row + 2][index_bloc - 2] = 0
                dynamic_data[index_row + 2][index_bloc - 1] = 0
                dynamic_data[index_row + 2][index_bloc    ] = 0
                dynamic_data[index_row + 2][index_bloc + 1] = 0
                dynamic_data[index_row + 2][index_bloc + 2] = 0

                # Adjsut the price of the current solution when removing the costly bloc
                current_solution -= bloc_value
                
                # Add the whole to the list of points to remove
                remove_hole(current_hole[0], current_hole[1], holes_list)
                
                # We now have a new better solution
                has_better_solution_to_display = True
                
        if must_display_path and has_better_solution_to_display and (data_height - index_row) >= min_row_start_display and (index_row % row_interval_must_display == 0):
            print("(data_height - index_row)", (data_height - index_row))
            print("min_row_start_display", min_row_start_display)
            print("(data_height - index_row) <= min_row_start_display", (data_height - index_row) <= min_row_start_display)
            print("POSITION ======>", index_row, index_bloc)
            display_solution(holes_list)
            has_better_solution_to_display = False
    
    # Last display
    if must_display_path and has_better_solution_to_display:
        display_solution(holes_list)
        has_better_solution_to_display = False
    
    # Get the best solution
    best_solution = max(current_solution, 0)
    
    # Remove the buffer from the input data
    input_data = input_data[:vertical_buffer_size]

    # RETURN S
    if must_compute_time :
        end_time = time.time()
        computation_time = end_time - start_time
        return (int(best_solution), holes_list, computation_time)
    else :
        return (int(best_solution), holes_list)


# In[7]:


solve(result_data, must_compute_time=True, must_display_path=True)


# # Tests

# ## Definitions

# In[169]:


def perform_power_test(x_exp, y_exp, label):
    
    # solve linear system
    x = np.log(x_exp)
    y = np.log(y_exp)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    y_plot = m * x + c
    
    # Plot data
    plt.figure(figsize=(10,8))
#     plt.xscale("log")
#     plt.yscale("log")
    
    # Experimental data
    plt.plot(np.log(x_exp), np.log(y_exp), color='blue')
    plt.scatter(np.log(x_exp), np.log(y_exp), color='blue', label=label)
    
    # Fit linear
    plt.plot(x, y_plot, color='red', label=str(m) + ' x + ' + str(c))
    

    plt.title("Power Test Representation")
    plt.xlabel("Dataset size (powers of 10)")
    plt.ylabel("Consumption (powers of 10)")
    plt.legend()
    plt.grid()
    plt.show()


# In[68]:


def perform_ratio_test(x_exp, y_exp, label, function, function_title='f(x)'):
    
    # Compute ratios
    x = np.array(x_exp)
    y = np.array(y_exp)
    
    function_evaluation = function(x)
    y_plot = y / function_evaluation
    
    # Plot data
    plt.figure(figsize=(10,8))
#     plt.xscale("log", basex=2)
#     plt.yscale("log", basey=2)
    
    # Plot ratio
    plt.plot(x, y_plot, color='red', label='Ratio : {}'.format(function_title))

    plt.title("Ratio Test Representation (" + label + ")")
    plt.xlabel("Dataset size")
    plt.ylabel("Consummation Ratio")
    plt.legend()
    plt.grid()
    plt.show()
    
    return y/(x**2)


# In[184]:


def perform_constant_test(x_exp, y_exp, label, function, function_title='f(x)'):
    
    vectorized_function = np.vectorize(function) 
    
    # solve linear system
    x = np.array(x_exp, dtype=np.int64)
    y = np.array(y_exp, dtype=np.int64)
    
    
    x_base_function = vectorized_function(x)
    A = np.vstack([x_base_function, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_plot = np.linspace(min(x_exp), max(x_exp), 100)
    y_plot = m * vectorized_function(x_plot) + c
    
    # Plot data
    plt.figure(figsize=(10,8))
    plt.xscale("log")
    plt.yscale("log")
    
    # Experimental data
    plt.plot(x_exp, y_exp, color='blue')
    plt.scatter(x_exp, y_exp, color='blue', label=label)
    
    # Fit linear
    plt.plot(x_plot, y_plot, color='red', label=str(m) + ' f(x) + ' + str(c) + ' (' + function_title+ ')')
    

    plt.title("Constant Test Representation")
    plt.xlabel("Dataset size")
    plt.ylabel("Consummation (power of 10)")
    plt.legend()
    plt.grid()
    plt.show()


# ## Performance calculation for the dynamic programming algorithm

# In[180]:


# Data sizes
SIZES_DYNAMIC = list(range(6,21))
dynamic_cities_data_list = []

# Generate files
cities_data_list = []
for size in SIZES_DYNAMIC:
    dynamic_generated_file_name = generate_data(number_cities=size)
    dynamic_cities_data_list.append(import_data(dynamic_generated_file_name))


# In[181]:


list_city_count_dynamic = []
list_time_dynamic = []

list_city_count_greedy = []
list_time_greedy = []

list_city_count_approximative = []
list_time_approximative = []

print('Dynamic Programing (DP)')
for data in dynamic_cities_data_list:
    
    solution_dynamic, distance_dynamic, time_dynamic = dynamic_programming_solve(data, must_compute_time=True)
    solution_greedy, distance_greedy, time_greedy = greedy_solve(data, must_compute_time=True)
    solution_approximative, distance_approximative, time_approximative = approximative_solve(data, must_compute_time=True)
    
    list_city_count_dynamic.append(len(data))
    list_time_dynamic.append(time_dynamic)
    
    list_city_count_greedy.append(len(data))
    list_time_greedy.append(time_greedy)
    
    list_city_count_approximative.append(len(data))
    list_time_approximative.append(time_approximative)
 
    print(len(data), time_dynamic, distance_dynamic, time_greedy, distance_greedy, time_approximative, distance_approximative)


# In[210]:


# Funtions
def exponential_comparison_function(x):
    return 1/100000 * x * 2.0851 ** x


# In[212]:


# We have reasons to believe that the function is of the order O(n^2)
perform_constant_test(x_exp=list_city_count_dynamic, 
                      y_exp=list_time_dynamic, 
                      label="Dynamic Algorithm", 
                      function=exponential_comparison_function, 
                      function_title='f(x)= 1/100000 * x^2.0851 * 2^x')


# In[213]:


perform_ratio_test(x_exp=list_city_count_dynamic, 
                    y_exp=list_time_dynamic, 
                    label="Dynamic Algorithm",
                    function=exponential_comparison_function,
                    function_title='f(x)= 1/100000 * x^2.0851 * 2^x')


# ## Performance calculation for the approximative and greedy algorithm

# In[214]:


# Data sizes
SIZES_APPROXIMATIVE = list(range(20, 410, 20))
approximative_cities_data_list = []

# Generate files
cities_data_list = []
for size in SIZES_APPROXIMATIVE:
    approximative_generated_file_name = generate_data(number_cities=size)
    approximative_cities_data_list.append(import_data(approximative_generated_file_name))


# In[215]:


list_city_count_approximative = []
list_time_approximative = []

list_city_count_greedy = []
list_time_greedy = []

print('Approximative (Approx)')
for data in approximative_cities_data_list:
    
    solution_greedy, distance_greedy, time_greedy = greedy_solve(data, must_compute_time=True)
    solution_approximative, distance_approximative, time_approximative = approximative_solve(data, must_compute_time=True)
    
    list_city_count_greedy.append(len(data))
    list_time_greedy.append(time_greedy)
    
    list_city_count_approximative.append(len(data))
    list_time_approximative.append(time_approximative)
 
    print(len(data), time_greedy, distance_greedy, time_approximative, distance_approximative)


# ### Greedy algorithm

# In[216]:


perform_power_test(x_exp=list_city_count_greedy, y_exp=list_time_greedy, label='Approximative Algorithm')   


# ### Approximative algorithm

# In[217]:


# Approximative algorithm
perform_power_test(x_exp=list_city_count_approximative, y_exp=list_time_approximative, label='Approximative Algorithm')   


# ## Performance calculation

# In[24]:


list_time_greedy = []
list_time_dynamic = []
list_time_approximative = []

print('Greedy (G), Dynamic Programing (DP), Approximative (A)')
for test_data in cities_data_list:
    solution_greedy, distance_greedy, time_greedy = greedy_solve(cities_data, must_compute_time=True)
    solution_dynamic, distance_dynamic, time_dynamic = dynamic_programming_solve(cities_data, must_compute_time=True)
    solution_approximative, distance_approximative, time_approximative = approximative_solve(cities_data, must_compute_time=True)
    
    
    list_time_greedy.append(time_greedy)
    list_time_dynamic.append(time_dynamic)
    list_time_approximative.append(time_approximative)
 
    print('For ', len(test_data), ' city(ies) --> G:', time_greedy, '; DP:', time_dynamic, '; A: ', time_approximative)


# In[120]:


# plt.figure(figsize=(10,8))

# plt.xscale("log", basex=2)
# plt.yscale("log", basey=2)

# # Brute force Algorithm
# plt.plot(SIZES, list_time_brute_force, color='blue')
# plt.scatter(SIZES, list_time_brute_force, color='blue', label='Brute Force')

# # Divide to reign Algorithm with threshold 1
# plt.plot(SIZES, list_time_divide_to_reign_1, color='turquoise')
# plt.scatter(SIZES, list_time_divide_to_reign_1, color='turquoise', label='Divide to Reign with treshold 1')

# # Divide to reign Algorithm with threshold THRESHOLD
# plt.plot(SIZES, list_time_divide_to_reign_2, color='green')
# plt.scatter(SIZES, list_time_divide_to_reign_2, color='green', label='Divide to Reign with treshold '+ str(THRESHOLD))

# plt.title("Power Test Representation")
# plt.xlabel("Dataset size")
# plt.ylabel("Consummation")
# plt.legend()
# plt.grid()
# plt.show()


# ## Power test

# In[116]:


perform_power_test(x_exp=SIZES, y_exp=list_time_brute_force, label='Brute Force')   


# In[117]:


perform_power_test(x_exp=SIZES, y_exp=list_time_divide_to_reign_1, label='Divide to Reign with treshold 1')


# In[118]:


perform_power_test(x_exp=SIZES, y_exp=list_time_divide_to_reign_2, label='Divide to Reign with treshold '+ str(THRESHOLD))


# ## Ratio Test

# In[156]:


perform_ratio_test(x_exp=SIZES, y_exp=list_time_brute_force, label="Brute Force")   


# In[157]:


perform_ratio_test(x_exp=SIZES, y_exp=list_time_divide_to_reign_1, label='Divide to Reign with treshold 1')


# In[158]:


perform_ratio_test(x_exp=SIZES, y_exp=list_time_divide_to_reign_2, label='Divide to Reign with treshold '+ str(THRESHOLD))


# ## Constant Test

# In[204]:


# Funtions
def power_1(x):
    return x * np.log2(x)

def power_2(x):
    return x**2


# In[205]:


# We have reasons to believe that the function is of the order O(n^2)
perform_constant_test(x_exp=SIZES, y_exp=list_time_brute_force, label="Brute Force", function=power_2, function_title='f(x)=x^2')


# In[206]:


# We have reasons to believe that the function is of the order O(n)
perform_constant_test(x_exp=SIZES, y_exp=list_time_divide_to_reign_1, label='Divide to Reign with treshold 1', function=power_1_log_n, function_title='f(x)=x log(x)')


# In[207]:


# We have reasons to believe that the function is of the order O(n)
perform_constant_test(x_exp=SIZES, y_exp=list_time_divide_to_reign_2, label='Divide to Reign with treshold '+ str(THRESHOLD), function=power_1_log_n, function_title='f(x)=x log(x)')


# # Question 5: Résolution des échantillons difficiles

# In[44]:


hard_file_names = ['hard_N52', 'hard_N91', 'hard_N130', 'hard_N169', 'hard_N199']
for file_name in hard_file_names:
    cities_data = import_data(file_name)
    solution_greedy, distance_greedy, time_greedy = greedy_solve(cities_data, must_compute_time=True)
    solution_approximative, distance_approximative, time_approximative = approximative_solve(cities_data, must_compute_time=True)
 
    print('For ', file_name, ' city(ies) --> Distance Greedy:', distance_greedy,'; Distance Approximative: ', distance_approximative)


# # Script to generate python script

# In[2]:


get_ipython().system('jupyter nbconvert --to script solution_tp2.ipynb')


# In[ ]:




