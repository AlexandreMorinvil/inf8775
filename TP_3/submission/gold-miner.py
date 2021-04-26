#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################
##### Solution developped ######
################################

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
    value_data = imported_data[:int(blocs_data_height/2)]
    cost_data = imported_data[int(blocs_data_height/2):]
    result_data = value_data - cost_data
    return result_data


# # # Data Generation

# # In[3]:


# generated_file_name = generate_data(number_blocs_vertical=100, number_blocs_horizontal=100)
# blocs_data = import_data(generated_file_name)


# # In[4]:


# result_data = merge_value_cost(blocs_data)
# result_data


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


####################################
##### Interface specifications #####
####################################

import sys
import argparse

"""
Parse the parameters sent to execute
"""
def parse_arguments():
    def dir_path(string):
        absolute_path = string
        relative_path = dir_path = os.path.join(os.getcwd(), string)
        if os.path.isfile(absolute_path):
            return absolute_path
        elif os.path.isfile(relative_path):
            return relative_path
        else:
            texte = "'" + string + "' n'est pas un chemin absolu our relatif valide vers un fichier de données."
            raise Exception(texte)

     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exemplaire", \
                        help="Représente le chemin vers les données à traiter (relatif ou absolu)", \
                        action='store', required=True, metavar='EXEMPAIRE', type=dir_path)

    args = parser.parse_args()
    file_name = args.exemplaire

    return file_name

"""
Import the data.
"""
def import_data(directory):
    return np.genfromtxt(directory, dtype=np.int64, skip_header=1)

"""
Computation of the solution according to the algorithm provided
"""
def handle_processing(cities_data):
    return solve(input_data, must_compute_time=True, must_display_path=True, min_row_start_display=300, row_interval_must_display=100)

"""
Entrypoint function
"""
if __name__== "__main__":
    # Initialize the output
    sys.stdout.flush()

    # Argument parsing
    file_name = parse_arguments()
    
    # Solution computation
    input_data = import_data(file_name)
    input_data = merge_value_cost(input_data)

    solution_reached = handle_processing(input_data)

# %%
