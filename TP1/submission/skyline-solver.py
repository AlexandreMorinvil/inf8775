#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import os
import time

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
    parser.add_argument("-a", "--algorithme", \
                        help="Représente l'algorithme à utiliser (brute, recursif, seuil)", \
                        action='store', required=True, metavar='ALGORITHME', type=str)

    parser.add_argument("-e", "--exemplaire", \
                        help="Représente l'algorithme à utiliser (brute, recursif, seuil)", \
                        action='store', required=True, metavar='EXEMPAIRE', type=dir_path)
    
    parser.add_argument("-p", "--present", \
                        help="Affiche, sur chaque ligne, les couples définissant la silhouette de bâtiments", \
                        action='store_true', required=False)

    parser.add_argument("-t", "--temps", \
                        help="Affiche le temps d’exécution en millisecondes", \
                        action='store_true', required=False)

    args = parser.parse_args()

    algorithm = args.algorithme
    file_name = args.exemplaire
    
    if not (algorithm in ['brute', 'recursif', 'seuil']):
        texte = "L'algorithme '" + algorithm + "' n'est pas valide. Les valeurs accesptées sont : brute, recursif ou seuil"
        raise Exception(texte)

    must_present = False
    if args.present:
        must_present = True
    
    must_display_time = False
    if args.temps:
        must_display_time = True

    return (algorithm, file_name, must_present, must_display_time)

"""
Import the data.
"""
def import_data(directory):
    return np.genfromtxt(directory, delimiter=' ', dtype=None, usecols=range(0,3), skip_header=1)

"""
Generate the critical points from the building data.
"""
def find_critical_points(imported_data):
    number_buildings = len(imported_data)

    upper_left_critical_point = np.array([imported_data[:, 0], imported_data[:, 2]]).T
    lower_right_critical_point = np.array([imported_data[:, 1], np.zeros(number_buildings, dtype=np.int8)]).T
    critical_points = np.column_stack([upper_left_critical_point, lower_right_critical_point]).reshape((-1,2))

    return critical_points

"""
Sort the points critical points accorting to their x axis
FOR every critical points
    FOR every building
        IF (the critical point is in a building) AND (the height is below the building height)
            Raise the critical point

    IF the critical point is not redundant
        Add the critical points to the solution
"""
def brute_force_solve(buildings_data, must_compute_time=True):
    
    if must_compute_time :
        start_time = time.time()
    
    critical_points = find_critical_points(buildings_data)
    sorted_critical_points = critical_points[np.argsort(critical_points, axis=0)[:, 0]]
    
    points_to_keep = [[-1,0]]
    for point in sorted_critical_points:
        for building in buildings_data:
            if point[0] > building[0] and point[0] < building[1] and point[1] < building[2]:
                point[1] = building[2]

            if point[0] < building[0]:
                break
        
        previous_point_kept = points_to_keep[-1]
        if previous_point_kept[1] != point[1]:
            if point[0] == previous_point_kept[0]:
                if point[1] > previous_point_kept[1]:
                    points_to_keep.pop()
                    points_to_keep.append(point)
                
            else :
                points_to_keep.append(point)

    points_to_keep = np.array(points_to_keep[1:])
    
    if must_compute_time :
        end_time = time.time()
        computation_time = end_time - start_time
        return (points_to_keep, computation_time)
    else :
        return points_to_keep

"""
Code to recombine two solutions
"""
def recombine_solutions(first_group, second_group): 
    index_first_group = 0
    index_second_group = 0
    
    total_index_first_group = len(first_group)
    total_index_second_group = len(second_group)
    
    is_first_group_covered = False
    is_second_group_covered = False
    
    first_element = first_group[index_first_group]
    second_element = second_group[index_second_group]
    
    height_1 = 0
    height_2 = 0
    current_height = 0
    
    current_element = [0, 0]
    new_solution = [[-1, 0]]
    
    while not(is_first_group_covered) or not(is_second_group_covered) :
        if (not(is_first_group_covered) and first_element[0] <= second_element[0]) or is_second_group_covered:
            current_element = first_element
            height_1 = current_element[1]
            
            if current_element[1] < height_2 :
                current_element[1] = height_2
            
            index_first_group += 1
            if index_first_group < total_index_first_group:
                first_element = first_group[index_first_group]
            else:
                is_first_group_covered = True
            
        elif (not(is_second_group_covered) and second_element[0] <= first_element[0]) or is_first_group_covered:
            current_element = second_element
            height_2 = current_element[1]
            
            if current_element[1] < height_1 :
                current_element[1] = height_1
            
            index_second_group += 1
            if index_second_group < total_index_second_group:
                second_element = second_group[index_second_group]
            else:
                is_second_group_covered = True
                
        else :
            print ('ERROR: IMPOSSIBLE STATE')
            break
        
        # Add the points that are not redundant
        previous_point_kept = new_solution[-1]
        if previous_point_kept[1] != current_element[1]:
            if current_element[0] == previous_point_kept[0]:
                if current_element[1] > previous_point_kept[1]:
                    new_solution.pop()
                    new_solution.append(current_element)
                
            else :
                new_solution.append(current_element)
    
    new_solution = np.array(new_solution[1:])
    return new_solution

"""
Pseudocode :

Function Divide-to-Reign(x {Sample}) : y {solution}
    IF is small return brute-force-algorithm(x)
    Decompose x en 2 sub-samples x_0, x_1
    FOR i=1 to n do :
        yi <-- Divide-to-Reign(x_i)
    Recombine y_n to y

    RETURN y
"""
def divide_to_reign_solve(buildings_data, recursion_threshold=1, must_compute_time=True):
    
    if must_compute_time :
        parent_start_time = time.time()
    
    # IF is small return brute-force-algorithm(x)
    if len(buildings_data) <= max(1, recursion_threshold):
        return brute_force_solve(buildings_data, must_compute_time=must_compute_time)
    
    # Decompose x en 2 sub-samples x_0, x_1
    left_group = None
    right_group = None
    if len(buildings_data) % 2 == 0 :
        splited_buidings_data = np.vsplit(buildings_data[:], 2)
        left_group = splited_buidings_data[0]
        right_group = splited_buidings_data[1]
    else :
        splited_buidings_data = np.vsplit(buildings_data[:-1], 2)
        left_group = splited_buidings_data[0]
        right_group = np.vstack((splited_buidings_data[1], buildings_data[-1]))
        
    # FOR i=1 to n do : yi <-- Divide-to-Reign(x_i)
    left_solution = divide_to_reign_solve(left_group, recursion_threshold=recursion_threshold, must_compute_time=False)
    right_solution = divide_to_reign_solve(right_group, recursion_threshold=recursion_threshold, must_compute_time= False)
    
    # Recombine y_n to y
    solution = recombine_solutions(left_solution, right_solution)
    
    # RETURN y
    if must_compute_time :
        parent_end_time = time.time()
        computation_time = parent_end_time - parent_start_time
        return (solution, computation_time)
    else :
        return solution

"""
Computation of the solution according to the algorithm provided
"""
def handle_processing(buildings_data, algorithm):
    if algorithm == 'brute':
        return brute_force_solve(buildings_data, must_compute_time=True)
    elif algorithm == 'recursif':
        return divide_to_reign_solve(buildings_data, recursion_threshold=1, must_compute_time=True)
    elif algorithm == 'seuil':
        return divide_to_reign_solve(buildings_data, recursion_threshold=25, must_compute_time=True)
    else :
        raise Exception("L'algorithme est invalide.")

"""
Display of the critical points
"""
def handle_solution_presentation(must_present, points_to_keep):
    if not must_present :
        return
    for point in points_to_keep:
        print(point[0], point[1])


"""
Display of the time in miliseconds
"""
def handle_time_display(must_display_time, time_used):
    if not must_display_time :
        return
    print(time_used * 1000)


"""
Entrypoint function
"""
if __name__== "__main__":
    # Argument parsing
    algorithm, file_name, must_present, must_display_time = parse_arguments()
    
    # Solution computation
    buildings_data = import_data(file_name)
    points_to_keep, time_used = handle_processing(buildings_data, algorithm)

    # Presentation
    handle_solution_presentation(must_present, points_to_keep)
    handle_time_display(must_display_time, time_used)
