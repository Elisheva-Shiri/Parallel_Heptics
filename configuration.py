import csv
import os
import random
from typing import List, Tuple, Union
from consts import PairFinger

def mix_values(comparisons: List[int]) -> List[int]:
    """Randomly shuffle the comparison values."""
    comparisons_mix = comparisons.copy()
    random.shuffle(comparisons_mix)
    return comparisons_mix

def unit(comparisons: List[int], standard: int) -> List[List[Union[int, int, int, int]]]:
    """Create a basic unit with three combinations of object pairs."""
    basic_unit = []
    comparisons_mix = mix_values(comparisons)
    
    # Define the object combinations we want
    combinations = [
        (0, 0),  # First object pair
        (1, 1),  # Second object pair 
        (1, 0)   # Mixed object pair
    ]
    
    for value in comparisons_mix:
        # For each value, create all three combinations
        for obj1, obj2 in combinations:
            if random.randint(0, 1) == 0:
                # [value1, object1, value2, object2]
                basic_unit.append([standard, obj1, value, obj2])
            else:
                basic_unit.append([value, obj1, standard, obj2])
    
    # Shuffle the combinations to mix them up
    random.shuffle(basic_unit)
    return basic_unit

def trail(amount: int, comparisons: List[int], standard: int) -> List[List[int]]:
    """Create a trail consisting of repeated units."""
    trail = []
    for _ in range(amount):
        basic_unit = unit(comparisons, standard)
        trail.extend(basic_unit)
    return trail

def configuration(amount_list: List[int], comparisons: List[int], standard: int) -> List[List[int]]:
    """Create a configuration protocol with trails and zero-relevant values."""
    protocol = []
    for amount in amount_list:
        current_trail = trail(amount, comparisons, standard)
        protocol.extend(current_trail)
        protocol.append([0, 0, 0, 0])  # Zero padding with 4 values
    protocol.append([-1, -1, -1, -1])  # End marker with 4 values
    return protocol

if os.path.exists('configuration.csv'):
    os.remove('configuration.csv')

with open('configuration.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerows(configuration([1,2], [40,130], 85))