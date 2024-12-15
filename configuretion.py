import csv
import os
import random
from typing import List, Union

def mix_values(comparisons: List[int]) -> List[int]:
    """Randomly shuffle the comparison values."""
    comparisons_mix = comparisons.copy()
    random.shuffle(comparisons_mix)
    return comparisons_mix

def unit(comparisons_mix: List[int], standard: int) -> List[List[int]]:
    """Create a basic unit of randomized standard-comparison pairs."""
    basic_unit = []
    for value in comparisons_mix:
        if random.randint(0, 1) == 0:
            basic_unit.append([standard, value])
        else:
            basic_unit.append([value, standard])
    return basic_unit

def trail(amount: int, comparisons_mix: List[int], standard: int) -> List[List[int]]:
    """Create a trail consisting of repeated units."""
    trail = []
    for _ in range(amount):
        basic_unit = unit(comparisons_mix, standard)
        trail.extend(basic_unit)
    return trail

def configuration(amount_list: List[int], comparisons_mix: List[int], standard: int) -> List[List[int]]:
    """Create a configuration protocol with trails and zero-relevant values."""
    protocol = []
    for amount in amount_list:
        current_trail = trail(amount, comparisons_mix, standard)
        protocol.extend(current_trail)
        protocol.append([0, 0])
    protocol.append([-1, -1])
    return protocol

if os.path.exists('configuretion.csv'):
    os.remove('configuretion.csv')

with open('configuretion.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(configuration([24,72,72,72], [40,53,66,79,91,104,117,130], 85))