#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:30:17 2020

@author: cecile.becarie, modified by Alex Roocroft
"""

from ctypes import cdll, byref
import os
import ctypes as ct
import pandas as pd
import xml.etree.ElementTree as ET


def run_symuvia(network_input, network_input_stem, demand, iteration_cycle, iteration_day):
    """
    Runs the SymuVia microscopic traffic simulation for a given iteration cycle and day.

    Parameters:
        network_input (str): Path to the network configuration XML file.
        network_input_stem (str): Base filename (without extension) for saving the updated network configuration.
        demand (pd.DataFrame): DataFrame containing vehicle demand data, including origin, destination, type, and paths.
        iteration_cycle (int): The current iteration cycle number.
        iteration_day (int): The current iteration day number.

    Returns:
        None
    """

    # Load the network configuration XML file
    tree = ET.parse(network_input)
    root = tree.getroot()

    # Find the SCENARIO element within the XML structure
    element_to_update = root.find('SCENARIOS').find('SCENARIO')

    # Modify the output directory attribute to reflect the current iteration and day
    new_value = 'TEMP' + f'_Cycle_{iteration_cycle}_Day_{iteration_day}'
    element_to_update.set('dirout', new_value)

    # Define the filename for the modified network configuration
    network_input_parallel = network_input_stem + f'_Cycle_{iteration_cycle}_Day_{iteration_day}.xml'

    # Save the updated network configuration
    tree.write(network_input_parallel)


    current_dir = os.getcwd()

    lib_path = os.path.abspath(os.path.join(current_dir, os.pardir, 'env_symupy', 'lib')) + os.sep

    # Define the library name and its full path
    # lib_name = 'libSymuFlow.so'
    # full_name = lib_path + lib_name
    #
    # # Change working directory to the library path
    # os.chdir(lib_path)
    # os.environ['PATH'] = lib_path + ';' + os.environ['PATH']
    #
    # # Load the SymuFlow shared library
    # symuflow_lib = cdll.LoadLibrary(full_name)

    lib_path = r'/home/alexroocroft/.conda/envs/env_symupy/lib/' #Need to set to the correct path for user.

    lib_name = 'libSymuFlow.so'
    full_name = lib_path + lib_name
    os.chdir(lib_path)
    os.environ['PATH'] = lib_path + ';' + os.environ['PATH']
    # print(full_name)
    symuflow_lib = cdll.LoadLibrary(full_name)



    # Check if the library was loaded successfully
    if symuflow_lib is None:
        print('Error: SymuVia library not loaded!')

    # Load the SymuVia network configuration
    m = symuflow_lib.SymLoadNetworkEx(network_input_parallel.encode('UTF8'))
    if m != 1:
        print('Error: SymuVia input file not loaded!')
    else:
        print(f'SymuVia input data are loaded for day: {iteration_day}')

    # Initialize simulation variables
    time = 0  # Simulation time step counter
    bEnd = ct.c_int(0)  # Flag to indicate simulation termination
    VNC = 0  # Count of vehicles not created
    VC = 0  # Count of vehicles successfully created

    # -------------------------------------------------
    # Time-step based vehicle flow simulation
    # -------------------------------------------------
    while bEnd.value == 0:

        # Create vehicles at the current time step (vehicles at t=0 are not generated)
        if time > 0:
            # Filter demand data for vehicles scheduled to be created at this timestep
            squery = str(time) + ' < creation_time <= ' + str(time + 1)
            dts = demand.query(squery)

            # Iterate over the filtered demand data
            for index, row in dts.iterrows():
                tc = ct.c_double(row.creation_time - time)  # Adjust creation time relative to the current timestep

                # Ensure the origin and destination are different before creating a vehicle
                if row.origin != row.destination:
                    ok = symuflow_lib.SymCreateVehicleWithRouteEx(
                        row.origin.encode('UTF8'),
                        row.destination.encode('UTF8'),
                        row.type.encode('UTF8'),
                        1,
                        tc,
                        row.path.encode('UTF8')
                    )

                    # Check if vehicle creation was successful
                    if ok < 0:
                        print('Vehicle not created:', ok, row)
                        VNC += 1
                    else:
                        VC += 1
                else:
                    print('Vehicle not created due to identical origin and destination:', row)
                    VNC += 1

        # Advance the simulation to the next timestep
        ok = symuflow_lib.SymRunNextStepLiteEx(1, byref(bEnd))

        # Increment simulation time
        time += 1

        # Print progress every 100 timesteps
        if time % 100 == 0:
            print(f'Cycle {iteration_cycle}, Day {iteration_day}, Timestep: {time}')

        # If simulation has ended, unload the network and print statistics
        if bEnd.value != 0:
            symuflow_lib.SymUnloadCurrentNetworkEx()
            print(f'Microscopic simulation completed for day: {iteration_day}')
            print(f'Vehicles created: {VC}, Vehicles not created: {VNC}')

    # Cleanup by deleting the library reference
    del symuflow_lib
