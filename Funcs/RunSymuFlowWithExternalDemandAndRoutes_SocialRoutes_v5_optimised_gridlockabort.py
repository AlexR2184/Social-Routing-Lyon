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



# hard cutoff per day (seconds)
MAX_DAY_SECONDS = 30 * 60  # tune this (e.g. 1700)


# =========================================================
# Load SymuFlow library ONCE per worker (major speedup)
# =========================================================
_symuflow_lib = None

def _load_symuflow(lib_path):
    global _symuflow_lib
    if _symuflow_lib is None:
        full_name = os.path.join(lib_path, 'libSymuFlow.so')
        _symuflow_lib = cdll.LoadLibrary(full_name)
    return _symuflow_lib

def run_symuvia(lib_path, network_input, network_input_stem, demand, iteration_cycle, iteration_day):
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
    import os
    import time

    # SMOKE_TEST = bool(int(os.environ.get("SMOKE_TEST", "0")))

    start_wall_time = time.time()


    # print(
    #     f"[run_symuvia] Running in PID {os.getpid()} — Called in iteration_day {iteration_day}, cycle {iteration_cycle}")
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

    # lib_path = os.path.abspath(os.path.join(current_dir, os.pardir, 'env_symupy', 'lib')) + os.sep

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

    # lib_path = r'/home/alexroocroft/.conda/envs/env_symupy/lib/' #Need to set to the correct path for user.

    lib_name = 'libSymuFlow.so'
    full_name = lib_path + lib_name
    os.chdir(lib_path)
    os.environ['PATH'] = lib_path + ';' + os.environ['PATH']
    # print(full_name)
    # symuflow_lib = cdll.LoadLibrary(full_name)
    symuflow_lib = _load_symuflow(lib_path)

    # Check if the library was loaded successfully
    if symuflow_lib is None:
        print('Error: SymuVia library not loaded!')

    # print(f"[{os.getpid()}] Loading SymuVia network: {network_input_parallel}")

    # Load the SymuVia network configuration
    m = symuflow_lib.SymLoadNetworkEx(network_input_parallel.encode('UTF8'))
    if m != 1:
        print(f'Error: SymuVia input file not loaded! Day: {iteration_day}')
    # else:
    #     print(f'SymuVia input data are loaded for day: {iteration_day}')

    # Initialize simulation variables
    sim_time = 0  # Simulation time step counter
    bEnd = ct.c_int(0)  # Flag to indicate simulation termination
    VNC = 0  # Count of vehicles not created
    VC = 0  # Count of vehicles successfully created

    # -------------------------------------------------
    # Time-step based vehicle flow simulation
    # -------------------------------------------------
    # print(f"[{os.getpid()}] Starting timestep loop")

    while bEnd.value == 0:

        elapsed = time.time() - start_wall_time

        # ---- [START] EARLY ABORT IF GRIDLOCKED (disabled during smoke test) ----
        if elapsed > MAX_DAY_SECONDS:
            print(
                f"[GRIDLOCK-ABORT] "
                f"Cycle {iteration_cycle}, Day {iteration_day}: "
                f"aborted after {elapsed/60:.1f} min "
                f"(limit = {MAX_DAY_SECONDS/60:.1f} min)",
                flush=True
            )
            # symuflow_lib.SymUnloadCurrentNetworkEx()
            return {
                "status": "gridlock_abort",
                "cycle": iteration_cycle,
                "day": iteration_day,
                "reason": "time"  # or "sps"
            }
        # ---- [END] EARLY ABORT IF GRIDLOCKED ----


        # Create vehicles at the current time step (vehicles at t=0 are not generated)
        if sim_time > 0:
            # Filter demand data for vehicles scheduled to be created at this timestep
            squery = str(sim_time) + ' < creation_time <= ' + str(sim_time + 1)
            dts = demand.query(squery)

            # Iterate over the filtered demand data
            for index, row in dts.iterrows():
                tc = ct.c_double(row.creation_time - sim_time)  # Adjust creation time relative to the current timestep

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
        # print(f"[{os.getpid()}] Unloading network after {sim_time} steps")

        ok = symuflow_lib.SymRunNextStepLiteEx(1, byref(bEnd))

        # Increment simulation time
        sim_time += 1

        # Print progress every 100 timesteps - useful for debugging to check simulations are running.
        # if sim_time % 1000 == 0:
        #     print(f'Cycle {iteration_cycle}, Day {iteration_day}, Timestep: {sim_time}')

        # If simulation has ended, unload the network and print statistics
        if bEnd.value != 0:
            elapsed_complete = time.time() - start_wall_time
            print(
                        f"[COMPLETED] "
                        f"Cycle {iteration_cycle}, Day {iteration_day}: "
                        f"completed after {elapsed_complete/60:.1f} min "
                        f"(limit = {MAX_DAY_SECONDS/60:.1f} min)",
                        flush=True
                    )

            symuflow_lib.SymUnloadCurrentNetworkEx()
            # print(f'Microscopic simulation completed for day: {iteration_day}')
            # print(f'Vehicles created: {VC}, Vehicles not created: {VNC}')

    # Cleanup by deleting the library reference
    del symuflow_lib

    return {
    "status": "completed",
    "cycle": iteration_cycle,
    "day": iteration_day
    }   

