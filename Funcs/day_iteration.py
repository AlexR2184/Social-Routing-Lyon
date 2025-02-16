import os
import sys

# Get the absolute path of the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to the system path if not already included
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the SymuFlow simulation function from an external module
from Funcs import RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v3 as SymuVia_python


def run_iteration_day(args):
    """
    Runs a single iteration of the traffic simulation for a given day.

    Parameters:
        args (list): A list containing the following parameters:
            - iteration_day (int): The current iteration day number.
            - network_input (str): Path to the network input file.
            - network_input_stem (str): Base name of the network input file.
            - demand (str): Path to the demand input file.
            - iteration_cycle (int): The current iteration cycle number.
            - out_folder (str): Directory for output storage.
            - symuvia_output (str): Name of the SymuVia output file.
            - sim_hours (int): Number of simulation hours.

    Returns:
        None
    """

    iteration_day=args[0]
    network_input = args[1]
    network_input_stem = args[2]
    demand = args[3]
    iteration_cycle = args[4]
    lib_path = args[5]



    SymuVia_python.run_symuvia(lib_path,network_input,network_input_stem,demand,iteration_cycle,iteration_day)

