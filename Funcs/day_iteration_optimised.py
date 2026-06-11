import os
import sys
import pandas as pd

import time


# # Get the absolute path of the parent directory of the current script
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#
# # Add the parent directory to the system path if not already included
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# # Import the SymuFlow simulation function from an external module
# from Funcs import RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v3 as SymuVia_python


# Get the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up to the parent directory
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Add it to the Python path if needed
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the SymuFlow simulation function from an external module

# from Funcs import RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v4_optimised as SymuVia_python

from Funcs import RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v5_optimised_gridlockabort as SymuVia_python

# from Funcs import RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v6_optimised_gridlockDetect as SymuVia_python

# from Funcs import RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v6_2 as SymuVia_python

def run_iteration_day(args):
    """
    Runs a single iteration of the traffic simulation for a given day
    and logs the wall-clock runtime.
    """

    import time
    import pandas as pd

    start = time.time()

    iteration_day = args[0]
    network_input = args[1]
    network_input_stem = args[2]
    demand = args[3]
    iteration_cycle = args[4]
    lib_path = args[5]

    # Load demand for this worker
    df_demand = pd.read_pickle(demand)

    # Run SymuVia
    result = SymuVia_python.run_symuvia(
        lib_path,
        network_input,
        network_input_stem,
        df_demand,
        iteration_cycle,
        iteration_day
    )

    elapsed = time.time() - start

    # Progress output
    # print(
    #     f"[Worker] SymuVia finished → "
    #     f"Cycle {iteration_cycle}, Day {iteration_day} "
    #     f"({elapsed / 60:.1f} min)",
    #     flush=True
    # )

    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"

    print(
    f"[DAY-END] Cycle {iteration_cycle}, Day {iteration_day} "
    f"status={status} "
    f"wall_time={elapsed/60:.1f} min",
    flush=True
    )

    # if not isinstance(result, dict):
    #     result = {}

    # result["wall_time"] = elapsed
    # result["day"] = iteration_day

    # return result

    # if not isinstance(result, dict):
    #     result = {"status": "unknown"}

    # status = result.get("status", "unknown")

    # if status == "completed":
    #     # print(
    #     #     f"[Worker] SymuVia completed → "
    #     #     f"Cycle {iteration_cycle}, Day {iteration_day} "
    #     #     f"({elapsed / 60:.1f} min)",
    #     #     flush=True
    #     # )
    #     pass

    # elif status == "gridlock_abort":
    #     print(
    #         f"[Worker] Day {iteration_day} aborted ({result.get('reason')}) "
    #         f"after {elapsed/60:.1f} min",
    #         flush=True
    #     )
    # else:
    #     print(
    #         f"[Worker] SymuVia finished with status={status} → "
    #         f"Cycle {iteration_cycle}, Day {iteration_day} "
    #         f"({elapsed / 60:.1f} min)",
    #         flush=True
    #     )


    return result


