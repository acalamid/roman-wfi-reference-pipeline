from astropy.time import Time, TimeDelta
from pathlib import Path
import subprocess
import numpy as np
import astropy.units as u
import asdf
from wfi_reference_pipeline.constants import WFI_FRAME_TIME, WFI_MODE_WIM
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

"""
After cloning romanisim, the following needs to be done. 

First compiling on central storage or grp/roman/ requires the following command:
pip install -e . --no-build-isolation

I have not yet figured out how to properly access the MA Table reference file from CRDS
within the romanisim command romanisim-make-image --usecrds correctly. 

The work around is to clone romanisim and install it in the current environment 
also available in this directory. Now edit the parameters.py file located in 
/romanisim_clone/romanisim/romanisim/parameters.py line numbers around 150-180
and add a hardcoded diagnostic MA Table that has every read from 1 to 100 as 
each read as its own resultant. 

---> add this below:
18: [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30],
    [31], [32], [33], [34], [35], [36], [37], [38], [39], [40],
    [41], [42], [43], [44], [45], [46], [47], [48], [49], [50],
    [51], [52], [53], [54], [55], [56], [57], [58], [59], [60],
    [61], [62], [63], [64], [65], [66], [67], [68], [69], [70],
    [71], [72], [73], [74], [75], [76], [77], [78], [79], [80],
    [81], [82], [83], [84], [85], [86], [87], [88], [89], [90],
    [91], [92], [93], [94], [95], [96], [97], [98], [99], [100]],

Because of memory constraints for an unknown reason right now,
I am not simulating the realistic short and long darks in the calibration plan
that have 46 and 98 single read resultants, respectively.
For this development we instead set short darks to be 16 single read
resultants while we set the long darks to be 28 single read resultants.
"""

output_dir = Path("/grp/roman/RFP/DEV/sim_inflight_calplan/romanisim_darks")

optical_element = 'F213'  # cant simulate "DARK optical element in romanisim for some reason
ma_table_number = 18
seed = 44
level = 1
cal_level = 'cal' if level == 2 else 'uncal'

# Initialize start time for short darks
program_shortdarks = '00444'
truncate_shortdarks = 16
obs_time_shortdarks = Time('2026-10-01T00:00:00')
num_exp_shortdarks = 26


# Simulate for detectors 1 through 1 (adjust range as needed)
for det in range(1, 2):
    sca = det  # Detector number (1–18)

    # Reset time for this detector
    current_time = obs_time_shortdarks.copy()

    for exp in range(1, 7):  # Exposures increment
    #for exp in range(1, num_exp_shortdarks):  # Exposures increment
        exp_str = f"{exp:04d}"
        sca_str = f"wfi{sca:02d}"

        filename = output_dir / f"r{program_shortdarks}01001001001004_{exp_str}_{sca_str}_{optical_element.lower()}_{cal_level}.asdf"

        command = [
            "romanisim-make-image",
            "--date", current_time.isot,
            "--nobj", "0",
            "--sca", str(sca),
            "--level", str(level),
            "--ma_table_number", str(ma_table_number),
            "--truncate", str(truncate_shortdarks),
            str(filename)
        ]

        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error with exposure {exp}: {result.stderr}")
        else:
            print(f"Success making file: {filename.name}")

        # Simulate dark reads with the RFP utility
        # Note to change the dark rate here is set to be much higher than the default in order to
        # track development - default is 0.01 e-/s
        #
        # We can also modify the number and properties of hot, warm, and dead pixels in the
        # RFP simulated dark cube to add to the romanisim noise model.
        dark_cube, _ = simulate_dark_reads(n_reads=truncate_shortdarks,
                                           dark_rate=1.0)

        with asdf.open(filename, mode='rw') as af:
            # Update some meta values
            af.tree["roman"]["meta"]["instrument"]["optical_element"] = "DARK"
            af.tree["roman"]["meta"]["exposure"]["ma_table_name"] = "DIAGNOSTIC"
            # Add additional dark signal to data from RFP that includes hot, warm, dead pixels to darks
            # Some copies of data arrays to track development.
            #data_cube_orig = af.tree["roman"]["data"].copy()
            af.tree["roman"]["data"] += dark_cube.astype(np.uint16)
            #data_cube_new = af.tree["roman"]["data"].copy()
            af.update()

        # Brad or anyone else look here and try to help me figure out ways to update data from romanisim after file creation
        # When there are at least 36 read resultants none of this works below
        #with asdf.open(filename, mode='rw') as af:
            #meta = af.tree["roman"]["meta"]
            #data_cube = af.tree["roman"]["data"]
            # Update some meta to make consistent with what we expect to see in the dark calibration program
            #meta["instrument"]["optical_element"] = "DARK"
            # need to check that filename for darks is going to be in file string and meta https://stsci-docs.stsci.edu/display/DRAFTSOC/.Data+Levels+and+Products+v2025
            #meta["exposure"]["ma_table_name"] = "DIAGNOSTIC"
            #af.update()

            #TODO work on figuring out why this gives failed tag error
            #start_time = meta["exposure"]["start_time"]
            #new_file_date = start_time + TimeDelta(1, format="jd")  # Add 1 day
            #meta["file_date"] = new_file_date.isot

            #data = data_cube + dark_cube
            #data = data_cube + dark_cube.astype(np.uint16)
            #af.tree["roman"]["meta"] = meta
            #af.tree["roman"]["data"] = data
            #af.write_to(str(filename))

        # Add 10 seconds for the next exposure
        current_time += (truncate_shortdarks * WFI_FRAME_TIME[WFI_MODE_WIM] + 10) * u.s


