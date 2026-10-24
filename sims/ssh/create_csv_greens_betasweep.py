import os
import numpy as np

# Special ID for input and output files
sID = 1

# Type of correlation function
# TYPE = 'momentum'
TYPE = 'position'

# Phonon frequency, chemical potential, lattice size
w = 0.5;    m = 0.0;    L = 12
# alphas = [0.0460, 0.0470, 0.0480, 0.0490, 0.0500]
alphas = [0.240, 0.245, 0.250, 0.255]
betas = np.linspace(1, 23, 45)  

# Snapshot size and sample size (number of bins in the QMC simlation)
snapshot_size = L * L;      sample_size = 100 # 100 bins set in the QMC simlation

# Target orbits considered and their total number (S33, S44, S55, S43, S54, S53)
all_orbits = ['S11', 'S22', 'S12'];  total_orbits = len(all_orbits)

# Orbits to be selected 
TARGET_ORBITS = ['S22']
# TARGET_ORBITS = ['S33', 'S44', 'S55']
# TARGET_ORBITS = ['S33', 'S44', 'S55', 'S43', 'S54', 'S53']

# Convert target orbits to indices and create a string for the folder name
orbit_idces = [all_orbits.index(orbit) for orbit in TARGET_ORBITS]
s = ''.join(str(i) for i in orbit_idces)

# Change the main path to the path leading to the folder w0.1_10perBin containing input data
INPUT_PATH = '/nfs/home/gissa/SSH/Data'

# Create a folder to store the data
OUTPUT_PATH = '/nfs/home/gissa/SSH/LBC'

# beta sweep
for a in alphas:
    
    print(f'\n' + '#' * 80)
    print(f'alpha = {a:.4f}')
    print(f'#' * 80 + '\n')

    print(f'\n Array of b: \n {betas} \n'); print(f'Array size: \n {np.size(betas)}\n')

    # Create output folder
    folder_name = f'greens{s}_{TYPE}_L{L}_w{w:.1f}_m{m:.1f}_a{a:.4f}_csv-{sID}'
    os.makedirs(f'{OUTPUT_PATH}/{folder_name}', exist_ok=True)
    
    for b in betas:
        
        print('-' * 20 + f'b = {b:.2f}' + '-' * 20)

        # Locate each summary file for every value of a
        summary_file = f'{INPUT_PATH}/ossh_honeycomb_w{w:.1f}_a{a:.4f}_m{m:.1f}_L{L}_b{b:.2f}-1/equal-time/greens/greens_{TYPE}_equal-time_bins.csv'

        if os.path.exists(summary_file):
            print(f'Now accessing folder {os.path.basename(summary_file)} for a = {a:.4f}\n')

        # Load all the summary file of shape (sample_size * total_orbits * snapshot_size, n_cols)
        data = np.loadtxt(summary_file, skiprows=1)

        # Compute magnitudes for greens functions and reshape
        magnitudes = np.hypot(data[:, 3], data[:, 4])
        real = data[:, 3]
        imag = data[:, 4]
        real = real.reshape(sample_size, total_orbits, snapshot_size)
        imag = imag.reshape(sample_size, total_orbits, snapshot_size)
        magnitudes = magnitudes.reshape(sample_size, total_orbits, snapshot_size)

        # Pick out the target orbits and flatten
        # selected = real[:, orbit_idces, :]
        # selected = imag[:, orbit_idces, :]
        selected = magnitudes[:, orbit_idces, :]
        selected = selected.reshape(sample_size * len(orbit_idces), snapshot_size)

        # Build final array with last column = a
        Sbb = np.zeros((selected.shape[0], selected.shape[1] + 1))
        Sbb[:, :-1] = selected
        Sbb[:, -1]  = b
        
        # Sum over the target orbits if more than one
        if len(orbit_idces) > 1:
            
            Sbb_sum = np.zeros((sample_size, snapshot_size + 1))
            Sbb_sum[:, :-1] = np.sum(selected.reshape(sample_size, len(orbit_idces), snapshot_size), axis=1)
            Sbb_sum[:, -1]  = b
            Sbb = Sbb_sum

        # Save the data
        file_name = f'greens{s}_{TYPE}_L{L}_w{w:.1f}_m{m:.1f}_b{b:.2f}_a{a:.4f}-{sID}.csv'
        np.savetxt(f'{OUTPUT_PATH}/{folder_name}/{file_name}', Sbb, delimiter=',')
        print(f'Saved greens data size: {np.shape(Sbb)}\n')

    print('#' * 80)
    print(f'All data saved in folder {os.path.basename(folder_name)}!')
    print('#' * 80 + '\n')

    Sbb_giant = []

    for i, b in enumerate(betas):
        file_name = f'greens{s}_{TYPE}_L{L}_w{w:.1f}_m{m:.1f}_b{b:.2f}_a{a:.4f}-{sID}.csv'
        Sbb_giant.append(np.loadtxt(f'{OUTPUT_PATH}/{folder_name}/{file_name}', delimiter=','))

    Sbb_giant = np.vstack(Sbb_giant)

    # Save the giant matrix
    giant_file_name = f'greens{s}_{TYPE}_L{L}_w{w:.1f}_m{m:.1f}_a{a:.4f}-{sID}.csv'
    np.savetxt(f'{OUTPUT_PATH}/{folder_name}/{giant_file_name}', Sbb_giant, delimiter=',')
    print(f'Giant d saved! | Data size is {np.shape(Sbb_giant)}\n')