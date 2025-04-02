# Importing libraries
import eec
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from numpy.lib import real

# Dictionary to store dataset paths
datasets = {
    'light_quark_real': {'path': '../datasets/', 'file': 'q.hdf5'},
    'light_quark_gen': {'path': '../trained_models/mp_q/', 'file': 'gen_jets.npy'},
    'gluon_real': {'path': '../datasets/', 'file': 'g.hdf5'},
    'gluon_gen': {'path': '../trained_models/mp_g/', 'file': 'gen_jets.npy'},
    'top_quark_real': {'path': '../datasets/', 'file': 't.hdf5'},
    'top_quark_gen': {'path': '../trained_models/mp_t/', 'file': 'gen_jets.npy'}
}

# EEC parameters
bins = 100
axis_range = (1e-3, 1)

# Function to load real datasets (HDF5 format)
def load_real_dataset(data_path, filename):
    full_path = os.path.join(data_path, filename)
    print(f"Loading real dataset from: {full_path}")
    
    try:
        # Check if the file exists
        if not os.path.exists(full_path):
            print(f"Error: File {full_path} not found!")
            return None
        
        # Open the HDF5 file
        with h5py.File(full_path, 'r') as f:
            # Print keys to understand structure
            print(f"HDF5 file keys: {list(f.keys())}")
            
           # Extract the particle_features dataset
            if 'particle_features' in f:
                particle_data = f['particle_features'][:]
                print(f"Loaded particle_features with shape: {particle_data.shape}")
                
                # Process each jet's particles
                jets = []
                for jet in particle_data:
                    # Extract the features: original order is [prel, yrel, PTrel, mask]
                    # Rearrange to [PTrel, yrel, prel] for [pt, y, phi]-like structure
                    features = jet[:, :3]  # Get first 3 features
                    rearranged = np.zeros_like(features)
                    
                    # Rearrange as [pt, y, phi]
                    rearranged[:, 0] = features[:, 2]  # PTrel → pt
                    rearranged[:, 1] = features[:, 0]  # yrel → y
                    rearranged[:, 2] = features[:, 1]  # prel → third component
                    
                    # # Use mask to filter valid particles
                    # mask = jet[:, 3] > 0
                    # particles = rearranged[mask]
                    particles = rearranged

                    if particles.size > 0:
                        jets.append(particles)
                
                print(f"Processed {len(jets)} jets")
                return np.array(jets, dtype=np.float64)
            else:
                print("Error: 'particle_features' not found in the HDF5 file")
                return None
    except Exception as e:
        print(f"Error loading real dataset: {str(e)}")
        return None


# Function to load generated datasets (NPY format)
def load_gen_dataset(data_path, filename):
    full_path = os.path.join(data_path, filename)
    print(f"Loading generated dataset from: {full_path}")
    
    try:
        # Check if the file exists
        if not os.path.exists(full_path):
            print(f"Error: File {full_path} not found!")
            return None
        
        # Load the NumPy file
        data = np.load(full_path, allow_pickle=True)
        print(f"Successfully loaded NumPy file. Shape: {data.shape}")
        
        # Print sample data for debugging
        if len(data) > 0:
            print(f"First jet shape: {data[0].shape}")
            print(f"First few entries: {data[0][:3]}")
        
        # Process the data to have the correct format [pt, y, phi]
        jets = []
        for jet in data:
            # Ensure valid shape and rearrange features
            if jet.shape[1] == 3:
                # Rearrange columns explicitly based on expected order
                # Assuming current order in generated data is [phi_rel, y_rel, pt_rel]
                rearranged = np.zeros_like(jet)
                rearranged[:, 0] = jet[:, 2]  # pt_rel → pt (index 2)
                rearranged[:, 1] = jet[:, 1]  # y_rel → y (index 1)
                rearranged[:, 2] = jet[:, 0]  # phi_rel → phi (index 0)
                
                # Filter out zero-padded particles
                mask = ~np.all(rearranged == 0, axis=1)
                particles = rearranged[mask]
                
                if particles.size > 0:
                    jets.append(particles)
        
        print(f"Processed {len(jets)} jets from generated data")
        return np.array(jets, dtype=np.float64)
        
    except Exception as e:
        print(f"Error loading generated dataset: {str(e)}")
        return None


# Function to compute EEC
def compute_eec(particles, n=2):
    if particles is None or len(particles) == 0:
        print(f"Cannot compute EEC for n={n}: Empty or None particles data")
        return None
    
    print(f"Computing E{n}C for {len(particles)} jets")
    eec_calculator = eec.EECLongestSide(n, bins, axis_range=axis_range)
    eec_calculator(particles)
    
    # Check if calculation succeeded
    if eec_calculator.sum() > 0:
        eec_calculator.scale(1/eec_calculator.sum())
        return eec_calculator
    else:
        print(f"Warning: EEC calculation resulted in zero sum for n={n}")
        return None

# Function to create EEC plots (3 subplots for different jet types)
def create_eec_plots():
    print("Creating EEC plots...")
    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Jet types and corresponding dataset keys
    jet_types = ['Light Quark', 'Gluon', 'Top Quark']
    keys = [
        ('light_quark_real', 'light_quark_gen'),
        ('gluon_real', 'gluon_gen'),
        ('top_quark_real', 'top_quark_gen')
    ]
    
    # With these two dictionaries:
    errorbar_opts_real = {
        'fmt': 'o',
        'markeredgecolor': 'blue', 
        'markerfacecolor': 'none',
        'markersize': 6,
        'linestyle': '',               # No connecting line
        'capsize': 1.5,
        'capthick': 1,

        # 'fmt': 'o',  # Circle for real
        # 'lw': 1.5,
        # 'capsize': 1.5,
        # 'capthick': 1,
        # 'markersize': 6,
        # 'markerfacecolor': 'none',  # Hollow marker
        # 'markeredgecolor': 'blue',
        # 'markeredgewidth': 1.2
    }

    errorbar_opts_gen = {
        'fmt': 's',
        'markeredgecolor': 'red',
        'markerfacecolor': 'none',
        'markersize': 6,
        'linestyle': '',               # No connecting line
        'capsize': 1.5,
        'capthick': 1,

        # 'fmt': 's',  # Square for generated
        # 'lw': 1.5,
        # 'capsize': 1.5,
        # 'capthick': 1,
        # 'markersize': 6,
        # 'markerfacecolor': 'none',  # Hollow marker
        # 'markeredgecolor': 'red',
        # 'markeredgewidth': 1.2
    }   

    # Process each jet type
    for i, (jet_type, (real_key, gen_key)) in enumerate(zip(jet_types, keys)):
        print(f"\nCreating plot for {jet_type} jets...")
        
        # Load real dataset
        real_particles = load_real_dataset(datasets[real_key]['path'], datasets[real_key]['file'])
        
        # Load generated dataset
        gen_particles = load_gen_dataset(datasets[gen_key]['path'], datasets[gen_key]['file'])
        
        # Compute EECs if data is available
        if real_particles is not None and len(real_particles) > 0:

            # Reshaping and filtering out zero-padded rows
            result = []
            
            for j in range(real_particles.shape[0]):
                # Filter out zero-padded rows
                non_zero_particles = real_particles[j][~np.all(real_particles[j] == 0, axis=1)]
                result.append(non_zero_particles)

            real_particles = result

            real_eec = compute_eec(real_particles)
            
            # Get bin information and histogram for real data
            if real_eec is not None:
                real_midbins, real_bins = real_eec.bin_centers(), real_eec.bin_edges()
                real_hist, real_errs = real_eec.get_hist_errs(0, False)
                
                # Plot real data
                axs[i].errorbar(
                real_midbins, real_hist,
                xerr=(real_midbins - real_bins[:-1], real_bins[1:] - real_midbins),
                yerr=real_errs,
                label="Real Jets",
                **errorbar_opts_real
                )
        else:
            print(f"Warning: No valid real jet data for {jet_type}")
        
        # Plot generated data if available
        if gen_particles is not None and len(gen_particles) > 0:
            gen_eec = compute_eec(gen_particles)
            
            # Get bin information and histogram for generated data
            if gen_eec is not None:
                gen_midbins, gen_bins = gen_eec.bin_centers(), gen_eec.bin_edges()
                gen_hist, gen_errs = gen_eec.get_hist_errs(0, False)
                
                # Plot generated data
                axs[i].errorbar(
                gen_midbins, gen_hist,
                xerr=(gen_midbins - gen_bins[:-1], gen_bins[1:] - gen_midbins),
                yerr=gen_errs,
                label="Generated Jets",
                **errorbar_opts_gen
                )
        else:
            print(f"Warning: No valid generated jet data for {jet_type}")
        
        # Set scales and limits
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_xlim(1e-3, 1)
        axs[i].set_ylim(1e-4, 1)
        
        # Set labels and title
        axs[i].set_xlabel('R$_{L}$')
        axs[i].set_ylabel('Normalized EEC')
        axs[i].set_title(f'{jet_type} Jets')
        axs[i].legend(loc='lower center', frameon=False)
    
    # Adjust layout and save
    fig.tight_layout()
    os.makedirs('./eec_plots/plots', exist_ok=True)
    plt.savefig('./eec_plots/plots/eec_comparison.png')
    plt.close(fig)
    print("EEC comparison plot saved")

# Function to create EEC ratio plots
def create_eec_ratio_plots():
    print("Creating EEC ratio plots...")
    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Jet types and corresponding dataset keys
    jet_types = ['Light Quark', 'Gluon', 'Top Quark']
    keys = [
        ('light_quark_real', 'light_quark_gen'),
        ('gluon_real', 'gluon_gen'),
        ('top_quark_real', 'top_quark_gen')
    ]
    
    # Colors for different ratios
    colors = {3: 'tab:blue', 4: 'tab:green', 5: 'tab:red', 6: 'tab:orange'}

    # Error bar options
    errorbar_opts_real = {
        'fmt': 'o',  # Circle for real
        'lw': 1.5,
        'linestyle': '',
        'capsize': 1.5,
        'capthick': 1,
        'markersize': 6,
        'markerfacecolor': 'none',  # Hollow marker
        'markeredgewidth': 1.2
    }

        # 'fmt': 's',
        # 'markeredgecolor': 'red',
        # 'markerfacecolor': 'none',
        # 'markersize': 6,
        # 'linestyle': '',               # No connecting line
        # 'capsize': 1.5,
        # 'capthick': 1,

    errorbar_opts_gen = {
        'fmt': 's',  # Square for generated
        'lw': 1.5,
        'linestyle': '',
        'capsize': 1.5,
        'capthick': 1,
        'markersize': 6,
        'markerfacecolor': 'none',  # Hollow marker
        'markeredgewidth': 1.2
    }
    
    # Process each jet type
    for i, (jet_type, (real_key, gen_key)) in enumerate(zip(jet_types, keys)):
        print(f"\nCreating ratio plot for {jet_type} jets...")
        
        # Load real dataset
        real_particles = load_real_dataset(datasets[real_key]['path'], datasets[real_key]['file'])
        
        # Load generated dataset
        gen_particles = load_gen_dataset(datasets[gen_key]['path'], datasets[gen_key]['file'])
        
        # Compute base EECs (n=2)
        if real_particles is not None and len(real_particles) > 0:
            real_eec = compute_eec(real_particles)
            
            if real_eec is not None:
                real_hist_base, real_errs_base = real_eec.get_hist_errs(0, False)
            
                # Compute higher order EECs and ratios for real data
                for n in range(3, 6):
                    real_enc = compute_eec(real_particles, n)
                    
                    if real_enc is not None:
                        real_midbins, real_bins = real_enc.bin_centers(), real_enc.bin_edges()
                        real_hist, real_errs = real_enc.get_hist_errs(0, False)
                        
                        # Calculate ratio
                        real_ratio = real_hist / real_hist_base
                        real_ratio_err = real_ratio * np.sqrt(
                            (real_errs/real_hist)**2 + (real_errs_base/real_hist_base)**2
                        )
                        
                        # Plot real ratio
                        axs[i].errorbar(
                            real_midbins, real_ratio,
                            xerr=(real_midbins - real_bins[:-1], real_bins[1:] - real_midbins),
                            yerr=real_ratio_err,
                            color=colors[n],
                            label=f'E{n}C/EEC (Real)',
                            markeredgecolor=colors[n],
                            **errorbar_opts_real
                        )
        
        # Compute base EECs (n=2) for generated data
        if gen_particles is not None and len(gen_particles) > 0:
            gen_eec = compute_eec(gen_particles)
            
            if gen_eec is not None:
                gen_hist_base, gen_errs_base = gen_eec.get_hist_errs(0, False)
            
                # Compute higher order EECs and ratios for generated data
                for n in range(3, 6):
                    gen_enc = compute_eec(gen_particles, n)
                    
                    if gen_enc is not None:
                        gen_midbins, gen_bins = gen_enc.bin_centers(), gen_enc.bin_edges()
                        gen_hist, gen_errs = gen_enc.get_hist_errs(0, False)
                        
                        # Calculate ratio
                        gen_ratio = gen_hist / gen_hist_base
                        gen_ratio_err = gen_ratio * np.sqrt(
                            (gen_errs/gen_hist)**2 + (gen_errs_base/gen_hist_base)**2
                        )
                        
                        # Plot generated ratio
                        axs[i].errorbar(
                            gen_midbins, gen_ratio,
                            xerr=(gen_midbins - gen_bins[:-1], gen_bins[1:] - gen_midbins),
                            yerr=gen_ratio_err,
                            color=colors[n],
                            label=f'E{n}C/EEC (Gen)',
                            markeredgecolor=colors[n],
                            **errorbar_opts_gen
                        )
        
        # Set scales and limits
        axs[i].set_xscale('log')
        axs[i].set_xlim(1e-3, 1)
        axs[i].set_ylim(0, 4)
        
        # Set labels and title
        axs[i].set_xlabel('R$_{L}$')
        axs[i].set_ylabel('ENC/EEC Ratio')
        axs[i].set_title(f'{jet_type} Jets')
        
        # Create custom legend with fewer entries
        handles, labels = axs[i].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[i].legend(by_label.values(), by_label.keys(), loc='upper left', frameon=False)
    
    # Adjust layout and save
    fig.tight_layout()
    os.makedirs('./eec_plots/plots', exist_ok=True)
    plt.savefig('./eec_plots/plots/eec_ratio_comparison.png')
    plt.close(fig)
    print("EEC ratio comparison plot saved")

# Main execution
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('./eec_plots/plots', exist_ok=True)
    
    # Create plots
    create_eec_plots()
    create_eec_ratio_plots()
    
    print("All plots created successfully!")

