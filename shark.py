import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SharkSnap:
    overdensities = None
    df = None
    tree = None
    cosmo = None
    figsize = (6.4, 4.8)
    dpi = 150
    axes_fontsize = 14
    
    def __init__(self, data_path, snapshot, nsubvols, min_gal_mass_Msun_h=1.0e8):
        self.min_gal_mass_Msun_h = min_gal_mass_Msun_h
        self.data_path = data_path
        self.snapshot = snapshot
        self.nsubvols = nsubvols

        # open the first file to define the column names
        galaxies_file_fmt = "{:s}/{:d}/{:d}/galaxies.hdf5"
        
        columns = []
        with h5py.File(galaxies_file_fmt.format(data_path, snapshot, 0)) as f:
            self.lbox = f["run_info/lbox"][()]
            self.volume = f["run_info/effective_volume"][()]
            self.h = f["cosmology/h"][()]
            self.omega_m = f["cosmology/omega_m"][()]
            self.omega_l = f["cosmology/omega_l"][()]
            self.omega_b = f["cosmology/omega_b"][()]
            self.redshift = f["run_info/redshift"][()]
            self.particle_mass = f["run_info/particle_mass"][()]

        from astropy.cosmology import FlatLambdaCDM
        self.cosmo = FlatLambdaCDM(Om0=self.omega_m, H0=100.*self.h)
        
        for i in range(nsubvols):
            with h5py.File(galaxies_file_fmt.format(data_path, snapshot, i)) as f:
                data = {}
                for key in f["galaxies"].keys():
                    data.update({key: np.array(f["galaxies/{:s}".format(key)])})

                # Keep track of the subvolume because the ID is unique to the subvolume only
                data.update({"subvolume": np.uint64(i) * np.ones(len(data[key]), dtype=np.uint64)})
                if i > 0:
                    self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)
                    # Add each subvolume to the total volume
                    self.volume += f["run_info/effective_volume"][()]  # cMpc/h
                else:
                    self.df = pd.DataFrame(data)

        sentinel32 = np.uint32(2**32 - 1)  # 0xFFFFFFFF

        id_gal = (
            self.df["id_galaxy"]
                .astype("Int64")
                .fillna(-1)
                .replace(-1, sentinel32)
                .to_numpy()
                .astype(np.uint32)
        )

        id_desc = (
            self.df["descendant_id_galaxy"]
                .astype("Int64")
                .fillna(-1)
                .replace(-1, sentinel32)
                .to_numpy()
                .astype(np.uint32)
        )

        subvol = self.df["subvolume"].to_numpy().astype(np.uint64)

        self.df["global_descendant_id_galaxy"] = (subvol << 32) | id_desc
        self.df["global_id_galaxy"] = (subvol << 32) | id_gal

        # These keys are split into disk and bulge components, useful to have the
        # sum total.
        keys = ["mstars", "mstars_metals", "mmol", "matom", "mgas", "mgas_metals"]
        
        # Derived columns
        for key in keys:
            self.df[key] = self.df["{:s}_disk".format(key)] + self.df["{:s}_bulge".format(key)]

        # Keep only those with valid stellar masses, and are either centrals or valid satellites
        self.df = self.df[(self.df["mstars"] > min_gal_mass_Msun_h) & ((self.df["type"] == 0) | (self.df["type"] == 1))]
        self.df.reset_index(drop=True, inplace=True)
        
        # Clean up the periodic boundaries
        keys = ["position_x", "position_y", "position_z"]
        for key in keys:
            self.df[key] = self.df[key] % self.lbox

        # IMPORTANT: Must set keys here that do not need to have a floor set
        # for log operations.
        keys = ["descendant_id_galaxy",
                "id_galaxy",
                "id_halo",
                "id_halo_tree",
                "id_subhalo",
                "id_subhalo_tree",
                "on_hydrostatic_eq",
                "position_x",
                "position_y",
                "position_z",
                "redshift_merger",
                "type",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "subvolume",
                "global_id_galaxy",
                "global_descendant_id_galaxy"]
        for col in self.df.columns:
            if col in keys:
                continue

            # Set to a very small number to identify zeros later
            self.df.loc[self.df[col] <= 0, col] = np.float32(1.0e-20)

        # More common derived quantities
        self.df["sfr"] = self.df["sfr_disk"] + self.df["sfr_burst"]
        self.df["ssfr"] = self.df["sfr"] / self.df["mstars"]

        # Total number of valid galaxies
        self.ngal = len(self.df)
        
        print("Found {:d} galaxies in L = {:g} cMpc/h".format(len(self.df), np.cbrt(self.volume)))

        # Set up the initial overdensity filters
        self.overdensities = {"filters": {}, "hist3d": {}}

    def convert_filter_radius_to_key(self, filter_radius):
        return int(np.floor(filter_radius).astype(np.uint32))
        
    def get_bins_from_edges(self, nedges):
        if nedges <= 1:
            raise ValueError("There must be >1 edge to compute the bins.")

        return np.linspace(0, self.lbox, nedges)
            
    def get_bin_width(self, nedges):
        if nedges <= 1:
            raise ValueError("There must be >1 edge to compute the bin width.")

        return self.lbox / (nedges - 1)

    def generate_overdensity_grid(self, xyz, bins):
        hist3d, _ = np.histogramdd(xyz, bins=[bins, bins, bins])
        hist3d /= np.mean(hist3d)
        hist3d -= 1.0
        
        return hist3d
        
    def compute_galaxy_overdensities(self, filter_radius=None, nedges=None, cache=True, random=None):
        if filter_radius is None and nedges is None:
            raise Exception("Must set either a filter_radius in cMpc/h or nedges to do a 3D histogram.")

        if random is not None:
            if not isinstance(random, int):
                raise ValueError("If you set random, it must be the number of samples.")
                
            rand_xyz = np.random.rand(random, 3) * self.lbox

        xyz = np.column_stack([
            self.df["position_x"],
            self.df["position_y"],
            self.df["position_z"]
        ])
        
        # Do everything relative to a 3D histogram grid given by nedges
        if nedges is not None:
            if random is not None:
                raise ValueError("Using random locations is not implemented for gridded overdensities.")
                
            bins = self.get_bins_from_edges(nedges)
            
            print("Using nedges={:d} edges to create a 3D histogram.".format(int(nedges)))
            print("The number of bins is {:d}".format(len(bins)-1))
            print("The bin width is {:g} cMpc/h".format(bins[1] - bins[0]))
            hist3d = self.generate_overdensity_grid(xyz, bins)

            # Determine the bin indices of all of the galaxies
            bin_indices = np.floor(xyz / (bins[1] - bins[0])).astype(np.uint64)

            overdensities = np.zeros(len(self.df["mstars"]), dtype=np.float32)
            for i in range(len(overdensities)):
                overdensities[i] = hist3d[bin_indices[i, 0], bin_indices[i, 1], bin_indices[i, 2]]
    
            self.df["hist3d_delta_{:d}".format(int(nedges))] = overdensities
                
        if filter_radius is not None:
            # reset the filter radius to the nearest floored integer
            filter_key = self.convert_filter_radius_to_key(filter_radius)
            filter_radius = float(filter_key)
            
            # center on each galaxy and manually compute the overdensity at a given
            # filter radius
            if not cache or self.tree is None:
                from scipy.spatial import cKDTree
                self.tree = cKDTree(xyz, boxsize=self.lbox)

            if random is not None:
                number_nearby = np.zeros(random, dtype=np.uint32)
                for i in range(random):
                    lists = self.tree.query_ball_point(rand_xyz[i], r=filter_radius)
                    number_nearby[i] = len(lists)-1

                rho = number_nearby / ((4. * np.pi / 3.) * (filter_radius)**3.)
                rho_mean = self.ngal / self.lbox**3.

                return (rho - rho_mean) / rho_mean
            else:
                number_nearby = np.zeros(self.ngal, dtype=np.uint32)
                idx_nearby = np.zeros(self.ngal, dtype=np.uint64)
                for i in range(self.ngal):
                    lists = self.tree.query_ball_point(xyz[i], r=filter_radius)
                    idx_nearby[i] = lists[np.argmax(self.df["mstars"].iloc[lists])]
                    number_nearby[i] = len(lists)-1
    
                rho = number_nearby / ((4. * np.pi / 3.) * (filter_radius)**3.)
                rho_mean = self.ngal / self.lbox**3.
    
                overdensities = (rho - rho_mean) / rho_mean
                
                self.df["filter_delta_{:d}".format(filter_key)] = overdensities
                self.df["filter_nearest_{:d}".format(filter_key)] = idx_nearby
        
    def plot_histogram2d(self, x_key, y_key, nbins=64, transform=None, mask=None):
        if mask is None:
            mask = (self.df["mstars"] > 0.)

        x_label = x_key
        y_label = y_key

        x = self.df[x_key][mask]
        y = self.df[y_key][mask]
        
        if isinstance(transform, list):
            if transform[0] == "log":
                x = np.log10(self.df[x_key][mask])
                x_label = "log({:s})".format(x_key)
                
            if transform[1] == "log":
                y = np.log10(self.df[y_key][mask])
                y_label = "log({:s})".format(y_key)

        bins_x = np.linspace(np.amin(x), np.amax(x), nbins)
        bins_y = np.linspace(np.amin(y), np.amax(y), nbins)
        
        X, Y = np.meshgrid(bins_x, bins_y)
        
        hist, _, __ = np.histogram2d(x, y, bins=[bins_x, bins_y])
        
        plt.figure(figsize=self.figsize, dpi=self.dpi, facecolor="w")
        plt.xlabel(x_label, fontsize=self.axes_fontsize)
        plt.ylabel(y_label, fontsize=self.axes_fontsize)
        if len(self.df[x_key][mask]) > 1000:
            from matplotlib.colors import LogNorm
            plt.pcolormesh(X, Y, hist.T, norm=LogNorm())
        else:
            plt.pcolormesh(X, Y, hist.T)
        ax = plt.gca()
        ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
        plt.show()
        plt.close()

