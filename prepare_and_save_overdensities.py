import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import pickle
import lzma
from shark import SharkSnap
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("filter_radius")
parser.add_argument("snap_list_file")
args = parser.parse_args()

data_dir = "/mnt/home/drennehan/ceph/simulating_the_universe/overdensity_intersections/sims/shark/1024_200/STU/default"
redshifts_file = "./info/output_redshifts_fixed.txt"
save_file = "./data/SharkSnap_{:s}.pkl".format(str(args.snapshot).zfill(3))
overwrite_save = True
nsubvols = 8

redshift_map = np.loadtxt(redshifts_file)
snap_list = np.loadtxt(args.snap_list_file, dtype=int)
redshifts = redshift_map[snap_list][:, 1]
redshifts

zX = SharkSnap(data_dir, int(args.snapshot), nsubvols)

# radius of the sample sphere, should be the same for all samples during a comparison
filter_radius = float(args.filter_radius) / zX.h

zX.compute_galaxy_overdensities(filter_radius=filter_radius)

with lzma.open(save_file, "wb") as f:
    pickle.dump(zX, f, protocol=pickle.HIGHEST_PROTOCOL)

