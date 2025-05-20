import numpy as np
import pandas as pd
import pickle
import lzma
import argparse as ap
import os
from shark import SharkSnap

data_dir = "/mnt/home/drennehan/ceph/simulating_the_universe/" \
           "overdensity_intersections/sims/shark/1024_200/STU/default"
shark_fmt = "./data/SharkSnap_{snap:03d}.pkl"
save_file_fmt = "./data/desc_matrix_{:03d}.pkl"

id_col = "global_id_galaxy"
desc_col = "global_descendant_id_galaxy"

sentinel32 = np.uint32(0xFFFFFFFF)
sentinel64 = np.uint64(0xFFFFFFFF)
nsubvols = 8
invalid = (nsubvols << 32) | sentinel32

parser = ap.ArgumentParser()
parser.add_argument("snap_list_file",
                    help="One snapshot number per line from highest to lowest.")
parser.add_argument("--cache", action=ap.BooleanOptionalAction,
                    default=True, help="Cache SharkSnap to disk")
args = parser.parse_args()

snap_list = np.loadtxt(args.snap_list_file, dtype=int)[::-1]
print("Will process snapshots (latest to earliest):", snap_list, "\n")

snap_cache: dict[int, SharkSnap] = {}

def get_snap(idx: int) -> SharkSnap:
    """Load or build the SharkSnap for zero-padded snapshot idx."""
    if idx in snap_cache:
        return snap_cache[idx]

    shark_file = shark_fmt.format(snap=idx)
    if args.cache and os.path.isfile(shark_file):
        print(f"Loading SharkSnap {idx:03d}")
        shark_snap = pickle.load(lzma.open(shark_file, "rb"))
    else:
        print(f"Building SharkSnap {idx:03d}")
        # PASS idx as a *zero-padded string* internally
        shark_snap = SharkSnap(data_dir, str(idx).zfill(3), nsubvols)
        if args.cache:
            with lzma.open(shark_file, "wb") as f:
                pickle.dump(shark_snap, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Caching SharkSnap {shark_file}")

    snap_cache[idx] = shark_snap
    return shark_snap

# Go through the earliest snapshots first and then loop through
# every other snapshot, building the snap_cache along the way to
# prevent multiple loads.
#
# Start with the global_descendant_id_galaxy in the earlier snapshot
# and compare to the global_id_galaxy in the subsequent snapshot.
# Only have to do this for pairs of snapshots since only the
# descendant/progenitor connection is meaningful.
#
# Want to end up with an array for each snapshot that follows
# the descendant pathway to the final snapshot.
for i, snap_i in enumerate(snap_list):
    prog_snap = get_snap(snap_i)
    prog_snap.df.set_index(id_col)

    # Use the same ngal value for all descendants
    ngal = prog_snap.ngal

    prog_col_name = "snap{:03d}".format(snap_i)
    # Contains the global_id_galaxy for all galaxies
    desc_matrix = {prog_col_name: prog_snap.df[id_col].to_numpy()}

    # Go through every other snapshot 
    for j, snap_j in enumerate(snap_list[i + 1:]):
        desc_snap = get_snap(snap_j)
        desc_snap.df.set_index(id_col)

        desc_col_name = f"snap{snap_j:03d}"
        desc_matrix.update({desc_col_name: np.zeros(ngal, dtype=np.uint64)})

        print(f"Preparing {prog_col_name} -> {desc_col_name}.")
        # Look for all descendant galaxies
        for k in range(ngal):
            desc_id = prog_snap.df.loc[prog_snap.df[id_col] == desc_matrix[prog_col_name][k], desc_col]
            if not desc_id.empty:
                desc_matrix[desc_col_name][k] = desc_id.values[0]
            else:
                desc_matrix[desc_col_name][k] = invalid

        # Reset prog_snap because only comparing adjacent 
        # snapshots
        prog_snap = desc_snap
        prog_col_name = desc_col_name

    desc_matrix = pd.DataFrame(desc_matrix)
    save_file = save_file_fmt.format(snap_i)
    print(f"Writing desc_matrix to {save_file}.")
    with lzma.open(save_file, "wb") as f:
        pickle.dump(desc_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

