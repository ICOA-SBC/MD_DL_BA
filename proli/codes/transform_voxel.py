
from math import ceil
import numpy as np
from sklearn.gaussian_process.kernels import RBF

column_names = ['x', 'y', 'z', 'B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', \
                 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'molcode', \
                 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring'] 
feature_names = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', \
                 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'molcode', \
                 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring'] 

columns_dict = {name: i for i, name in enumerate(column_names)}
features_dict = {name: i for i, name in enumerate(feature_names)}

def apply_gauss_and_convert_to_grid(coords, features, grid_resolution=1.0, max_dist=12.0):
    expected_size = 2*max_dist + 1
    
    c_shape, f_shape = coords.shape, features.shape
    #N = len(coords)
    num_features = f_shape[1]
    #box_size = ceil(2 * max_dist / grid_resolution + 1)
    
    atoms = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal']
    
    # get dimension of complex (sample)
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    xyz_center = (xyz_max + xyz_min)/2.0
    
    # filter dimension so it fits a bounding box of max 25 Angstrom per side
    in_box = ((coords >= xyz_center - 0.5 - max_dist) & (coords <= xyz_center + 0.5 + max_dist)).all(axis=1) 
    filtered_features = features[in_box] # lines, 19
    filtered_coords = coords[in_box] # lines, 3

    if len(filtered_coords)==0: # TODO: quick fix when nothing is in the bounding box!
        print(f"-------------------------−−> Found no coords! {num_features=}, {filtered_features=} ")
        #print(f"Total volume of protein-ligand: {xyz_min} x {xyz_max} \n\t with a center at {xyz_center}")
        #print(f"Keep coords between {xyz_center - 0.5 - max_dist=} and {xyz_center + 0.5 + max_dist=}")
        #print(f"{in_box.shape=}")
        #print(f"{features=}")
        #print(f"{coords=}")
    else:
        #print(f"Sample size {features.shape[0]} --> filtered {filtered_features.shape[0]}")
        # recompute dimension
        xyz_min = filtered_coords.min(axis=0)
        xyz_max = filtered_coords.max(axis=0)
        #xyz_center = (xyz_max + xyz_min)/2.0
        #print(f"Dimension of the protein-ligand filtered: {xyz_min} x {xyz_max} \n\t with a new center at {xyz_center}") 
    
    # Same bins for all atoms
    xi, yi, zi = get_bins(spacing=grid_resolution, expected_size=expected_size, xyz_min=xyz_min, xyz_max=xyz_max)
    nx, ny, nz = xi.size, yi.size, zi.size
    #if nx < expected_size or  ny < expected_size or  nz < expected_size or nx > expected_size or  ny > expected_size or  nz > expected_size :
        # TODO: can be one off in any dimension, if it happens, last will be dropped at the end of the process
    #    print(f"-------------------------−−> Wrong size for {nx=} {ny=} {nz=} ")
        #print(f"\t {xi=}")
        #print(f"\t {yi=}")
        #print(f"\t {zi=}")
              
    final_grid = np.zeros(shape=(nx, ny, nz, num_features), dtype=np.float32)
    #print(f"Final shape will be {final_grid.shape}")
    
    # Apply gauss on each type of atom (necessary if we eventually apply a normalization in case of a superposition of the Gauss process
    for atom in atoms:
        all_atoms_of_interest = filtered_features[:,features_dict[atom]]>0 # BUG feature_dict
        filtered_features_with_atom = filtered_features[all_atoms_of_interest]
        filtered_coords_with_atom = filtered_coords[all_atoms_of_interest]
        #print(f"Working on {atom=} found {len(filtered_features_with_atom)} instances")
        
        if len(filtered_features_with_atom)>0:
            grid_for_atom = gaussian_blur(filtered_coords_with_atom, xi, yi, zi)
            final_grid += combine_features(features_dict[atom], grid_for_atom, filtered_coords_with_atom, 
                                filtered_features_with_atom, nx, ny, nz, xyz_min)
    # Quick fix when grid is not expected size
    #if ny>expected_size:
    #print(f"{final_grid.shape=}")
    return final_grid[0:expected_size, 0:expected_size, 0:expected_size, :]


def get_bins(spacing, expected_size, xyz_min=None, xyz_max=None):
    xm, ym, zm = xyz_min
    xM, yM, zM = xyz_max

    def update(m, M):
        if M-m < expected_size:
            return M + (expected_size - (M-m))
        else:
            return M
    
    xM, yM, zM = update(xm, xM), update(ym, yM), update(zm, zM)
    xi = np.arange(xm, xM, spacing)
    yi = np.arange(ym, yM, spacing)
    zi = np.arange(zm, zM, spacing)
    
    #if len(yi)<25 or len(yi)>25 or len(xi)<25 or len(xi)>25 or len(zi)<25 or len(zi)>25:
    #    print(f"-------------------------−−> anomaly {len(xi)=} {len(yi)=} {len(zi)=}")
        #print(f"{xm=}, {xM=}")
        #print(f"{ym=}, {yM=}")
        #print(f"{zm=}, {zM=}")
    
    return xi, yi, zi

def apply_kernel(coord, xi, yi, zi, sigma, atom_grid):
    #  Find subgrid
    nx, ny, nz = xi.size, yi.size, zi.size

    bound = int(2 * sigma) # extend of the gaussian originally 4
    
    x, y, z = coord # only one coord at a time
    binx = np.digitize(x, bins=xi) # Return the indices of the bins to which each value in input array belongs.
    biny = np.digitize(y, yi)
    binz = np.digitize(z, zi)
    #print(f"Bin {binx.shape=} {biny.shape=} {binz.shape=} ")

    min_bounds_x, max_bounds_x = max(0, binx - bound), min(nx, binx + bound) 
    min_bounds_y, max_bounds_y = max(0, biny - bound), min(ny, biny + bound)
    min_bounds_z, max_bounds_z = max(0, binz - bound), min(nz, binz + bound)
    #print(f"bounds: {min_bounds_z=} {max_bounds_z=} for {binz=} and {bound=}")
    
    X, Y, Z = np.meshgrid(xi[min_bounds_x: max_bounds_x],
                          yi[min_bounds_y: max_bounds_y],
                          zi[min_bounds_z:max_bounds_z],
                          indexing='ij')
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

    #  Compute RBF
    rbf = RBF(sigma)
    subgrid = rbf(coord, np.c_[X, Y, Z])
    subgrid = subgrid.reshape((max_bounds_x - min_bounds_x,
                               max_bounds_y - min_bounds_y,
                               max_bounds_z - min_bounds_z))
    # update grid
    atom_grid[min_bounds_x: max_bounds_x, min_bounds_y: max_bounds_y, min_bounds_z:max_bounds_z] += subgrid


def gaussian_blur(coords, xi, yi, zi, sigma=1.):
    """
    Compute RBF on a set of coords,
    We loop over each coord to compute only a neighbourhood and add it to the right grid
    TODO: sigma meaning
    """
    nx, ny, nz = xi.size, yi.size, zi.size
    atom_grid = np.zeros(shape=(nx, ny, nz))
    
    for coord in coords:
        apply_kernel(coord, xi=xi, yi=yi, zi=zi, sigma=sigma, atom_grid=atom_grid)
        
    return atom_grid

def combine_features(atom_column, grid_for_atom, filtered_coords_with_atom, filtered_features_with_atom, nx, ny, nz, xyz_min):
    # combine Gauss on atom and other features 
    num_features= filtered_features_with_atom.shape[1]
    complete_grid = np.zeros(shape=(nx, ny, nz, num_features))

    # copy all features
    # shift all coordinates
    translated_coord = filtered_coords_with_atom - xyz_min
    translated_coord = translated_coord.round().astype(int)
    filtering = ((translated_coord >= 0) & (translated_coord < [25,25,25])).all(axis=1)

    for (x, y, z), f in zip(translated_coord[filtering], filtered_features_with_atom[filtering]):
        complete_grid[x, y, z] += f 
      
    # update with grid_for_atom
    complete_grid[:,:,:,atom_column]= grid_for_atom

    return complete_grid