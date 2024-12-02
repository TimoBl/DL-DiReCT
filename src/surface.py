import os
import csv
import sys
import numpy as np
import pandas as pd
import argparse
import trimesh
import nibabel as nib
import skimage.measure as measure
from scipy import spatial, ndimage
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import gaussian
from tca import topology



# global var
topo = topology()
spike_filter = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
dien = [101,10,11,12,13,26,28,125,17,18] + [112,49,50,51,52,58,60,125,53,54] + [14] # diencephalic structures we ignore


# get cortical labels
def get_labels(offset):
    lut = pd.read_csv('{}/fs_lut.csv'.format(os.path.dirname(os.path.realpath(sys.argv[0]))))
    lut = {l.Key: l.Label for _, l in lut.iterrows() if l.Label >= offset and l.Label <= 3000+offset}
    labels = list(filter(lambda x: ((x.startswith('lh') or x.startswith('rh')) and not 'nknown' in x and not 'corpuscallosum' in x and not 'Medial_wall' in x), lut.keys()))
    all_labels = labels + ['lh-MeanThickness', 'rh-MeanThickness']
    
    return lut, labels, all_labels

# remove spikes in
def clean_seg(seg):
    cc, nc = measure.label(seg, connectivity=2, return_num=True)
    cc_max = 1 + np.argmax(np.array([np.count_nonzero(cc == i) for i in range(1, nc + 1)]))
    seg_c = (cc==cc_max).astype(np.float64)
    seg_c = ndimage.correlate(seg_c, spike_filter)
    seg_c = np.where(seg_c == 1, 0, seg_c)
    seg_c = np.where(seg_c, 1, 0)
    return seg_c

# Transformation for FreeSurfer Space
def get_vox2ras_tkr(t1):
    ds = t1.header._structarr['pixdim'][1:4]
    ns = t1.header._structarr['dim'][1:4] * ds / 2.0
    v2rtkr = np.array([[-ds[0], 0, 0, ns[0]],
                       [0, 0, ds[2], -ns[2]],
                       [0, -ds[1], 0, ns[1]],
                       [0, 0, 0, 1]], dtype=np.float32)            
    return v2rtkr

# transform to FreeSurfer Space
def register(mesh, ref_volume, output, volume_info=None):
    
    # we transform mesh
    mesh = mesh.copy()
    affine = get_vox2ras_tkr(ref_volume)
    
    # apply affine for FS visualization and matching with the MRI
    mesh.vertices = nib.affines.apply_affine(affine, mesh.vertices)
    mesh.invert()
    
    nib.freesurfer.io.write_geometry(output, mesh.vertices, mesh.faces, create_stamp=None, volume_info=volume_info)

    return mesh

# generate mesh from seg using marching cubes
def gen_mesh(seg: np.ndarray) -> trimesh.base.Trimesh:
    #Topology correction algorithm taken from [CORTEXODE]
    sdf = -ndimage.distance_transform_cdt(seg) + ndimage.distance_transform_cdt(1-seg)
    sdf = gaussian(sdf.astype(float), sigma=0.5)
    sdf_topo = topo.apply(sdf, threshold=20)
    vert, fcs, _, val = measure.marching_cubes(sdf_topo, level=0)
    mesh = trimesh.base.Trimesh(vertices=vert, faces=fcs)
    return mesh

# move with DiReCT deformation Field    
def apply_deformation(mesh, def_field, step_size=0.05):
    
    vertices = mesh.vertices.copy().astype(precision)
    faces = mesh.faces.copy()
    thickness = np.zeros(len(vertices)).astype(precision)

    method = "linear" 
    x, y, z, intstep, vi = def_field.shape
    dx, dy, dz = np.arange(x), np.arange(y), np.arange(z)
    
    # we integrate our 10 time points
    print("Deforming Surface", end="")
    for i in range(0, 10, 1):
        
        vex = RegularGridInterpolator((dx, dy, dz), def_field[ :, :, :, i, 0], method=method)
        vey = RegularGridInterpolator((dx, dy, dz), def_field[ :, :, :, i, 1], method=method)
        vez = RegularGridInterpolator((dx, dy, dz), def_field[ :, :, :, i, 2], method=method) 
        
        for j in np.arange(0, 1, step_size):
            v = np.array([vex(vertices), vey(vertices), vez(vertices)]).T
            vertices += (step_size * v)
            thickness +=  np.linalg.norm(step_size * v, axis=1)
        print(".", end="", flush=True)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    return mesh, thickness


# used to match points to the segmentation
def label_points(points, aparc, max_dst=3):

    # build tree for labeling
    coords_parc = np.array(np.where(aparc >= 1000)).T # we limit to cortex, and offset to be at center of voxel
    tree = spatial.cKDTree(coords_parc + 0.5) # offset to be at the center of voxel

    # and label for thickness
    nearest_distances, nearest_indices = tree.query(points, k=1) # we use the wm, it's more stable
    nearest_coords = coords_parc[nearest_indices] 
    nearest_labels = np.array(aparc[nearest_coords[:, 0], nearest_coords[:, 1], nearest_coords[:, 2]], dtype=int)
    nearest_labels[nearest_distances > max_dst] = -1 # contrain distance,
    
    return nearest_labels


# we get the mesh using binary mesh as ref
def get_mesh(labels, white_srf, pial_srf):
    
    faces = white_srf.faces.flatten()
    vertices = white_srf.vertices[labels]
    
    # check if face has all vertices labeled
    labeled = np.take(labels, faces) 
    labeled = labeled.reshape(pial_srf.faces.shape)
    faces = pial_srf.faces[labeled.all(axis=1)] # all vertices must be labeeld
    
    return trimesh.Trimesh(vertices=pial_srf.vertices, faces=faces)


# calculate thickness
# We calculate the thickness by labeling our vertices and taking their thickness (works only when voxel space 1mm ISO!)
def get_thickness_stats(thickness, white_srf, pial_srf, aparc):
    nearest_labels = label_points(white_srf.vertices, aparc)
    out_thickness = {}

    for k, v in lut.items():
        values = thickness[(nearest_labels == k)]
        out_thickness[v] = values[values > 0].mean() if sum(values > 0) > 0 else np.nan

    out_thickness["lh-hemi"] = thickness[np.isin(nearest_labels, lh_labels)].mean()
    out_thickness["rh-hemi"] = thickness[np.isin(nearest_labels, rh_labels)].mean()
    
    return out_thickness


# calculate surface
# We label each face by region, and then construct the mesh to calculate the area (works only when voxel space 1mm ISO!)
def get_surface_stats(white_srf, pial_srf, aparc):
    nearest_labels = label_points(white_srf.vertices, aparc)
    out_surface = {}

    for k, v in lut.items():
        labels = (nearest_labels == k)
        pial_mesh = get_mesh(labels, white_srf, pial_srf)
        out_surface[v] = pial_mesh.area if pial_mesh.area > 0 else np.nan # we don't save one

    out_surface["lh-hemi"] = get_mesh(np.isin(nearest_labels, lh_labels), white_srf, pial_srf).area
    out_surface["rh-hemi"] = get_mesh(np.isin(nearest_labels, rh_labels), white_srf, pial_srf).area
    
    return out_surface


# helper
def mkdir(OUT_DIR):
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

def write_stats(stats, subject_id, fname, label_names):    
    with open(fname, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['SUBJECT'] + label_names)
        writer.writerow([subject_id] + list(stats))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract surface stats')
    parser.add_argument("velocity_file")
    parser.add_argument("seg_file")
    parser.add_argument("aparc_file")
    parser.add_argument("subject_id")
    args = parser.parse_args()
    
    velocity_file = args.velocity_file
    seg_file = args.seg_file
    aparc_file = args.aparc_file
    subject_id = args.subject_id
    dst_dir = os.path.dirname(seg_file)

    lowmem = True # if lowmen -> likely we can use this by default
    precision = np.float32 if lowmem else np.float64 

    # load files
    ref_file = nib.load(seg_file)
    seg, affine = ref_file.get_fdata().astype(precision), ref_file.affine
    aparc = nib.load(aparc_file).get_fdata().astype(precision)
    velocity_field = nib.load(velocity_file).get_fdata().astype(precision)

    # labels
    offset = 0 if np.max(aparc) <= 3000 else 10000
    lut, labels, all_labels = get_labels(offset)
    lut = {k: v for v, k in lut.items() if k >= 1000+offset and k <= 3000+offset}

    labl = np.unique(aparc).astype(int)
    lh_labels = labl[(labl >= 1000+offset) & (labl < 2000+offset)]
    rh_labels = labl[(labl >= 2000+offset) & (labl < 3000+offset)]

    # get wm surface mesh
    white = clean_seg(np.clip((seg == 3) + np.isin(aparc, dien), 0, 1))
    white_srf = gen_mesh(white)
    white_srf = trimesh.smoothing.filter_humphrey(white_srf)

    # get pial mesh and thickness measurements
    pial_srf, thickness = apply_deformation(white_srf, velocity_field)
    pial_srf = trimesh.smoothing.filter_humphrey(pial_srf)

    # get metrics
    out_thickness = get_thickness_stats(thickness, white_srf, pial_srf, aparc)
    out_surface = get_surface_stats(white_srf, pial_srf, aparc)

    # save metrics
    write_stats(list(out_thickness.values()), subject_id, '{}/result-thick-surf.csv'.format(dst_dir), list(out_thickness.keys())) # thickness
    write_stats(list(out_surface.values()), subject_id, '{}/result-surf.csv'.format(dst_dir), list(out_surface.keys())) # surface

    # save surfaces
    mkdir(dst_dir + "/srf/")
    register(white_srf, ref_file, dst_dir + "/srf/" + '/white')
    register(pial_srf, ref_file, dst_dir  + "/srf/" + '/pial')

    print('mean thick: {:.3f} mm'.format(np.mean(list(out_thickness.values())[-2:])))
    print('total surface: {:.3f} mm^2'.format(np.sum(list(out_surface.values())[-2:])))


    # Closest estimate 
    tree = spatial.cKDTree(white_srf.vertices) 
    closest_thickness, idx = tree.query(pial_srf.vertices, k=1)
    closest_thickness = get_thickness_stats(closest_thickness, white_srf, pial_srf, aparc)
    write_stats(list(closest_thickness.values()), subject_id, '{}/result-thick-nearest.csv'.format(dst_dir), list(closest_thickness.keys()))
