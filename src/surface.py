#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import sys
import numpy as np
import pandas as pd
import argparse
import trimesh
import nibabel as nib
from skimage import measure
from skimage import metrics
import skimage.morphology as morphology
from scipy import spatial, ndimage
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import gaussian
from tca import topology
#from src import get_crop, apply_crop, apply_uncrop


# In[2]:


spike_filter = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])


# In[3]:


dien = [101,10,11,12,13,26,28,125,17,18] + [112,49,50,51,52,58,60,125,53,54] + [14] # diencephalic structures we ignore


# In[4]:


topo = topology()


# In[5]:


def get_crop(volume):
    nonempty = np.argwhere(volume)
    top_left = nonempty.min(axis=0)
    bottom_right = nonempty.max(axis=0)
    
    return (top_left, bottom_right)


# In[6]:


# get cortical labels
def get_labels(offset):
    #lut = pd.read_csv('{}/fs_lut.csv'.format(os.path.dirname(os.path.realpath(sys.argv[0]))))
    lut = pd.read_csv('fs_lut.csv')
    lut = {l.Key: l.Label for _, l in lut.iterrows() if l.Label >= offset and l.Label <= 3000+offset}
    labels = list(filter(lambda x: ((x.startswith('lh') or x.startswith('rh')) and not 'nknown' in x and not 'corpuscallosum' in x and not 'Medial_wall' in x), lut.keys()))
    all_labels = labels + ['lh-MeanThickness', 'rh-MeanThickness']
    return lut, labels, all_labels


# In[7]:


def clean_seg(seg):
    cc, nc = measure.label(seg, connectivity=2, return_num=True)
    cc_max = 1 + np.argmax(np.array([np.count_nonzero(cc == i) for i in range(1, nc + 1)]))
    seg_c = (cc==cc_max).astype(np.float64)
    seg_c = ndimage.correlate(seg_c, spike_filter)
    seg_c = np.where(seg_c == 1, 0, seg_c)
    seg_c = np.where(seg_c, 1, 0)
    return seg_c


# In[8]:


# Transformation for FreeSurfer Space
def get_vox2ras_tkr(t1):
    
    ds = t1.header._structarr['pixdim'][1:4]
    ns = t1.header._structarr['dim'][1:4] * ds / 2.0
    v2rtkr = np.array([[-ds[0], 0, 0, ns[0]],
                       [0, 0, ds[2], -ns[2]],
                       [0, -ds[1], 0, ns[1]],
                       [0, 0, 0, 1]], dtype=np.float32)            
    return v2rtkr


# In[9]:


# transform to FreeSurfer Space
def register(mesh, ref_volume, output, volume_info=None, translate=[0, 0, 0]):
    
    # we transform mesh
    mesh = mesh.copy()
    affine = get_vox2ras_tkr(ref_volume)
    
    # apply affine for FS visualization and matching with the MRI
    mesh.vertices = nib.affines.apply_affine(affine, mesh.vertices + translate)  
    mesh.invert()
    
    nib.freesurfer.io.write_geometry(output, mesh.vertices, mesh.faces, create_stamp=None, volume_info=volume_info)
    return mesh


# In[10]:


'''
# DEPRICATED
def gen_mesh(seg: np.ndarray) -> trimesh.base.Trimesh:
    #Topology correction algorithm taken from [CORTEXODE]
    sdf = -ndimage.distance_transform_cdt(seg) + ndimage.distance_transform_cdt(1-seg)
    sdf = gaussian(sdf.astype(float), sigma=0.5)
    sdf_topo = topo.apply(sdf, threshold=20)
    vert, fcs, _, val = measure.marching_cubes(sdf_topo, level=0)
    mesh = trimesh.base.Trimesh(vertices=vert, faces=fcs)
    mesh = trimesh.smoothing.filter_humphrey(mesh)
    return mesh
'''


# In[46]:


# get wm
def fill_wm(wm_prob, aparc):
    # to deal with subcortical regions
    mask = np.isin(aparc, dien) | (aparc == 41) | (aparc == 2) # diencephalic + wm
    mask = morphology.binary_erosion(morphology.binary_closing(mask)) # not ideal but works
    wm_prob[mask.astype(bool)] = 1
    
    # correct topology
    #sdf_topo = topo.apply(wm_prob, threshold=0.5)
    
    # makes sure to remove border 
    #sdf_topo *= clean_seg(sdf_topo > 0) # fully connected
    
    return wm_prob

# # create marching cubes mesh
def gen_mesh(prob):
    vert, fcs, _, val = measure.marching_cubes(prob, level=0.5, allow_degenerate=False) # use 0.5 as level set
    srf = trimesh.base.Trimesh(vertices=vert, faces=fcs)
    trimesh.repair.fix_normals(srf, multibody=False)
    return srf


# In[47]:


# DiReCT deformation Field    
def apply_deformation(points, def_field, step_size=0.1, order=3):
    
    points = points.copy().astype(precision)
    thickness = np.zeros(len(points)).astype(precision)
    
    for i in range(0, 10, 1):

        vx = def_field[ :, :, :, i, 0]
        vy = def_field[ :, :, :, i, 1]
        vz = def_field[ :, :, :, i, 2]

        for j in np.arange(0, 1, step_size):
            v = np.array([
                ndimage.map_coordinates(vx, points.T, order=order),
                ndimage.map_coordinates(vy, points.T, order=order),
                ndimage.map_coordinates(vz, points.T, order=order)
            ]).T

            points += (step_size * v)
            thickness +=  np.linalg.norm(step_size * v, axis=1)

        print(".", end="", flush=True)
        
    return points, thickness


# In[48]:


# used to match points to the segmentation
def label_points(points, aparc, max_dst=3):

    # build tree for labeling
    coords_parc = np.array(np.where(aparc >= 1000)).T # we limit to cortex
    tree = spatial.cKDTree(coords_parc + 0.5) # offset to be at the center of voxel

    # and label for thickness
    nearest_distances, nearest_indices = tree.query(points, k=1) # we use the wm, it's more stable
    nearest_coords = coords_parc[nearest_indices] 
    nearest_labels = np.array(aparc[nearest_coords[:, 0], nearest_coords[:, 1], nearest_coords[:, 2]], dtype=int)
    nearest_labels[nearest_distances > max_dst] = 0 # contrain distance, so we set as unknown
    
    return nearest_labels


# In[49]:


# we get the mesh using binary mesh as ref
def get_mesh(pial_srf, nearest_labels):
    
    # get faces
    faces = pial_srf.faces.flatten()
    vertices = pial_srf.vertices[nearest_labels]
    
    # check if face has all vertices labeled
    labeled = np.take(nearest_labels, faces) 
    labeled = labeled.reshape(pial_srf.faces.shape)
    faces = pial_srf.faces[labeled.all(axis=1)] # all vertices must be labeeld
    
    return trimesh.Trimesh(vertices=pial_srf.vertices, faces=faces)


# In[50]:


# calculate thickness
# We calculate the thickness by labeling our vertices and taking their thickness (works only when voxel space 1mm ISO!)
def get_thickness_stats(thickness, nearest_labels):
    out_thickness = {}

    for k, v in lut.items():
        values = thickness[(nearest_labels == k)]
        out_thickness[v] = values[values > 0].mean() if sum(values > 0) else np.nan

    out_thickness["lh-hemi"] = thickness[np.isin(nearest_labels, lh_labels)].mean()
    out_thickness["rh-hemi"] = thickness[np.isin(nearest_labels, rh_labels)].mean()
    
    return out_thickness


# In[51]:


# calculate surface
# We label each face by region, and then construct the mesh to calculate the area (works only when voxel space 1mm ISO!)
def get_surface_stats(pial_srf, nearest_labels):
    out_surface = {}

    for k, v in lut.items():
        lbl = (nearest_labels == k)
        pial_mesh = get_mesh(pial_srf, lbl)
        out_surface[v] = pial_mesh.area if pial_mesh.area > 0 else np.nan # we don't save one

    out_surface["lh-hemi"] = get_mesh(pial_srf, np.isin(nearest_labels, lh_labels)).area
    out_surface["rh-hemi"] = get_mesh(pial_srf, np.isin(nearest_labels, rh_labels)).area
    
    return out_surface


# In[79]:


# used to calculate freesurfer distance
def get_freesurfer_distance(white_srf, pial_srf):
    # closest distance from white to pial
    tree = spatial.cKDTree(pial_srf.vertices) 
    closest_thickness_pial, idx = tree.query(white_srf.vertices, k=1)

    # from those points calculate the closest distance to white
    tree = spatial.cKDTree(white_srf.vertices) 
    closest_thickness_wm, idx = tree.query(pial_srf.vertices[idx], k=1)
    
    return (closest_thickness_pial + closest_thickness_wm) / 2


# In[53]:


def mkdir(OUT_DIR):
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)


# In[54]:


def write_stats(stats, subject_id, fname, label_names):    
    with open(fname, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['SUBJECT'] + label_names)
        writer.writerow([subject_id] + list(stats))


# In[55]:


def create_annot(labels, lut, dst):
    mapping = {v: k for k, v in lut.Label.items()}
    labels = pd.Series(labels).map(mapping).values
    ctab = lut[["R", "G", "B", "T", "Label"]].values.astype(int)
    nib.freesurfer.io.write_annot(dst, labels, ctab, lut.Key.values, fill_ctab=True)


# ## MAIN

# In[56]:


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


# In[57]:


    src_dir = "/Users/timoblattner/Desktop/sub-POB-HC0001_ses-17473A_acq-t1-mpr-adni3-sag-iso-1mm-ns-TR-2.3-IR-0.9_run-1/"
    velocity_file = os.path.join(src_dir, "ForwardVelocityField.nii.gz")
    seg_file = os.path.join(src_dir, "seg.nii.gz")
    aparc_file = os.path.join(src_dir, "softmax_seg.nii.gz")
    subject_id = "sub-POB-HC0001_ses-17473A_acq-t1-mpr-adni3-sag-iso-1mm-ns-TR-2.3-IR-0.9_run-1"
    dst_dir = src_dir


    # In[58]:


    t1w = nib.load(os.path.join(src_dir, "T1w_norm_noskull_cropped.nii.gz")).get_fdata()


    # In[59]:


    lowmem = True # if lowmen -> likely we can use this by default
    precision = np.float32 if lowmem else np.float64 


    # In[60]:


    # load files
    ref_file = nib.load(seg_file)
    seg, affine = ref_file.get_fdata().astype(precision), ref_file.affine
    aparc = nib.load(aparc_file).get_fdata().astype(precision)
    velocity_field = nib.load(velocity_file).get_fdata().astype(precision)


    # In[61]:


    # use the probability map
    wm_prob = nib.load(os.path.join(src_dir, "wmprob.nii.gz")).get_fdata()
    gm_prob = nib.load(os.path.join(src_dir, "gmprob.nii.gz")).get_fdata()


    # In[62]:


    offset = 0 if np.max(aparc) <= 3000 else 10000
    lut, labels, all_labels = get_labels(offset)
    lut = {k: v for v, k in lut.items() if k >= 1000+offset and k <= 3000+offset}


    # In[63]:


    labl = np.unique(aparc).astype(int)
    lh_labels = labl[(labl >= 1000+offset) & (labl < 2000+offset)]
    rh_labels = labl[(labl >= 2000+offset) & (labl < 3000+offset)]


    # In[64]:


    # get wm surface mesh (old method)
    #white = clean_seg(np.clip((seg == 3) + np.isin(aparc, dien), 0, 1))
    #white_srf = gen_mesh(white)
    #white_srf = trimesh.smoothing.filter_humphrey(white_srf)


    # In[65]:


    wm_prob = fill_wm(wm_prob, aparc)
    white_srf = gen_mesh(wm_prob)
    white_srf = trimesh.smoothing.filter_humphrey(white_srf)


    # In[66]:


    #print(white_srf.area)
    #white_srf.show()


    # In[67]:


    # get pial mesh and thickness measurements
    vertices, thickness = apply_deformation(white_srf.vertices, velocity_field, step_size=0.1, order=3)
    pial_srf = trimesh.Trimesh(vertices=vertices, faces=white_srf.faces)
    pial_srf = trimesh.smoothing.filter_humphrey(pial_srf)


    # In[68]:


    #print(pial_srf.area)
    #pial_srf.show()


    # In[69]:


    # get metrics
    nearest_labels = label_points(white_srf.vertices, aparc)
    out_thickness = get_thickness_stats(thickness, nearest_labels)
    out_surface = get_surface_stats(pial_srf, nearest_labels)


    # In[70]:


    # save metrics
    write_stats(list(out_thickness.values()), subject_id, '{}/result-thick_direct.csv'.format(dst_dir), list(out_thickness.keys())) # thickness
    write_stats(list(out_surface.values()), subject_id, '{}/result-surf.csv'.format(dst_dir), list(out_surface.keys())) # surface


    # In[71]:


    # save surfaces
    mkdir(dst_dir + "/srf/")
    register(white_srf, ref_file, dst_dir + "/srf/" + '/white')
    register(pial_srf, ref_file, dst_dir  + "/srf/" + '/pial')


    # In[72]:


    print('mean thick: {:.3f} mm'.format(np.mean(list(out_thickness.values())[-2:])))
    print('total surface: {:.3f} mm^2'.format(np.sum(list(out_surface.values())[-2:])))


    # In[73]:


    # save annotation
    color_lut = pd.read_csv("fs_color.csv")
    create_annot(nearest_labels, color_lut, dst_dir + "/srf/aparc.annot")


    # In[74]:


    # save thciknnes mapp
    nib.freesurfer.io.write_morph_data(dst_dir + "/srf/thickness", thickness, fnum=0)


    # ## Alternate metrics

    # In[75]:





    # In[76]:


    closest_thickness = get_freesurfer_distance(white_srf, pial_srf)
    out_thickness_fs = get_thickness_stats(closest_thickness, nearest_labels)


    # In[77]:


    # save metrics
    write_stats(list(out_thickness_fs.values()), subject_id, '{}/result-thick_fs.csv'.format(dst_dir), list(out_thickness_fs.keys())) # thickness


    # In[78]:


    # save thciknnes mapp
    nib.freesurfer.io.write_morph_data(dst_dir + "/srf/thickness_fs", closest_thickness, fnum=0)


    # In[99]:


    #nib.save(nib.Nifti1Image(velocity_field[:,:,:,-1,:], affine), os.path.join(src_dir, 'velocity.nii.gz')) 


    # In[ ]:





    # In[ ]:




