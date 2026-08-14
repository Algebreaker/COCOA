# import packages
import os
from pathlib import Path
from subprocess import PIPE, run

import nibabel as nib
import numpy as np
import matplotlib as plt 
import pandas as pd
import json
from bids.layout import BIDSLayout, BIDSLayoutIndexer

from nilearn.glm.first_level import (FirstLevelModel,
                                     make_first_level_design_matrix)
from nilearn.image import binarize_img, load_img, math_img, mean_img
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.surface import load_surf_data, SurfaceImage
from nilearn.reporting import make_glm_report
from nilearn.interfaces.bids import save_glm_to_bids

# Input parameters: File paths
BIDS_DIR = Path('/data/pt_02825/MPCDF/BIDS')
DERIVATIVES_DIR = Path('/data/pt_02825/MPCDF/') 
# FMRIPREP_DIR = DERIVATIVES_DIR / 'fmriprep'
# PYBIDS_DIR = DERIVATIVES_DIR / 'pybids'
# UNIVARIATE_DIR = DERIVATIVES_DIR / 'univariate_surf'

# # Input parameters: Inclusion/exclusiong criteria
# FD_THRESHOLD = 0.7
# DF_QUERY = 'perc_outliers <= 0.30 & n_sessions >= 2' #to sort out which subjects to include 

# # Input parameters: First-level GLM
# TASK = 'Literacy'
# SPACE = ['MNI152NLin6Asym', 'fsaverage']
# BLOCKWISE = False
# SMOOTHING_FWHM = 5.0
# HRF_MODEL = 'glover + derivative + dispersion'
# SAVE_RESIDUALS = True


# # Define contrast specs in a clean, readable format
# CONTRASTS_SPEC = {'unimodal_audios': {'unimodal_audios': 1},
#     'unimodal_images': {'unimodal_images': 1},
#     'congruent': {'congruent': 1},
#     'incongruent': {'incongruent': 1},
    
#     'congruent-audio': {'congruent': 1, 'unimodal_audios': -1},
#     'congruent-visual': {'congruent': 1, 'unimodal_images': -1},
#     'congruent-incongruent': {'congruent': 1, 'incongruent': -1},
#     'incongruent-congruent': {'congruent': -1, 'incongruent': 1},
#     'congruent-audio-visual': {'congruent': 1, 'unimodal_audios': -0.5, 'unimodal_images': -0.5}
# }

# functions

# load event files of a given (sub,ses)
def load_events(layout, subject, session, task, bad_runs, trial=False):
    """Loads events files for a given subject, session, and task.
    Input:
        layout: BIDSlayout object
        subject: String; subject id
        session: String ;session id
        task: String; task id
        bad_runs : List; list of runs that its functional imgs cannot be loaded; only for surf images (run starts from 1)
        tria: Boolean; block-wise = False, trial-wise = True; default = False
    Return: 
        list; a list of data frames from events.tsv of each run"""
    
    if not trial:
        # a list of BIDSlayout oubjects
        events_files = layout.get(subject=subject, session=session, task=task, scope = 'raw',
                                  suffix='events', extension='tsv')
        
    else:
        # a list of Posixpaths
        # PLEASE change it to be relevant for your file directories
        # In our case, the event files containing stimuli details (e.g., congruent_B) 
        # are saved separately from event files without details (e.g., congruent).
        f_dir = Path('/data/pt_02825/Logfiles/Trial_Logs')   
        subject = subject.upper()  # its file names: AE_EAX3_001_*_run1.csv in our case
        session = '0' + session    # change the subject, session to be matched to the file name
        events_files = list(f_dir.glob(f'**/*{subject}_{session}*.csv'))
        events_files = sorted(events_files)
        
    print(f"Subject={subject}, Session={session}, Task={task}")
    print(f"Events files found: {events_files}")

    if not events_files:
        raise ValueError(f"No events files found for subject={subject}, session={session}, task={task}.")
    
    # Combine all event files into a single DataFrame
    events_dfs = []
    for events_file in events_files:
        if not trial:
            df = pd.read_csv(events_file, sep='\t')
            # Add metadata for which run this file corresponds to
            r  = events_file.entities.get("run", "unknown")
        else:
            df     = pd.read_csv(events_file, sep=',')
            f_name = str(events_file)
            r      = f_name[f_name.find('run')+3]
            
        if bad_runs:
            if r not in bad_runs:
                df["run"] = r
                events_dfs.append(df)
        else:        
            events_dfs.append(df)
        
    return events_dfs

# load confounds of a given (sub,ses)
def get_confounds(layout, subject, session, task, bad_runs, fd_threshold=0.5, out_threshold=0.3):
    """
    Loads and combines confounds for a given subject, session, and task of all runs.
    Return: list, list, list, int, int, list;
            a list of confounds, a list of percentage of non steady outlier,
            a list of percent of outlier from each run, min of acomcor len
            max num non_steady_state outlier, mas num of fd_outlier
            a list of valid runs passing perc outlier threshold (run starts from 0)
    """
    # Fetch all confounds files for the given subject, session, and task
    confounds_files = layout.get(subject=subject, session=session, task=task,
                                 desc='confounds', suffix='timeseries',
                                 extension='tsv')
    
    # proceed only loaded functional images
    if bad_runs:
        for f in confounds_files:
            r = f.entities.get('run')
            if r in bad_runs:
                confounds_files.remove(f)
    
    print(f"\n Confounds files found for subject={subject}, session={session}, task={task}: {len(confounds_files)} runs.")
    
    # prepare lists of outputs for all run for a (subject, session)
    confounds_lst       = []
    perc_non_steady_lst = []
    perc_outliers_lst   = []

    non_steady_lst = []
    outlier_lst    = []
    compcor_lst    = []
   
    # prepare a list of valid run indices - only runs that under perc_outliers <= 0.3 (check the DE_QUERY in run.py script)
    valid_runs      =[]
    valid_run_names =[]
    
    for r in range(len(confounds_files)):
        confounds = pd.read_csv(confounds_files[r], sep='\t')
        r_name    = confounds_files[r].entities.get('run')
        
        # movement
        hmp_cols     = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        
        # a comp cor 
        compcor_cols = [col for col in confounds if col.startswith('a_comp_cor_')]
        compcor_cols = compcor_cols[:6]
        compcor_lst.append(len(compcor_cols))
        
        # cosine dirft
        cosine_cols     = [col for col in confounds if col.startswith('cosine')]
        
        # non steady volume
        non_steady_cols = [col for col in confounds
                           if col.startswith('non_steady_state_outlier')]
        n_non_steady    = len(non_steady_cols)
        n_volumes       = len(confounds)
        perc_non_steady = n_non_steady / n_volumes
        
        perc_non_steady_lst.append(perc_non_steady)
        non_steady_lst.append(len(non_steady_cols))
        
        print(f'\n Found {n_non_steady} non-steady-state volumes ' +
              f'({perc_non_steady * 100:.1f}%) for subject {subject}, session {session}, run {r_name}"')
    
        # Add outlier regressors based on FD threshold
        fd = confounds['framewise_displacement']
        outlier_ixs   = np.where(fd > fd_threshold)[0]
        outliers      = np.zeros((len(fd), len(outlier_ixs)))
        outliers[outlier_ixs, np.arange(len(outlier_ixs))] = 1
        outlier_cols  = [f'fd_outlier{i}' for i in range(len(outlier_ixs))]
        outliers      = pd.DataFrame(outliers, columns=outlier_cols)
        confounds     = pd.concat([confounds, outliers], axis=1)
        n_outliers    = len(outlier_ixs)
        perc_outliers = n_outliers / n_volumes 
        
        # filter out valid runs based on fd threshold
        if perc_outliers <= out_threshold:
            valid_runs.append(r)
            valid_run_names.append(r_name)
        else:
            print(f"\n run {r_name} will be excluded due to high percentage of outliers")
        
        perc_outliers_lst.append(perc_outliers)
        outlier_lst.append(len(outlier_cols))
        
        print(f"Outlier columns added: {len(outlier_cols)}")
        print(f'   Found {n_outliers} outlier volumes ' +
              f'({perc_outliers * 100:.1f}%) for subject {subject}, session {session}, run {r_name}')
    
        # Collect all required columns
        cols_to_use = hmp_cols + compcor_cols + cosine_cols + non_steady_cols + outlier_cols
        missing_cols = [col for col in cols_to_use if col not in confounds.columns]
        if missing_cols:
            print(f"\n Warning: Missing columns in confounds file for run {r_name}: {missing_cols}")
        cols_to_use = [col for col in cols_to_use if col in confounds.columns]

        confounds_lst.append(confounds[cols_to_use])
    
    if len(valid_runs) > 0:
        compcor_m    = min([compcor_lst[i] for i in valid_runs])
        non_steady_M = max([non_steady_lst[i] for i in valid_runs ])
        outlier_M    = max([outlier_lst[i] for i in valid_runs])
    else:
        compcor_m    = None
        non_steady_M = None
        outlier_M    = None
        
    return confounds_lst, perc_non_steady_lst, perc_outliers_lst, compcor_m, non_steady_M, outlier_M, valid_runs, valid_run_names


# load a list of dim matched confound of a given (sub,ses)
def load_confound(confounds_lst, perc_non_steady, perc_outlier, compcor_m, non_steady_M, outlier_M, valid_runs):
    """" Regularize the size of confound matrix
        Retrun: list, float, float; list of confounds, mean percent of non_steady for the session, mean percent of outliers for the session
    """
    # Reshape confounds of valid runs so that all confounds from a session to have the same dimension.
    if len(valid_runs)>0:
        new_confounds = []
        for r in valid_runs:
            confounds      = confounds_lst[r]
            hmp_cols       = [col for col in confounds if col in ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
            non_steady_col = [col for col in confounds if col.startswith('non_steady')]
            fd_outlier_col = [col for col in confounds if col.startswith('fd_outlier')]
            compcor_col    = [col for col in confounds if col.startswith('a_comp_')]
            cosine_col     = [col for col in confounds if col.startswith('cosine')]
            print(f"\n trans and rot: {len(hmp_cols)},\n non_steady: {len(non_steady_col)}, \n outlier : {len(fd_outlier_col)}, \n acomp : {len(compcor_col)}, \n cosine : {len(cosine_col)}")
            
            # take the minimum length of acompcor to regularize the dim of design matrices of (sub,ses)
            if len(compcor_col) > compcor_m:
                keep_cols = compcor_col[:compcor_m]
                drop_cols = [col for col in compcor_col if col not in keep_cols]
                confounds = confounds.drop(columns=drop_cols)
            
            # add non_steady_state_outlier
            if len(non_steady_col) < non_steady_M:
                n_missing = non_steady_M - len(non_steady_col)
                t = len(confounds)
                padding_cols = [f'non_steady_state_outlier{i+len(non_steady_col)}' for i in range(n_missing)]
                padding_df = pd.DataFrame(np.zeros((t, n_missing)), columns=padding_cols)
                
                # Find the maximum index of the non_steady_state_outlier columns
                if any(col.startswith('non_steady') for col in confounds.columns): 
                    insert_at = confounds.columns.get_loc(non_steady_col[-1]) + 1
                    part1     = confounds.iloc[:, :insert_at]
                    part2     = confounds.iloc[:, insert_at:]
                    confounds = pd.concat([part1, padding_df, part2], axis=1)
            
                else: # no non_steady_state_outlier
                    insert_at = confounds.columns.get_loc(cosine_col[-1]) + 1
                    part1     = confounds.iloc[:, :insert_at]
                    part2     = confounds.iloc[:, insert_at:]
                    confounds = pd.concat([part1, padding_df, part2], axis=1)
                    
                all_ns_cols = [col for col in confounds.columns if col.startswith('non_steady_state_outlier')]
                new_names = [f'non_steady_state_outlier{i}' for i in range(non_steady_M)]
                confounds.rename(columns=dict(zip(all_ns_cols, new_names)), inplace=True)    
    
            # add fd_outlier
            if len(fd_outlier_col) < outlier_M:
                n_missing = outlier_M - len(fd_outlier_col)
                t = len(confounds)
                padding_cols = [f'fd_outlier{i+len(fd_outlier_col)}' for i in range(n_missing)]
                padding_df = pd.DataFrame(np.zeros((t, n_missing)), columns=padding_cols)
    
                insert_at = (confounds.columns.get_loc(fd_outlier_col[-1]) + 1
                             if fd_outlier_col else len(confounds.columns))
                confounds = pd.concat([confounds.iloc[:, :insert_at],
                                       padding_df,
                                       confounds.iloc[:, insert_at:]], axis=1)
                # Rename all fd_outlier columns to consistent names
                all_fd_cols = [col for col in confounds.columns if col.startswith('fd_outlier')]
                new_names = [f'fd_outlier{i}' for i in range(outlier_M)]
                confounds.rename(columns=dict(zip(all_fd_cols, new_names)), inplace=True)
                
            new_confounds.append(confounds)
            
        # mean prec_non_steady of a session
        
        vrun_non_steady   = [perc_non_steady[i] for i in valid_runs]
        vrun_outlier      = [perc_outlier[i] for i in valid_runs]
        m_perc_non_steady = np.mean(vrun_non_steady)
        m_perc_outlier    = np.mean(vrun_outlier)
    
    # When there is no valid run for a (sub, ses)
    else:
        new_confounds     = None
        m_perc_non_steady = None
        m_perc_outlier    = None
        
    return new_confounds, m_perc_non_steady, m_perc_outlier

# load a list of brain image mask files for a given (sub,ses)
def load_mask_img(layout, subject, session, task, space, bad_runs=False, valid_run=False):
    """Loads brain masks across runs.
    Input:
        layout: BIDS layout object
        subject: str; subject
        session: str; session
        task: str; task id
        space: str; space id
        bad_runs: list; list of bad runs - no func.nii or func.gii for the runs
        valid_runs: list; list of valid runs - valid runs after confounds based thresholidng 
        
    Return: 
        mask_imgs: list; a list of loaded mask images from each run
        mask_paths: list; a list of absolute location of mask files"""
    
    mask_files = layout.get(subject=subject, session=session, task=task,
                            space=space, desc='brain', suffix='mask',
                            extension='nii.gz')
    print(f"Subject={subject}, Session={session}, Task={task}, Space={space}")
    print(f"Mask files found: {mask_files}")
    
    if not mask_files:
        raise ValueError(f"No brain masks found for subject={subject}, session={session}, task={task}, space={space}.")

    # process only non bad functional runs
    if bad_runs:
        new_mask_files = []
        for f in mask_files:
            r = f.entities.get('run')
            if r not in bad_runs:
                new_mask_files.append(f)
        mask_files = new_mask_files
                
    # process only valid runs after confound based thesholding            
    if valid_run:
        new_mask_files = []
        for f in mask_files:
            r = f.entities.get('run')
            if r in valid_run:
                new_mask_files.append(f)
        mask_files = new_mask_files
            
    # Load images and extract absolute paths    
    mask_imgs  = [load_img(f.path) for f in mask_files]
    mask_paths = [os.path.abspath(f.path) for f in mask_files]
    
    return mask_imgs, mask_paths
    
# load a list of functional img fiels for a given (sub,ses)
def load_func_img(layout, subject, session, task, space):
    """Loads functional images across runs.
    Return: list; a list functional images from each run"""
    
    func_files = layout.get(subject=subject, session=session, task=task,
                            space=space, suffix='bold', extension='nii.gz')
    print(f"Subject={subject}, Session={session}, Task={task}, Space={space}")
    print(f"func files found: {func_files}")
    
    if len(func_files)==0:
        raise ValueError(f"No func img found for subject={subject}, session={session}, task={task}, space={space}.")
    
    bad_runs  = []
    func_imgs = []
    
    for i,f in enumerate(func_files):
        try:
            func_img = load_img(f.path)
            func_imgs.append(func_img)
        except:
            print(f'sub {subject} ses {session} run {i+1} func img cannot be loaded.')
            bad_runs.append(i+1)
    
    if len(bad_runs) == 0:
        bad_runs = False
    
    return func_imgs, bad_runs


# load a list of functional surface img fiels for a given (sub,ses)
def load_func_surf_img(layout, subject, session, task, space, hemis = ['L','R'], smooth = False):
    """Loads surface functional images of both hemis across runs.
    Input:
        layout : BIDSlayout object
        subject: String
        session: String
        task   : String
        space  : String
        hemis  : a List; default ['L','R']
        smooth : boolean; True= smoothed func image, False = unsmoothed func image
    Return: 
        list of numpy arrays; a list functional surf images of both hemis from existing runs"""
                
    hemi_func_data = {}

    for hemi in hemis:
        if hemi == 'L':
            h_hemi = 'left'
        elif hemi == 'R':
            h_hemi = 'right'
        else:
            print(f'{hemi} is neither L or R')
        
        if not smooth:
            surf_func_files = layout.get(
                subject=subject, session=session, task=task, space=space,
                suffix='bold', extension='func.gii', hemi=hemi
            )
        else:
            surf_func_files = layout.get(
                subject=subject, session=session, task=task, space=space,
                suffix='smoothed', extension='func.gii', hemi=hemi
            )
        
        
        print(f"\n Subject={subject}, Session={session}, Task={task}, Space={space}, Hemi={hemi}")
        print(f" \n Func files found: {surf_func_files}")
         
        hemi_func_data[f'{h_hemi}'] = surf_func_files
        
        if not surf_func_files:
            raise ValueError(f"\nNo func img found for subject={subject}, session={session}, task={task}, space={space}, hemi={hemi}.")
    
    # load mesh file (fsaverage)    
    mesh      = load_fsaverage(mesh='fsaverage')
    mesh_infl = mesh['inflated']
    
    # a list of functioanl surface imgs mapped on the inflated fsaverage surface (both hemis)
    func_surf_imgs = []
    bad_runs       = []
    for i in range(len(hemi_func_data['left'])):
        try:
            func_data = {'left':hemi_func_data['left'][i], 'right':hemi_func_data['right'][i] }
            func_surf_imgs.append(SurfaceImage(data=func_data, mesh=mesh_infl))
        except:
            print(f'sub {subject} ses {session} run {i+1} surf func img cannot be loaded.')
            bad_runs.append(i+1)
    
    if len(bad_runs)==0:
        bad_runs = False
        
    if bad_runs:
        print(f'\n func img of run {bad_runs} cannot be loaded.')
        print('\n    Exclude these runs for further analysis.')
        
    return func_surf_imgs, bad_runs

# Sort subjects and sessions
def get_subjects_sessions(layout, task, space, subject=None):
    """Gets a list of all subject-session pairs with preprocessed data.
        
    Input
        layout: BIDSlayout
        task: string; task ID
        space: string; space ID
        subject: a list; list of subjects to process (default: None)
    Return
        a list of tuples (sub,ses)
        """
        
    # get all relevant subjects in bids directory    
    if not subject:
        subjects = layout.get_subjects(task=task, space=space)
        
        
    # get the subjects in the given list     
    else:
        subjects = subject.split(',')
        
    all_sessions = [layout.get_sessions(subject=subject, task=task, 
                                        space=space) for subject in subjects]

    subjects_sessions = [(subject, session) 
                         for subject, sessions in zip(subjects, all_sessions) 
                         for session in sessions]

    return sorted(subjects_sessions)

# Build contrast array 
def make_contrast_vector(design_matrix_columns, condition_weights):
    """Build a contrast vector relevant to design matrix
       Return: np array; contrast values assigned array"""
       
    contrast = np.zeros(len(design_matrix_columns))
    
    for condition, weight in condition_weights.items():
        if condition not in design_matrix_columns:
            raise ValueError(f"'{condition}' not found in design matrix columns.")
            
        idx = design_matrix_columns.index(condition)
        contrast[idx] = weight
    return contrast


# Run first level glm within a session (3 or 4 runs)
def run_glm(layout, bids_dir, fmriprep_dir, pybids_dir, task, space,
            fd_threshold, hrf_model, smoothing_fwhm, output_dir,
            save_residuals, subject, session, contrast_spec):
    """Runs a first-level GLM for a given subject and session.
        Return: list, list, list, float, float, list;
                list of FirstLevelModel beta img (nilearn), list of mask imgs, list of mask imgs paths,
                mean perc_non_steady, mean perc_fd_outlier, list of residual files """
    
    # load inputs for nilearn.FirstLevelModel for multiple runs
    func_imgs, bad_runs   = load_func_img(layout, subject, session, task, space)
    events                = load_events(layout, subject, session, task, bad_runs)
    nr_confounds, perc_non_steady, perc_outliers, com_m, ns_M, o_M, valid_runs, valid_run_names = \
        get_confounds(layout, subject, session, task, bad_runs, fd_threshold)
    confounds, perc_non_steady, perc_outliers             = \
        load_confound(nr_confounds, perc_non_steady, perc_outliers, com_m, ns_M, o_M, valid_runs)
    mask_imgs, mask_paths = load_mask_img(layout, subject, session, task, space, bad_runs=bad_runs, valid_run=valid_run_names)

    # get tuple of (sub,ses,valid_runs)
    ssvr_tuple = (subject, session, len(valid_runs))
    # FirstLevelModel
    print(f"\n    Running GLM for subject {subject}, session {session}, valid runs: {len(valid_runs)}")
    
    # control the number of valid runs of a session (i.e. the number of runs which its the percentage of outliers <= 0.3 is)
    if len(valid_runs) <2:
        print(f"\n sub-{subject} ses-{session} has less than 1 run, exclude this session.")
        effect          = None
        perc_non_steady = None
        perc_outliers   = None
        mask_imgs       = None
        mask_paths      = None
        residuals       = None


        return effect, mask_imgs, mask_paths, perc_outliers, perc_non_steady, residuals, ssvr_tuple, bad_runs
    
    else:
        # FirstLevelModel
        print(f"Running GLM for subject {subject}, session {session}...")
    
        # Transform scans to time (Start from 1 sec, TR duration)    
        n_scans     = func_imgs[0].shape[-1]
        tr          = layout.get_tr()
        frame_times = tr * (np.arange(n_scans) + 0.5) 
       
        print(f"\n    Shape of frame_times: {frame_times.shape}")
        
        # create design matrices from event files for each run
        design_matrix = []
        for i,r in enumerate(valid_runs):
            d_matrix = make_first_level_design_matrix(frame_times,
                                                      events=events[r],
                                                      hrf_model=hrf_model,
                                                      drift_model=None,
                                                      high_pass=None,
                                                      add_regs=confounds[i],)
            design_matrix.append(d_matrix)
            
            
        # FirstLevelModel (signals mean to zero along time axis)
        glm = FirstLevelModel(smoothing_fwhm=smoothing_fwhm,
                                    mask_img=mask_imgs[0], minimize_memory=False)
    
            
        # Fit the model
        func_imgs = [func_imgs[r] for r in valid_runs]

            
        glm_result = glm.fit(run_imgs=func_imgs, design_matrices=design_matrix)
            
        # Save residual files
        func_dir = output_dir / f'sub-{subject}' / f'ses-{session}' / 'func'
        func_dir.mkdir(parents=True, exist_ok=True)
    
        residuals = glm.residuals
        residuals_files = []
        for i,r in enumerate(valid_run_names):
            residuals_filename = f'sub-{subject}_ses-{session}_task-{task}_run-{r}_space-{space}_desc-residuals.nii.gz'
            residuals_file = func_dir / residuals_filename
            residuals[i].to_filename(residuals_file)
            residuals_files.append(residuals_file)
            
        # Compute contrast, save effect, effect_variance and z score for second level analysis
        all_effect_size = []
        for index, (contrast_id, condition_weights) in enumerate(contrast_spec.items()):
            print(f"  Contrast {index + 1:02d} of {len(contrast_spec)}: {contrast_id}")
            
            # Create contrast vector assuming that all design matrices having the same dimension
            contrast_vector = make_contrast_vector(list(design_matrix[0].columns), condition_weights)
        
            # Compute contrasts
            effect_size = glm.compute_contrast(contrast_vector, output_type="effect_size")
        
            all_effect_size.append(effect_size)
        
            # Save outputs
            # ef_image_path = func_dir / f"sub-{subject}_ses-{session}_task-Literacy_FirstLevel_{contrast_id}_effect_size_map.nii.gz"
            # z_image_path = func_dir / f"sub-{subject}_ses-{session}_task-Literacy_FirstLevel_{contrast_id}_z_map.nii.gz"
            # effect_size.to_filename(ef_image_path)
            # z_score.to_filename(z_image_path)
    
                    
        # Save report of multiple run firstlevel model (i.e. fixed effects from each session across valid runs)

        design_columns = list(design_matrix[0].columns)
        contrast_vectors = {
            name: make_contrast_vector(design_columns, weights)
            for name, weights in contrast_spec.items()
        }
    
        save_glm_to_bids(glm, contrasts=contrast_vectors,
                out_dir= output_dir / "derivatives" ,
                prefix=f"sub-{subject}_ses-{session}_task-Literacy_FirstLevel",)
            

        return all_effect_size, mask_imgs, mask_paths, perc_outliers, perc_non_steady, residuals_files, ssvr_tuple, bad_runs

# Run first level glm within a session having more than 1 run
def run_glm_surf(layout, bids_dir, fmriprep_dir, pybids_dir, task, space,
                 fd_threshold, hrf_model, smoothing_fwhm, output_dir,
                 save_residuals, subject, session, hemis, trial, run_wise, smooth, contrast_spec):
    """Runs a first-level GLM for a given subject and session.
        Input:
            -layout: BIDSlayout object
            -bids_dir: PosixPath; 
                directory where original data including events.tsv are saved
            -fmriprep_dir: PosixPath; 
                directory where preprocessed images are saved
            -pybids_dir: PosixPath; 
                directory to save layout based data structure
            -task: str;
            -space: str; 
            -fd_threshold: str; 
            -hrf_model: str; 
                check the list of supported hrf model by nilearn
            -smoothing_fwhm: float; 
                no supported for surface based analysis
            -output_dir: PosixPath; 
                directory to save output 
            -save_residul: boolean; 
                True= save redisuals
            -subject: str;
            -session: str;
            -hemis: list; 
                ['L','R'] or ['L'] or ['R']
            -trial: boolean; 
                True = contrast*alphabet wise design matrix, False= contrast wise design matrix
            -run_wise: boolean; 
                True = run level glm, False = session level glm
            -smooth: boolean; 
                True = use smoothed func img (default), False = use non-smoothed func img
            -contrast_spec: dict; 
                {contrast_name: {contrast_item: item weight}}
            
            
        Return: 
            - all_effect_size: nilearn.object; first level glm object
            - perc_non_steady: float; mean percent of non steady volume across vaild runs of the (Sub,Ses)
            - perc_outliers:   float; mean percent of outliers across valid runs of the (sub,ses)
            - ssvr_tuple:      tuple; tuple of (sub,ses,the number of valid runs) 
            - bad_runs:        list; a list of runs having func imgs that cannot be loaded 
            """
    
    # load inputs for nilearn.FirstLevelModel for multiple runs
    
    # load functional imgs 
    func_imgs, bad_runs = load_func_surf_img(layout, subject, session, task, space, hemis, smooth)
    
    # proceed only runs that passed from load_func_surf_img 
    events    = load_events(layout, subject, session, task, bad_runs, trial) 
    nr_confounds, perc_non_steady, perc_outliers, com_m, ns_M, o_M, valid_runs, valid_run_names = \
        get_confounds(layout, subject, session, task, bad_runs, fd_threshold)
    confounds, perc_non_steady, perc_outliers             = \
        load_confound(nr_confounds, perc_non_steady, perc_outliers, com_m, ns_M, o_M, valid_runs)
    

    # get tuple of (sub,ses,valid_runs)
    ssvr_tuple = (subject, session, len(valid_runs))
    # FirstLevelModel
    print(f"\n    Running GLM for subject {subject}, session {session}, valid runs: {len(valid_runs)}")
    
    # control the number of valid runs of a session (i.e. the number of runs which its the percentage of outliers <= 0.3 is)
    if len(valid_runs) <2:
        print(f"\n sub-{subject} ses-{session} has less than 1 run, exclude this session.")
        perc_non_steady = None
        perc_outliers   = None    

        return perc_non_steady, perc_outliers, ssvr_tuple, bad_runs
    
    # only the session having more than two valid runs are included for the further analysis
    else:
        # Transform scans to time (Start from 1 sec, TR duration)    
        n_scans     = func_imgs[0].shape[-1]
        tr          = layout.get_tr()
        frame_times = tr * (np.arange(n_scans) + 0.5) 
       
        print(f"\n    Shape of frame_times: {frame_times.shape}")
        
        # create design matrices from event files for each run
        design_matrix = []
        for i,r in enumerate(valid_runs):
            d_matrix = make_first_level_design_matrix(frame_times,
                                                      events=events[r],      # events list items are pre-valid-test runs
                                                      hrf_model=hrf_model,   # i.e., events of all loaded functional image 
                                                      drift_model=None,
                                                      high_pass=None,
                                                      add_regs=confounds[i]) # confounds list items are post-valid-test runs
            design_matrix.append(d_matrix)                                   # i.e., confounds passing valid test th among loaded func
                 
        # FirstLevelModel (signals mean to zero along time axis)
        glm        = FirstLevelModel(minimize_memory=False)
        funcs      = [func_imgs[r] for r in valid_runs]
        
        # load fsaverage data
        fsaverage_data = load_fsaverage_data(mesh="fsaverage", mesh_type="inflated", data_type="curvature")
        
        if not run_wise:
            
            # SESSION LEVEL GLM
            glm_result = glm.fit(run_imgs=funcs, design_matrices=design_matrix)
                    
            # Save report of multiple run firstlevel model (i.e. fixed effects from each session across valid runs)
            design_columns = list(design_matrix[0].columns)
            contrast_vectors = {
                name: make_contrast_vector(design_columns, weights)
                for name, weights in contrast_spec.items()
            }
            
            # load fsaverage data
            fsaverage_data = load_fsaverage_data(mesh="fsaverage", mesh_type="inflated", data_type="curvature")
            
            # save glm results
            save_glm_to_bids(glm, contrasts=contrast_vectors,
                    out_dir= output_dir ,
                    prefix =f"sub-{subject}_ses-{session}_task-Literacy_space-fsaverage_FirstLevel",
                    bg_img = fsaverage_data,
                    height_control = None,
                    two_sided = True,
                    )
        
        else:
            # RUN LEVEL GLM
            for r in range(len(funcs)):
                glm_result = glm.fit(run_imgs=funcs[r], design_matrices=design_matrix[r])
        
                # Compute contrast, save effect, effect_variance and z score for second level analysis
                #for index, (contrast_id, condition_weights) in enumerate(contrast_spec.items()):
                    #print(f"  Contrast {index + 1:02d} of {len(contrast_spec)}: {contrast_id}")
                    
                    # Create contrast vector assuming that all design matrices having the same dimension
                    #contrast_vector = make_contrast_vector(list(design_matrix[r].columns), condition_weights)
                
                    # Compute contrasts
                    #effect_size = glm.compute_contrast(contrast_vector, output_type="effect_size")
                        
                # Save report of multiple run firstlevel model (i.e. fixed effects from each session across valid runs)
                
                design_columns = list(design_matrix[r].columns)
                contrast_vectors = {
                    name: make_contrast_vector(design_columns, weights)
                    for name, weights in contrast_spec.items()
                }
                
                # save glm results
                save_glm_to_bids(glm, contrasts=contrast_vectors,
                        out_dir= output_dir ,
                        prefix =f"sub-{subject}_ses-{session}_run-{r}_task-Literacy_space-fsaverage_FirstLevel",
                        bg_img = fsaverage_data,
                        height_control = None,
                        two_sided = True,
                        )
                            
        return perc_non_steady, perc_outliers, ssvr_tuple, bad_runs

# Get lists of percs_non_steady, percs_outliers, and sub-ses-valid-runs without glm
def run_confounds(layout, task, space, fd_threshold, subject, session):
    """Runs a first-level GLM for a given subject and session.
        Return: """
    
    func_imgs, bad_runs = load_func_surf_img(layout, subject, session, task, space)
    
    # proceed only runs that passed from load_func_surf_img 
    nr_confounds, perc_non_steady, perc_outliers, com_m, ns_M, o_M, valid_runs, valid_run_names = \
        get_confounds(layout, subject, session, task, bad_runs, fd_threshold)
    confounds, perc_non_steady, perc_outliers             = \
        load_confound(nr_confounds, perc_non_steady, perc_outliers, com_m, ns_M, o_M, valid_runs)
    

    # get tuple of (sub,ses,valid_runs)
    ssvr_tuple = (subject, session, len(valid_runs))
    # FirstLevelModel
    print(f"\n    Running GLM for subject {subject}, session {session}, valid runs: {len(valid_runs)}")
    
    # control the number of valid runs of a session (i.e. the number of runs which its the percentage of outliers <= 0.3 is)
    if len(valid_runs) <2:
        print(f"\n sub-{subject} ses-{session} has less than 1 run, exclude this session.")
        perc_non_steady = None
        perc_outliers   = None    

        return perc_non_steady, perc_outliers, ssvr_tuple, bad_runs
    # only the session having more than two valid runs are not None
    else:
        return perc_non_steady, perc_outliers, ssvr_tuple, bad_runs


# load meta data frame
def load_meta_df(layout, task, percs_outliers, percs_non_steady,
                 ssvr_tuple, df_query, out_dir, idx=None):

    basic_df = layout.get_collections(
        task=task,
        level='session',
        types='scans',
        merge=True
    ).to_df()

    basic_df = basic_df.sort_values(['subject', 'session', 'acq_time']).reset_index(drop=True)

    df = (
        basic_df.groupby(['subject', 'session'], as_index=False)
        .agg({'acq_time': 'min'})
    )

    valid_sessions = set([(sub, ses) for (sub, ses, _) in ssvr_tuple])
    print("len(ssvr_tuple):", len(ssvr_tuple))
    print("unique (sub,ses):",
      len(set((s, se) for s, se, _ in ssvr_tuple)))
    print("len(percs_non_steady):", len(percs_non_steady))
    print("len(percs_outliers):", len(percs_outliers))
    
    df = df[df[['subject', 'session']].apply(tuple, axis=1).isin(valid_sessions)].copy()

    if idx:
        df = df[df[['subject', 'session']].apply(tuple, axis=1).isin(idx)].copy()

    print(f"Length basic_df: {len(basic_df)}")
    print(f"Length filtered df: {len(df)}")
    df_pairs = set(zip(df['subject'], df['session']))
    ssvr_pairs = set((s, se) for s, se, _ in ssvr_tuple)

    print("Missing from df:")
    print(sorted(ssvr_pairs - df_pairs))


    df['total_sessions'] = df.groupby('subject')['session'].transform('count')
    df['valid_runs'] = 0

    lookup = {(sub, ses): vr for (sub, ses, vr) in ssvr_tuple}

    df['valid_runs'] = df.apply(
        lambda r: lookup.get((r['subject'], r['session']), np.nan),
        axis=1
    )

    metric_lookup = {
    (sub, ses): (pns, po)
    for (sub, ses, _), pns, po
    in zip(ssvr_tuple, percs_non_steady, percs_outliers)
        }

    df['perc_non_steady'] = df.apply(
        lambda r: metric_lookup[(r['subject'], r['session'])][0],
        axis=1
        )

    df['perc_outliers'] = df.apply(
        lambda r: metric_lookup[(r['subject'], r['session'])][1],
        axis=1
        )
    #df['perc_non_steady'] = percs_non_steady
    #df['perc_outliers'] = percs_outliers

    df['acq_time'] = pd.to_datetime(df['acq_time'])

    df['time_diff'] = df['acq_time'] - df['acq_time'].min()
    df['time'] = df['time_diff'].dt.days / 30.437

    good_df = df.query('valid_runs >= 2').copy()
    good_df['valid_ses'] = good_df.groupby('subject')['session'].transform('count')
    good_df = good_df.query(df_query).copy()
    good_df = good_df.set_index(['subject', 'session'])

    good_ixs = good_df.index.tolist()

    df['good_ixs'] = ['N'] * len(df)
    for (sub, ses) in good_ixs:
        df.loc[(df['subject'] == sub) & (df['session'] == ses), 'good_ixs'] = 'Y'

    metadata_file = out_dir / f'metadata_task-{task}.tsv'
    df.to_csv(metadata_file, sep='\t', index=False)

    print(f"Metadata saved to {metadata_file}")

    return df, good_ixs
#def load_meta_df(layout, task, percs_outliers, percs_non_steady, ssvr_tuple, df_query, out_dir, idx = None):
#    """Load the DataFrame with the subject/session metadata for the mixed model.
#    Input    
#    layout: BIDSlayout
#    task: string; task ID
#    percs_outliers: a list; a list of percent of outliers for all sub, ses
#    percs_non_steady: a list; a list of percent of non-steady state scans for all sub, ses
#    ssvr_tuple : a tuple; a tuple (sub, ses, the number of valid run)
#    valid_r_names: a list of integers; e.g. [1,2,3] valid run names
#    df_query: string; query to filter out (e.g. 'valid_ses >= 2')
#    out_dir :  Path object; output directory
#    idx: a list; None or a list of (subjects, session) to process
    
#    Output
#    df: panda DataFrame; meta data 
#    good_ixs: a list; indices of df passing sub,ses selection criteria"""
    
    # Fetch run-level collections and convert to DataFrame
#    basic_df = layout.get_collections(task=task, level='session', types='scans', merge=True).to_df() 
#    basic_df = basic_df.sort_values(['subject', 'session', 'acq_time']).reset_index(drop=True)
        
    # Aggregate to session-level by selecting the earliest acquisition time per session
#    df = (
#        basic_df.groupby(['subject', 'session'], as_index=False)
#        .agg({'acq_time': 'min'})  # Retain only the earliest acquisition time
#    )

#    if idx:
#        df = df[df[['subject', 'session']].apply(tuple, axis=1).isin(idx)]
   
    # Debugging
#    print(f"Length of basic_df: {len(basic_df)}")
#    print(f"Length of df after aggregation: {len(df)}")
    

    # Add metadata columns
#    df['total_sessions']  = df.groupby('subject')['session'].transform('count')
#    df['valid_runs']      = np.zeros(len(df))
    
    # add the number of valid runs
#    for (sub,ses,vruns) in ssvr_tuple:
#        df.loc[(df['subject'] == sub) & (df['session'] == ses), 'valid_runs' ] = vruns
        
#    df['perc_non_steady'] = percs_non_steady
#    df['perc_outliers']   = percs_outliers       # alrady excluded the runs having over 30 percent outliers 
#    df['acq_time']        = pd.to_datetime(df['acq_time'])
#    print(f"Length of percs_non_steady: {len(percs_non_steady)}")
#    print(f"Length of percs_outliers: {len(percs_outliers)}")
#
    # Add time-related columns
#    df['time_diff'] = df['acq_time'] - df['acq_time'].min()
#    df['time']      = df['time_diff'].dt.days / 30.437  # Convert to months
    #df['time2']    = df['time'] ** 2

    # Filter by query
#    good_df                    = df.query('valid_runs >= 2').copy()        # session having more than 2 runs
#    good_df.loc[:,'valid_ses'] = good_df.groupby('subject')['session'].transform('count')
#    good_df                    = good_df.query(df_query).copy()            # 'valid_ses >= 2'
#    good_df                    = good_df.set_index(['subject', 'session']) # set index as (sub,ses)
    
    # get a list of good (sub,ses)
#    good_ixs = good_df.index.tolist() 
    
    # add a colume marking good index
#    df['good_ixs'] = ['N']*(len(df))
#    for (sub, ses) in good_ixs:
#        df.loc[(df['subject'] ==sub) &(df['session']==ses),'good_ixs'] = 'Y'
                      
    # save the meta data
#    metadata_file = out_dir / f'metadata_task-{task}.tsv'
#    df.to_csv(metadata_file, sep='\t', index=False, float_format='%.5f')
#    print(f"Metadata saved to {metadata_file}")

#    return df, good_ixs

def combine_save_mask_imgs(mask_imgs, output_dir, task, space,
                           perc_threshold=0.9):
    """
    input: ...
    output: binarized image data, nii file object, nii file name
    
    Combines brain masks across subjects and sessions and saves the result.

    Only voxels that are present in at least `perc_threshold` of all masks are
    included in the final mask."""

    mask_img = combine_mask_imgs(mask_imgs, perc_threshold)

    mask_file, mask_name = save_img(mask_img, output_dir, task, space,
                         desc='brain', suffix='mask')

    return mask_img, mask_file, mask_name


def combine_mask_imgs(mask_imgs, perc_threshold=0.9):
    """
    Input: nii image or a list of images
           If a list of 4D images is given, the mean of each 4D image is 
           computed separately, and the resulting mean is computed after.
    Combines brain masks across subjects and sessions.

    Only voxels that are present in at least `perc_threshold` of all masks are
    included in the final mask."""

    return binarize_img(mean_img(mask_imgs), threshold=perc_threshold)


def save_img(img, output_dir, task, space, desc, suffix,
             subject=None, session=None):
    """Saves a NIfTI image to a file in the output directory."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.nii.gz'

    if session:
        filename = f'ses-{session}_{filename}'

    if subject:
        filename = f'sub-{subject}_{filename}'

    file = output_dir / filename
    img.to_filename(file)

    return file, filename

def load_and_align_masks(mask_paths, reference_img):
    aligned_data = []
    for path in mask_paths:
        img = nib.load(path)
        if not np.allclose(img.affine, reference_img.affine) or img.shape != reference_img.shape:
            raise ValueError(f"Image {path} does not match reference affine/shape.")
        data = img.get_fdata()
        aligned_data.append(data)
    return np.stack(aligned_data, axis=0)

def make_group_mask(mask_paths, output_path, threshold_ratio=0.5):
    print(f"Loading reference image: {mask_paths[0]}")
    ref_img = nib.load(mask_paths[0])
    
    print("Loading and aligning all mask images...")
    mask_stack = load_and_align_masks(mask_paths, ref_img)

    print("Averaging masks voxelwise...")
    mean_mask = np.mean(mask_stack, axis=0)

    print(f"Thresholding mean mask at ratio {threshold_ratio}...")
    binary_mask = (mean_mask >= threshold_ratio).astype(np.uint8)

    print(f"Saving final mask to {output_path}")
    final_img = nib.Nifti1Image(binary_mask, ref_img.affine, ref_img.header)
    nib.save(final_img, output_path)
