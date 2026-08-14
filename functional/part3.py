import os

from pathlib import Path
from subprocess import PIPE, run
import nibabel as nib
import numpy as np

from bids.layout import BIDSLayout, BIDSLayoutIndexer
from statsmodels.stats.multitest import fdrcorrection
from nilearn.image import binarize_img, load_img, math_img, mean_img
from nilearn.reporting import get_clusters_table
from nibabel import Nifti1Image
from nibabel.freesurfer.io import read_label

from scipy.stats import norm
from juliacall import Main as jl

#jl.seval("import Pkg; Pkg.add(\"MixedModels\")")

# Input parameters: File paths
BIDS_DIR = Path('/data/pt_02825/MPCDF/BIDS')
DERIVATIVES_DIR = Path('/data/pt_02825/MPCDF/') 
FMRIPREP_DIR = DERIVATIVES_DIR / 'fmriprep'
FREESURFER_DIR = DERIVATIVES_DIR/ 'freesurfer'
PYBIDS_DIR = DERIVATIVES_DIR / 'pybids'
UNIVARIATE_DIR = DERIVATIVES_DIR / 'univariate'

# Input parameters: Inclusion/exclusiong criteria
FD_THRESHOLD = 0.7
DF_QUERY = 'perc_outliers <= 0.25 & n_sessions >= 2' #to sort out which subjects to include 

# Input parameters: First-level GLM
TASK = 'Literacy'
SPACE = 'MNI152NLin6Asym'
RUN = 1
BLOCKWISE = False
SMOOTHING_FWHM = 5.0
HRF_MODEL = 'glover + derivative + dispersion'
SAVE_RESIDUALS = True


N_JOBS = 8

# Input parameters: Cluster correction
CLUSTSIM_SIDEDNESS = '2-sided'
CLUSTSIM_NN = 'NN1'  # Must be 'NN1' for Nilearn's `get_clusters_table`
CLUSTSIM_VOXEL_THRESHOLD = 0.001
CLUSTSIM_CLUSTER_THRESHOLD = 0.05
CLUSTSIM_ITER = 10000

# Input parameters: Group-level linear mixed models
FORMULA = 'beta ~ time + (time | subject)'


# Function: change the contrast dict key to fit the name of saved results
def transform_key(key: str) -> str:
    """
    input:
        key: string; string 
    return:
        a string; '-' -> 'Minus', '_' -> omitted, and the latter after them
                          get capitalized 
                          (e.g. congruent-incongruent -> congruentMinusIncongruent)
    """
    result = []
    i = 0
    while i < len(key):
        if key[i] == '-':
            result.append('Minus')
            i += 1
            if i < len(key):
                result.append(key[i].upper())
        elif key[i] == '_':
            i += 1
            if i < len(key):
                result.append(key[i].upper())
        else:
            result.append(key[i])
        i += 1
    return ''.join(result)


def compute_beta_img(glm, conditions_plus, conditions_minus):
    """
    output: nii image (z score)
    Computes a beta image from a fitted GLM for a given contrast."""

    design_matrices = glm.design_matrices_
    assert len(design_matrices) == 1
    design_matrix = design_matrices[0]

    contrast_values = np.zeros(design_matrix.shape[1])
    for col_ix, column in enumerate(design_matrix.columns):
        if column in conditions_plus:
            contrast_values[col_ix] = 1.0 / len(conditions_plus)
        if column in conditions_minus:
            contrast_values[col_ix] = -1.0 / len(conditions_minus)
    
    try:
        glm_efct_map = glm.compute_contrast(contrast_values, output_type='z_score')

        # Ensure the output is a valid Nifti1Image
        if not isinstance(glm_efct_map, Nifti1Image):
            raise ValueError("Unexpected return type from compute_contrast.")

        return glm_efct_map

    except Exception as e:
        print(f"Error happens in 1st level effect size: {e}")
        return None

def save_beta_img(beta_img, output_dir, subject, session, task, space,
                  contrast_label):
    """Saves a beta image to a NIfTI file in the output directory."""

    sub = f'sub-{subject}'
    ses = f'ses-{session}'
    tas = f'task-{task}'
    spc = f'space-{space}'
    des = f'desc-{contrast_label}'

    beta_dir = output_dir / sub / ses / 'func'
    beta_dir.mkdir(parents=True, exist_ok=True)
    beta_filename = f'{sub}_{ses}_{tas}_{spc}_{des}_z_score.nii.gz'
    beta_file = beta_dir / beta_filename
    beta_img.to_filename(beta_file)

# def fit_mixed_models(formula, dfs):
#     """Fits mixed models for a list of DataFrames using the `MixedModels`
#     package in Julia."""

#     model_cmd = f"""
#         using MixedModels
#         using Suppressor

#         function fit_mixed_model(df)
#           fml = @formula({formula})
#           mod = @suppress fit(MixedModel, fml, df)
#           bs = mod.beta
#           zs = mod.beta ./ mod.stderror
#         return bs, zs
#         end
        
#         function fit_mixed_models(dfs)
#           map(fit_mixed_model, dfs)
#         end"""
#     fit_mixed_models_julia = jl.seval(model_cmd)

#     return fit_mixed_models_julia(dfs)

def fit_mixed_models(formula, dfs):
    """Fits mixed models for a list of DataFrames using the `MixedModels`
       package in Julia. Using thread to prarellize voxel wise calculation """
       
    model_cmd = f"""
        using MixedModels
        using Suppressor
        using Base.Threads

        function fit_mixed_model(df)
            fml = @formula({formula})
            mod = @suppress fit(MixedModel, fml, df)
            bs = mod.beta
            zs = mod.beta ./ mod.stderror
        return bs, zs
        end

        function fit_mixed_models(dfs)
            results = Vector{{Tuple{{Vector{{Float64}}, Vector{{Float64}}}}}}(undef, length(dfs))
            @threads for i in 1:length(dfs)
                try
                    results[i] = fit_mixed_model(dfs[i])
                catch e
                    @warn "Model failed for index $i with error: $e"
                    results[i] = (fill(NaN, 2), fill(NaN, 2))  # fallback if model fails
                end
            end
            return results
        end
        """
    fit_mixed_models_julia = jl.seval(model_cmd)
    return fit_mixed_models_julia(dfs)


def save_array_to_nifti(array, ref_img, voxel_ixs, output_dir, task, space,
                        desc, suffix, subject=None, session=None):
    """Inserts a NumPy array into a NIfTI image and saves it to a file.
        Input:
            array: numpy array; 
            ref_img: path of reference image or Nifti imge;
    """
    
    if type(ref_img) == str:
        ref = nib.load(ref_img)
        aff = ref.affine
        full_array = np.zeros(ref.shape)
        full_array[tuple(voxel_ixs.T)] = array
        img = nib.Nifti1Image(full_array, aff)
    else:
        img = nib.Nifti1Image(array, ref_img.affine)

    img_file = save_img(img, output_dir, task, space, desc, suffix)

    return img, img_file


# LMM and calculate p value using parametric bootstrap
def fit_mixed_models_parametric_bootstrap(formula, dfs, B=500, seed=2025):
    """
    Calls Julia code to perform voxel/vertex LMM parametric bootstrap.
    Inputs:
        - formula: string like 'beta ~ time + (time | subject)'
        - dfs    : list of pandas DataFrames
        - B      : bootstrap iterations
        - seed   : random seed
    Returns:
        - list of tuples per voxel: (b_obs, se_boot, z_boot, p_boot)
    """
    model_cmd = f"""
    using MixedModels
    using Distributions
    using LinearAlgebra
    using Suppressor
    using Random
    using Base.Threads
    using StatsModels
    using DataFrames
    using SparseArrays
    
    @info "Julia running with $(Threads.nthreads()) threads"

    function fit_and_prepare(df, fml)
        #mod = @suppress fit(MixedModel, fml, df)
        
        mod = fit(MixedModel, fml, df)

        # Fixed-effect parameters
        beta_hat = mod.beta
        vc = VarCorr(mod)
        
        # Random-effect covariance (subject)
        # The vc.σρ[1].σ gives the standard deviation vector
        # The vc.σρ[1].ρ gives the correlation matrix
        s = collect(values(vc.σρ[1].σ)) # Use values() to get an array of numbers
        # Get the correlation value from the tuple and construct the 2x2 correlation matrix
        R_val = vc.σρ[1].ρ[1]
        R = Symmetric([1.0 R_val; R_val 1.0])
        Σ_u = Diagonal(s) * R * Diagonal(s)
        # Add a small amount of jitter to the diagonal to ensure positive definiteness
        Σ_u = Symmetric(Σ_u + 1e-6 * I)
        # Residual variance
        sigma2 = mod.σ^2
        
        # Design matrices (fixed and random)
        X = mod.X                            # fixed effect design matrix

        # The reterm `z` matrix is typically `n_obs x n_re_per_group`.
         # We need to reshape it to `n_obs x (n_re_per_group * n_groups)`.
         # First, get the number of random effects per group.
         q = size(Σ_u, 1) # 2 in this case
         
         # Get the subject levels
         subject_levels = mod.reterms[1].levels
         ngrp = length(subject_levels) 
         
         # Initialize a sparse, block-diagonal Z matrix of the correct size.
         Z = spzeros(size(df, 1), ngrp * q)
         
         # Get the random effects design matrix from the reterm
         re_z = mod.reterms[1].z
         
         # Populate the full Z matrix by placing the appropriate blocks.
         for i in 1:size(df, 1)
             subject_idx = findfirst(==(df.subject[i]), subject_levels)
             if subject_idx !== nothing
                 Z[i, (subject_idx - 1) * q + 1 : subject_idx * q] = re_z[:, i]
             end
         end
        
        return mod, beta_hat, Σ_u, sigma2, X, Z, ngrp, q
    end
    
    function simulate_parametric_y!(y_sim, X, Z, beta_hat, Σ_u, sigma2, ngrp, q, rng)
        # Preallocate vector for all random effects
        b = zeros(q * ngrp)
    
        # Sample random intercept/slope per subject
        mv = MvNormal(zeros(q), Σ_u)
        for s in 1:ngrp
            b[(s-1)*q + 1 : s*q] = rand(rng, mv)
        end
    
        # Add noise
        ε = rand(rng, Normal(0, sqrt(sigma2)), size(X,1))
    
        # Simulated outcome
        y_sim .= X * beta_hat .+ Z * b .+ ε

    end
    
    function voxel_bootstrap(df, fml, B, seed)
        df= DataFrame(df)
        local mod, beta_hat, Σ_u, sigma2, X, Z, ngrp, q, b_obs
        # Fit original model once
        try
            mod, beta_hat, Σ_u, sigma2, X, Z, ngrp, q = fit_and_prepare(df, fml)
            b_obs = beta_hat[2]  # slope for time
        catch e
            @warn "initial fit failed: $e"
            return (NaN, NaN, NaN, NaN)
        end
        
        b_star = Vector{{Float64}}(undef, B)
                    # reused data frame
        df_sim = copy(df)
        y_sim = similar(df_sim.beta)
    
        for k in 1:B
            rng = MersenneTwister(seed + k + Threads.threadid())
            try
                simulate_parametric_y!(y_sim, X, Z, beta_hat, Σ_u, sigma2, ngrp, q, rng)
                df_sim[!, :beta] = y_sim
                mod_sim = fit(MixedModel, fml, df_sim)
                b_star[k] = mod_sim.beta[2]

            catch e
                @warn "bootstrap iteration failed: $e"
                b_star[k] = NaN
            end
        end
    
        b_star_valid = b_star[.!isnan.(b_star)]
        if isempty(b_star_valid)
            return (b_obs, NaN, NaN, NaN)
        end   

        # empirical SE and z
        se_boot = std(b_star_valid; corrected=true)
        z_boot = b_obs / se_boot

        # two-sided empirical p-value
        p_boot = sum(abs.(b_star_valid) .>= abs(b_obs)) / length(b_star_valid)

        return (b_obs, se_boot, z_boot, p_boot)
    end


    # Simple timeout function for Julia
    function run_with_timeout(f::Function, seconds::Real)
        t = @async try
            return f()
        catch e
            return e
        end
        timer = Timer(x -> Base.throwto(t, InterruptException()), seconds)
        try
            return fetch(t)
        catch e
            @warn "Fit aborted due to timeout"
            return nothing
        finally
            close(timer)
        end
    end
    
    # Voxel bootstrap with timeout
    function voxelwise_bootstrap_safe(dfs, fml_str::String, B, seed; timeout=60)
        fml = eval(Meta.parse("@formula(" * fml_str * ")"))
        n = length(dfs)
        results = Vector{{NTuple{{4, Float64}}}}(undef, n)
    
        @threads for i in 1:n
            try
                res = run_with_timeout(timeout) do
                    voxel_bootstrap(dfs[i], fml, B, seed)
                end
                if res === nothing || isa(res, Exception)
                    results[i] = (NaN, NaN, NaN, NaN)
                else
                    results[i] = res
                end
            catch e
                @warn "voxel $i failed overall: $e"
                results[i] = (NaN, NaN, NaN, NaN)
            end
            if i % 100 == 0
                @info "Processed voxel $i / $n"
            end
        end
        return results
    end
    
    # Voxel bootstrap without timeout
    function voxelwise_bootstrap(dfs, fml_str::String, B, seed)
    fml = eval(Meta.parse("@formula(" * fml_str * ")"))
        n = length(dfs)
        results = Vector{{NTuple{{4, Float64}}}}(undef, n)
     
        @threads for i in 1:n
            try
                # Directly call the bootstrap function without a timeout wrapper
                res = voxel_bootstrap(dfs[i], fml, B, seed)
                results[i] = res
            catch e
                @warn "voxel $i failed overall: $e"
                results[i] = (NaN, NaN, NaN, NaN)
            end
            if i % 100 == 0
                @info "Processed voxel $i / $n"
            end
        end
        return results
    end

    
    """
    jl.seval(model_cmd)
    fit_boot_julia = jl.seval("voxelwise_bootstrap_safe")
    return fit_boot_julia(dfs, formula, int(B), int(seed))


# FDR correction on bootstraped pvalues
def apply_fdr_on_labels(p_vals, cort_verts, label_paths, out_prefix, hemi,
                        alpha=0.05, per_label=False):
    """
    Inputs:
        - p_vals              : 1D numpy array of p-values (length = n_vertices)
        - gifti_template_path : path to a .gii to copy affine/metadata (must match ordering)
        - label_paths         : list of .label files (strings). If empty, will run FDR across whole surface.
        - out_prefix          : prefix for outputs
        - per_label           : if True, will produce one per-label mask file (FDR applied inside each label).
                                if False, will do FDR on the union of all label vertices (one global FDR across union).
                                
    Return:
        - {'global'/'union' : array of mask, 'p_[global, local, union]_corrected' : array of corrected p values}
    """
    # number of vertices (use first DataArray length) recover to full vertices and mark 1 at non-cortical vertices
    
    n_vert = cort_verts
    if p_vals.shape[0] != n_vert.shape[0]:
        raise ValueError(f"p_vals length ({p_vals.shape[0]}) != gifti vertices ({cort_verts})")
    
    # check nan in p_vals: when the fixed effect is very close to zero, or couldn't find perfect fit within a given time, it 
    #                       returns to nan. Replace nan to 1.
    for i, val in enumerate(p_vals):
        if np.isnan(val):
            p_vals[i] = 1
    
    # full vertices
    full_vert         = nib.load(FREESURFER_DIR/'fsaverage_gii'/f'infl_{hemi}.gii.gz').darrays[0].data
    full_mask         = np.ones(full_vert.shape[0])
    full_mask[n_vert] = p_vals
    
    # save p-value map as a new GIFTI
    p_da  = nib.gifti.GiftiDataArray(full_mask.astype(np.float32))
    p_img = nib.gifti.GiftiImage(darrays=[p_da])
    nib.save(p_img, f"{out_prefix}hemi-{hemi}_p_map.func.gii")
    print(f"Saved p-map: {out_prefix}hemi-{hemi}_p_map.func.gii")

    # If no labels given, apply FDR over whole cortical surface
    if len(label_paths) == 0:
        # prepare the full surface vertices
        full_mask_p = np.ones(full_vert.shape[0])
        full_mask   = np.zeros(full_vert.shape[0])
        
        # FDR correction
        reject, pvals_corrected = fdrcorrection(p_vals, alpha=alpha)
        mask                    = reject.astype(np.uint8)
        full_mask_p[n_vert]     = pvals_corrected
        full_mask[n_vert]       = mask
        mask_da                 = nib.gifti.GiftiDataArray(full_mask.astype(np.float32))
        mask_img                = nib.gifti.GiftiImage(darrays=[mask_da])
        fdr_p_da                = nib.gifti.GiftiDataArray(full_mask_p.astype(np.float32))
        fdr_p_img               = nib.gifti.GiftiImage(darrays=[fdr_p_da])
        nib.save(mask_img, out_prefix + f"hemi-{hemi}_mask_fdr_{alpha}.func.gii")
        nib.save(fdr_p_img, out_prefix + f"hemi-{hemi}_corrected_p_fdr_{alpha}.func.gii")
        print(f"Saved global FDR mask: {out_prefix}hemi-{hemi}_mask_fdr.func.gii")
        return {"global": {"reject_mask": mask, "pvals_corrected": pvals_corrected}}

    results = {}
    if per_label:
        # apply FDR separately inside each label, write one mask per label
        for lab in label_paths:
            verts          = read_label(lab)     # array of vertex indices
            in_mask        = np.zeros(full_vert, dtype=bool)
            in_mask[verts] = True
            p_in           = p_vals[in_mask]
            
            if p_in.size == 0:
                print(f"Label {os.path.basename(lab)} has 0 vertices in this GIFTI; skipping.")
                results[lab] = {"reject_mask": np.zeros(full_vert, dtype=np.uint8)}
                continue
            
            reject_local, p_local_corr = fdrcorrection(p_in, alpha=alpha)
            mask_global                = np.zeros(full_vert, dtype=np.uint8)
            mask_global[verts]         = reject_local.astype(np.uint8)
            
            # save mask per-label
            outfn = f"{out_prefix}hemi-{hemi}_mask_{os.path.basename(lab).replace('.label','')}_fdr.func.gii"
            nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(mask_global)]), outfn)
            print(f"Saved per-label FDR mask: {outfn}")
            
            results[lab] = {"reject_mask": mask_global, "p_local_corrected": p_local_corr}
        return results
    else:
        # union of labels: do one FDR across all vertices in the union
        union_verts = np.zeros(full_vert.shape[0], dtype=bool) # full cortical vertices
        for lab in label_paths:
            verts              = read_label(lab)
            union_verts[verts] = True
            
        p_union = p_vals[union_verts]
        
        if p_union.size == 0:
            raise RuntimeError("Union of labels has zero vertices in this gifti.")
            
        reject_union, p_union_corr = fdrcorrection(p_union, alpha=alpha)
        mask_global                = np.zeros(full_vert, dtype=np.uint8)
        mask_global[union_verts]   = reject_union.astype(np.uint8)
        
        # save union mask
        nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(mask_global)]),
                 f"{out_prefix}hemi-{hemi}_mask_union_fdr.func.gii")
        print(f"Saved union FDR mask: {out_prefix}hemi-{hemi}_mask_union_fdr.func.gii")
        
        return {"union": {"reject_mask": mask_global, "p_union_corrected": p_union_corr}}

# Function: save data as .gii file
def save_surf(data, bg_img, fname):
    """
    
    Input:
        data  - 1 x V vector; data to be written
        img   - GiftiImage; template image object
        fname - string; filename of resulting image
    Return:    
        img   - GiftiImage; resulting image object
    """
    
    # wrap your data as a GiftiDataArray
    beta_array = nib.gifti.GiftiDataArray(data)
    
    # make a copy of the background image
    new_img = nib.gifti.GiftiImage(
        meta=bg_img.meta,
        labeltable=bg_img.labeltable,
        darrays=list(bg_img.darrays) + [beta_array]  # keep original geometry + add new data
    )
    
    # save the new GIFTI
    nib.save(new_img, fname)
    
    return new_img


def save_clusters(img, voxel_threshold, cluster_threshold, output_dir, task,
                  space, contrast_label, suffix):
    """Finds clusters in a z-map and saves them as a table and NIfTI image."""

    voxel_threshold_z = norm.ppf(1 - voxel_threshold / 2)  # p to z

    cluster_df, cluster_imgs = \
        get_clusters_table(img, voxel_threshold_z, cluster_threshold,
                           two_sided=True, return_label_maps=True)

    has_pos_clusters = any(cluster_df['Peak Stat'] > 0)
    has_neg_clusters = any(cluster_df['Peak Stat'] < 0)

    if has_pos_clusters:

        if has_neg_clusters:

            neg_ixs = cluster_df['Peak Stat'] < 0
            cluster_df.loc[neg_ixs, 'Cluster ID'] = \
                '-' + cluster_df.loc[neg_ixs, 'Cluster ID'].astype(str)

            cluster_img = math_img('img_pos - img_neg',
                                   img_pos=cluster_imgs[0],
                                   img_neg=cluster_imgs[1])

        else:

            cluster_img = cluster_imgs[0]

    elif has_neg_clusters:

        neg_ixs = cluster_df['Peak Stat'] < 0
        cluster_df.loc[neg_ixs, 'Cluster ID'] = \
            '-' + cluster_df.loc[neg_ixs, 'Cluster ID'].astype(str)

        cluster_img = math_img('-img', img=cluster_imgs[0])

    else:

        cluster_img = math_img('img - img', img=img)

    save_df(cluster_df, output_dir, task, space, contrast_label,
            suffix=f'{suffix}-clusters')

    save_img(cluster_img, output_dir, task, space, contrast_label,
             suffix=f'{suffix}-clusters')


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
    return file

def save_df(df, output_dir, task, space, desc, suffix):
    """Saves a DataFrame to a TSV file."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.tsv'
    file = output_dir / filename
    df.to_csv(file, sep='\t', index=False, float_format='%.5f')

    return file

def save_fallback(array, output_dir, suffix):
    fallback_file = output_dir / f'failed_save_{suffix}.npy'
    np.save(fallback_file, array)
    print(f"Failed to save NIfTI — array saved to: {fallback_file}")
