#!/bin/bash

# FreeSurfer Cluster-based ROI Creation Script (Both Hemispheres), surf label to volume label mask
# Usage: ./create_cluster_roi.sh <threshold> <measure> [minarea] [target_template]
# Example: ./create_cluster_roi.sh "1.222983880411199e-04" area 50 /path/to/MNI152_template.nii.gz

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <threshold> <measure> [minarea] [target_template]"
    echo "Example: $0 \"1.222983880411199e-04\" area 50 /path/to/MNI152_template.nii.gz"
    echo ""
    echo "Required arguments:"
    echo "  threshold    : Statistical threshold (e.g., \"1.222983880411199e-04\")"
    echo "  measure      : area, thickness, curv, volume, etc."
    echo ""
    echo "Optional arguments:"
    echo "  minarea      : Minimum cluster area in mm² (default: 50)"
    echo "  target_template : Path to target template (default: MNI152NLin6Asym)"
    exit 1
fi

# Input variables
THRESHOLD="$1"  # Quote to handle scientific notation
MEASURE="$2"
MINAREA=${3:-50}  # Default minarea is 50 if not provided
TARGET_TEMPLATE=${4:-"/data/pt_02825/MPCDF/freesurfer/ICMBMNI152Nlin6thAsym/tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz"}

# Validate threshold format
if ! echo "$THRESHOLD" | grep -qE '^-?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$'; then
    echo "Error: Invalid threshold format. Use quotes for scientific notation."
    echo "Example: ./script.sh \"1.222983880411199e-04\" area"
    exit 1
fi

# Define paths
BASE_DIR="/data/pt_02825/MPCDF/freesurfer"
INPUT_DIR="${BASE_DIR}/lme/Ver_based_res/lin/mat_results"
FSAVERAGE_DIR="${BASE_DIR}/fsaverage"

echo "========================================"
echo "FreeSurfer Cluster ROI Creation (Both Hemispheres)"
echo "========================================"
echo "Threshold: $THRESHOLD"
echo "Measure: $MEASURE"
echo "Min area: $MINAREA mm²"
echo "Target template: $TARGET_TEMPLATE"
echo "========================================"

# Process both hemispheres
declare -a ALL_LABEL_FILES=()

for HEMISPHERE in lh rh; do
    echo ""
    echo "Processing $HEMISPHERE hemisphere..."
    echo "--------------------------------"
    
    # Input and output files for current hemisphere
    INPUT_FILE="${INPUT_DIR}/${MEASURE}_p_${HEMISPHERE}.mgh"
    POS_BIN="${INPUT_DIR}/ROI_masks3/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}_pos.mgh"
    NEG_BIN="${INPUT_DIR}/ROI_masks3/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}_neg.mgh"
    COMBINED_BIN="${INPUT_DIR}/ROI_masks3/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}.mgh"
    NOSUB_BIN="${INPUT_DIR}/ROI_masks3/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}_nosub.mgh"
    CLUSTER_OUT="${INPUT_DIR}/ROI_masks3/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}_clustered"
    CLUSTER_SUM="${INPUT_DIR}/ROI_masks3/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}_sum.txt"
    LABEL_PREFIX="${INPUT_DIR}/ROI_masks3/${MEASURE}/${HEMISPHERE}_${MEASURE}.aparc.label"

    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: Input file not found: $INPUT_FILE"
        echo "Skipping $HEMISPHERE hemisphere..."
        continue
    fi

    # Step 1: Create positive threshold binary mask
    echo "Step 1: Creating positive threshold binary mask for $HEMISPHERE..."
    mri_binarize --i "$INPUT_FILE" \
                 --min 0 --max "$THRESHOLD" \
                 --o "$POS_BIN" --binval 1

    # Step 2: Create negative threshold binary mask
    echo "Step 2: Creating negative threshold binary mask for $HEMISPHERE..."
    NEG_THRESHOLD="-$THRESHOLD"
    mri_binarize --i "$INPUT_FILE" \
                 --min "$NEG_THRESHOLD" --max 0 \
                 --o "$NEG_BIN" --binval 1

    # Step 3: Combine positive and negative masks
    echo "Step 3: Combining positive and negative masks for $HEMISPHERE..."
    mri_concat --i "$POS_BIN" "$NEG_BIN" --sum --o "$COMBINED_BIN"

    # Step 4: Remove subcortical structures
    echo "Step 4: Removing subcortical structures for $HEMISPHERE..."
    mri_binarize --i "$COMBINED_BIN" \
                 --min 1 --max 1 \
                 --o "$NOSUB_BIN" --binval 1

    # Step 5: Create clustered image
    echo "Step 5: Creating clustered image for $HEMISPHERE (minarea: $MINAREA mm²)..."
    mri_surfcluster --in "$NOSUB_BIN" \
                    --subject fsaverage \
                    --hemi $HEMISPHERE \
                    --annot aparc.a2009s \
                    --thmin 1 --thmax 1 \
                    --minarea $MINAREA \
                    --sum "$CLUSTER_SUM" \
                    --o "$CLUSTER_OUT" \
                    --olab "$LABEL_PREFIX"

    # Collect label files from this hemisphere
    HEMI_LABEL_FILES=($(find "$INPUT_DIR/ROI_masks2/${MEASURE}" -name "${HEMISPHERE}_${MEASURE}.aparc.label-*.label" | sort))
    
    if [ ${#HEMI_LABEL_FILES[@]} -eq 0 ]; then
        echo "Warning: No label files found for $HEMISPHERE! Clustering may have failed or no clusters survived."
        echo "Check the summary file: $CLUSTER_SUM"
    else
        echo "Found ${#HEMI_LABEL_FILES[@]} label files for $HEMISPHERE:"
        for label in "${HEMI_LABEL_FILES[@]}"; do
            echo "  - $(basename $label)"
        done
        # Add to the combined array
        ALL_LABEL_FILES+=("${HEMI_LABEL_FILES[@]}")
    fi
done

echo ""
echo "========================================"
echo "Step 6: Converting all labels to volume..."
echo "========================================"

# Check if we have any labels at all
if [ ${#ALL_LABEL_FILES[@]} -eq 0 ]; then
    echo "Error: No label files found from either hemisphere!"
    echo "Processing failed. Check the summary files and input data."
    exit 1
fi

echo "Total label files found: ${#ALL_LABEL_FILES[@]}"
echo "Label files:"
for label in "${ALL_LABEL_FILES[@]}"; do
    echo "  - $(basename $label)"
done

# Output files for combined processing
ORIG_VOLUME="${INPUT_DIR}/ROI_masks3/${MEASURE}/orig_cluster_bilateral_${MEASURE}.nii"
MNI_VOLUME="${INPUT_DIR}/ROI_masks3/${MEASURE}/MNI_cluster_bilateral_${MEASURE}.nii.gz"
REG_FILE="${INPUT_DIR}/ROI_masks3/${MEASURE}/fsaverage_to_mni152.lta"

# Build mri_label2vol command with all labels
LABEL_ARGS=""
for label in "${ALL_LABEL_FILES[@]}"; do
    LABEL_ARGS="$LABEL_ARGS --label $label"
done

# Execute mri_label2vol with all labels
echo "Converting labels to fsaverage volume space..."
mri_label2vol $LABEL_ARGS \
              --temp "${FSAVERAGE_DIR}/mri/orig.mgz" \
              --identity \
              --o "$ORIG_VOLUME"

# Step 7: Create registration file between orig (in) and MNI1526thNlinAsym (out)
echo ""
echo "Step 7: Creating registration file (.lta)..."
mri_coreg --mov "${FSAVERAGE_DIR}/mri/orig.mgz" \
          --targ "$TARGET_TEMPLATE" \
          --reg "$REG_FILE"


# Step 8: Transform to target template
echo ""
echo "Step 8: Transforming to target template..."
mri_vol2vol --mov "$ORIG_VOLUME" \
            --targ "$TARGET_TEMPLATE" \
            --lta "$REG_FILE" \
            --o "$MNI_VOLUME" \
            --nearest

echo ""
echo "========================================"
echo "Processing completed successfully!"
echo "========================================"
echo "Output files:"
echo "  - Left hemisphere summary: ${INPUT_DIR}/ROI_masks2/${MEASURE}/binarized_${MEASURE}_lh_sum.txt"
echo "  - Right hemisphere summary: ${INPUT_DIR}/ROI_masks2/${MEASURE}/binarized_${MEASURE}_rh_sum.txt"
echo "  - Combined fsaverage volume: $ORIG_VOLUME"
echo "  - Combined target space volume: $MNI_VOLUME"
echo "  - Total clusters: ${#ALL_LABEL_FILES[@]}"

# Count clusters per hemisphere
LH_COUNT=$(find "$INPUT_DIR/ROI_masks2/${MEASURE}" -name "lh_${MEASURE}.aparc.label-*.label" | wc -l)
RH_COUNT=$(find "$INPUT_DIR/ROI_masks2/${MEASURE}" -name "rh_${MEASURE}.aparc.label-*.label" | wc -l)
echo "  - Left hemisphere clusters: $LH_COUNT"
echo "  - Right hemisphere clusters: $RH_COUNT"
echo "========================================"

# Clean up intermediate files (optional)
read -p "Do you want to remove intermediate binary files from both hemispheres? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then5
    for HEMISPHERE in lh rh; do
        rm -f "${INPUT_DIR}/ROI_masks2/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}_pos.mgh"
        rm -f "${INPUT_DIR}/ROI_masks2/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}_neg.mgh"
        rm -f "${INPUT_DIR}/ROI_masks2/${MEASURE}/binarized_${MEASURE}_${HEMISPHERE}.mgh"
    done
    echo "Intermediate files removed."
fi

echo "Done!"
