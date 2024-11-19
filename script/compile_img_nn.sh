#!/bin/bash

convert_onnx_to_sapeon() {
    # Define color codes
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color

    # Assign arguments to readable variables
    output_dir="$1"      # e.g., img_nn
    onnx_model_file="$2" # e.g., img_nn_opt.onnx

    pushd "$output_dir" >/dev/null

    # Convert the ONNX model to a SAPEON model with specified parameters
    echo -e "${GREEN}Converting ONNX model '${onnx_model_file}' to SAPEON format...${NC}"
    if onnx2sapeon --input "$onnx_model_file" --calib dummy --device_type x330 --output_dir "." --sp_layer_thresh 1 --input_batch 6; then
        echo -e "${GREEN}Conversion successful!${NC}"
    else
        echo -e "${RED}Conversion failed.${NC}"
        popd >/dev/null
        return 1 # Exit the function early on failure
    fi

    # Change directory to the output directory, compile the model for the device, and return to the previous directory
    echo -e "${GREEN}Compiling the SAPEON model...${NC}"
    if snc -i spear_1-1.sp -c x330 -t nf8 -m fast --dump_manual_ps; then
        echo -e "${GREEN}Compilation successful!${NC}"
    else
        echo -e "${RED}Compilation failed.${NC}"
        popd >/dev/null
        return 1 # Exit the function early on failure
    fi
    popd >/dev/null
}

convert_onnx_to_sapeon "img_nn" "img_nn_opt.onnx"
convert_onnx_to_sapeon "img_backbone" "img_backbone_opt.onnx"
convert_onnx_to_sapeon "img_depthnet_wo_aspp" "img_depthnet_wo_aspp_opt.onnx"
convert_onnx_to_sapeon "img_aspp" "aspp_opt.onnx"
