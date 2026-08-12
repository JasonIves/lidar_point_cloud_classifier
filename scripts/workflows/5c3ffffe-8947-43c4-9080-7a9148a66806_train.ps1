$sweep_params = @(
	@{block_size = 15}
)

#MANUALLY GENERATED - SAMPLED VIA ing_hd_20260716.csv
$files = @("LHD_FXX_0741_6247_PTS_LAMB93_IGN69", 
"LHD_FXX_0473_6383_PTS_LAMB93_IGN69", 
"LHD_FXX_0733_6244_PTS_LAMB93_IGN69", 
"LHD_FXX_0738_6247_PTS_LAMB93_IGN69", 
"LHD_FXX_0741_6243_PTS_LAMB93_IGN69", 
"LHD_FXX_0734_6244_PTS_LAMB93_IGN69", 
"LHD_FXX_0928_6561_PTS_LAMB93_IGN69", 
"LHD_FXX_0471_6386_PTS_LAMB93_IGN69", 
"LHD_FXX_0927_6561_PTS_LAMB93_IGN69", 
"LHD_FXX_0472_6383_PTS_LAMB93_IGN69", 
"LHD_FXX_0925_6569_PTS_LAMB93_IGN69", 
"LHD_FXX_0924_6562_PTS_LAMB93_IGN69"
	)



foreach ($p in $sweep_params) {
	##STATIC VARS
	$min_forward_batch = 8 #Minimum forward batch size, for effective batch normalization
	$target_density = 16 #8 + 100% for halo points
	$max_points_over_batches = 57344 # Dictates number of total points that can be accomodated - based on 8GB VRAM on GPU and point budget + activations of PointNet architecture
	$halo_allocation = .48 # Percent of total points to be allocated to halo - At 20% halo (96% of core area) .48 should be approx equal point density (with slight overweight due to density target rounding remaining in core)
	
	##TAG - UPDATE FOR RUN SEPARATION
	# $tag = "block_$($p.block_size)_testing"
    # $tag = "block_$($p.block_size)_points_$($p.target_points)"
	$tag = "block_$($p.block_size)_full_sample"
	
	##CALC VARS
	$target_points = ($p.block_size * $p.block_size) * $target_density #Ideal target points for a single block
	$max_allowed_points_per_block = [Math]::Floor($max_points_over_batches / $min_forward_batch) #Maximum points allowed per block if input batch is n blocks
	$adjusted_target_points = [Math]::Min($target_points, $max_allowed_points_per_block) #Adjust points per block to accomodate batch sizing
	$overlap_margin = $p.block_size / 5 #Overlap halo size, for boundary smoothing - 20% = 96% point count of core
	# $forward_batch_size = [Math]::Floor($max_points_over_batches / $adjusted_target_points) #If batch size is driven by point density
	$forward_batch_size = $min_forward_batch #If point density is driven by batch size

	##DIRECTORIES
    $split_output_dir = "D:/data6700_ext/data/labeled/ign_hd/${tag}_split" # Output directory for this sweep combination's merged train/val split
    $input_list_path = "D:/data6700_ext/data/labeled/ign_hd/${tag}_inputs.txt" # Response file for split-dataset --input-list (avoids the Windows CreateProcessW ~32,767-char command-line limit at 1500+ files)
    Set-Content -Path $input_list_path -Value "# --input-list for sweep: $tag"



    # ==DATA PREPROCESSING==
	# --block-overlap $overlap `
    foreach ($f in $files) {
        $output_dir = "D:/data6700_ext/data/labeled/ign_hd/${tag}/${f}"

        # cargo run --release --bin wb_lidar_train --features training -- preprocess-labeled`
        & (Join-Path $PSScriptRoot '..\wb_lidar_train.ps1') preprocess-labeled `
            --input "D:/data6700_ext/data/raw/ign_hd/$f.copc.laz" `
            --output $output_dir `
            --block-size $p.block_size `
			--block-overlap $overlap_margin `
			--halo-fraction  $halo_allocation `
            --target-points $adjusted_target_points `
            --min-density .75 `
            --min-neighbors 8 `
            --tile-grid 8 `
            --threads 8 `
            --label-map "D:/data6700_ext/config/label_maps/ign_hd_label_map.json" `
            --outlier-removal true `
            --oversample-jitter 0.0 `
			--dtm-resolution 1 `
			--hag-max 50

        # Append this file's output dir to the list file, one line at a
        # time, instead of accumulating a giant --input flag array.
        Add-Content -Path $input_list_path -Value $output_dir
    }

    # ==DATA SPLIT==
	##MOVES SPLITS INTO TARGET DIRS FOR TRAINING
    Write-Host "Starting split-dataset phase for sweep: $tag..."
    # cargo run --release --bin wb_lidar_train --features training -- split-dataset`
    & (Join-Path $PSScriptRoot '..\wb_lidar_train.ps1') split-dataset `
        --input-list $input_list_path `
        --output $split_output_dir `
        --val-split 0.15 `
        --test-split 0.10 `
        --move

    Write-Host "Starting training phase for sweep: $tag..."


    # ==TRAINING==
	# --forward-batch-size 32 `
	# --batch-size 32 `
    # --early-stopping-patience 10 `
	# cargo run --release --bin wb_lidar_train --features training -- train`
	& (Join-Path $PSScriptRoot '..\wb_lidar_train.ps1') train `
		--data-dir "$split_output_dir/train" `
		--val-data-dir "$split_output_dir/val" `
        --output-model "D:/data6700_ext/output/models/ign_hd/ign_hd_${tag}.wbmodel" `
        --metrics-out "D:/data6700_ext/output/metrics/ign_hd/ign_hd_${tag}_metrics.csv" `
        --epochs 150 `
		--forward-batch-size $forward_batch_size `
		--batch-size 32 `
        --learning-rate 0.001 `
        --weight-decay 5e-5 `
        --keep-best-n 3 `
        --n-classes 8 `
        --class-weight-beta 0.9999 `
        --device "auto" `
        --cache-blocks-max-mb 4096 `
        --warmup-steps 5 `
		--early-stopping-patience 25 `
		--use-feature-tnet `
		--checkpoint-dir "D:\data6700_ext\output\models\ign_hd\checkpoints\${tag}"



	# ==EVALUATE TRAINING ON TEST DATA==
	##NO FUSION
	# cargo run --release --bin wb_lidar_train  --features training -- evaluate`
	& (Join-Path $PSScriptRoot '..\wb_lidar_train.ps1') evaluate `
		--model "D:/data6700_ext/output/models/ign_hd/ign_hd_${tag}.wbmodel" `
		--data-dir "$split_output_dir/test" `
		--metrics-out "D:/data6700_ext/output/metrics/ign_hd/eval/ign_hd_${tag}_class.csv" `
		--confusion-out "D:/data6700_ext/output/metrics/ign_hd/eval/ign_hd_${tag}_conf_mtrx.csv" `
		--threads 8
}