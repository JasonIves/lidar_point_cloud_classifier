$files = @("LHD_FXX_0737_6246_PTS_LAMB93_IGN69"
	)

$block_size = 15

foreach ($f in $files) {
	$tag = "block_${block_size}_miou_exmpl_classify"

    # ==DATA PREPROCESSING==
	$output_dir = "D:/data6700_ext/data/raw/ign_hd/preprocessed/${tag}/${f}"

	# cargo run --release --bin wb_lidar_classify -- preprocess`
	& (Join-Path $PSScriptRoot '..\wb_lidar_classify.ps1') preprocess `
		--input "D:/data6700_ext/data/raw/ign_hd/${f}.copc.laz" `
		--output $output_dir `
		--block-size $block_size `
		--block-overlap 3 `
		--halo-fraction  .48 `
		--target-points 3600 `
		--min-density .75 `
		--min-neighbors 8 `
		--threads 8 `
		--outlier-removal true `
		--oversample-jitter 0.0 `
		--dtm-resolution 1 `
		--hag-max 50
	
	$classify_args = @(
		# "run", "--release", "--bin", "wb_lidar_classify", "--", "classify"
	)
	
	# Append the rest of the fixed parameters
	$classify_args += @(
		"--input", "D:/data6700_ext/data/raw/ign_hd/${f}.copc.laz", 
		"--model", "D:/data6700_ext/output/models/ign_hd/ign_hd_block_15_full_sample.wbmodel", 
		"--blocks", "D:/data6700_ext/data/raw/ign_hd/preprocessed/${tag}/${f}/blocks.json", 
		"--output", "D:/data6700_ext/data/classified/ign_hd/${f}_classified.las", 
		"--threads", "8"
	)
	
	Write-Host $classify_args

	# Execute the training step using the call operator
	Write-Host "Starting classification: $tag..."
	# & cargo $classify_args
	& (Join-Path $PSScriptRoot '..\wb_lidar_classify.ps1') classify `
	    --input "D:/data6700_ext/data/raw/ign_hd/${f}.copc.laz" `
	    --model "D:/data6700_ext/output/models/ign_hd/ign_hd_block_15_full_sample.wbmodel" `
	    --blocks "D:/data6700_ext/data/raw/ign_hd/preprocessed/${tag}/${f}/blocks.json" `
	    --output "D:/data6700_ext/data/classified/ign_hd/${f}_classified.las" `
	    --threads 8
}