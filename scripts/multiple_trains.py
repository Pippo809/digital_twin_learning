import subprocess

# Set the script name
script_name = "scripts/main.py"
script_name_dbg = "digital_twin_learning/scripts/main.py"

# Set the arguments
exp1 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_1", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test"]

exp2 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
        "--dropout_height", "0.1", "--max_c", "0.3", "--max_s", "0.3"]

exp3 = ["--experiment_name", "rand_gen_3_success", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "rand_gen_3_success", "--save_pred", "--num_epochs", "12"]

exp4 = ["--experiment_name", "mix3_success_downsample_0.3", "--world", "mix3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success_downsample_0.3", "--save_pred", "--num_epochs", "9", "--downsample", "0.3"]

exp5 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
        "--dropout_height", "0.0", "--max_c", "0.3", "--max_s", "0.3"]

exp6 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_5", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
        "--dropout_height", "0.1", "--max_c", "0.6", "--max_s", "0.6"]

exp7 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_6", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
        "--dropout_height", "0.0", "--max_c", "0.9", "--max_s", "0.3"]

exp8 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_7", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
        "--dropout_height", "0.0", "--max_c", "0.3", "--max_s", "0.9"]

exp9 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_8", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
        "--dropout_height", "0.0", "--max_c", "0.9", "--max_s", "0.9"]

exp10 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_9_downsample_0.4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "0.0", "--max_c", "0.4", "--max_s", "0.4", "--downsample", "0.4"]

exp11 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_10", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "0.9", "--max_s", "0.3"]

exp12 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_11", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "0.3", "--max_s", "0.9"]

exp13 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_12", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "0.9", "--max_s", "0.9"]

exp14 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_13", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "0.3", "--max_s", "0.3"]

exp15 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_14", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
         "--dropout_height", "1.1", "--max_c", "0.6", "--max_s", "0.6"]

exp16 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_15", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "0.1", "--max_s", "0.1"]

exp17 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_16", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "0.2", "--max_s", "0.2"]

exp18 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_17", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "0.3", "--max_s", "0.1"]

exp19 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_18", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "0.1", "--max_s", "0.3"]

exp20 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_19", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
         "--dropout_height", "0.5", "--max_c", "0.3", "--max_s", "0.3"]

exp21 = ["--experiment_name", "rand_gen_sampling_debug", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling", "--save_pred", "--num_epochs", "12"]

exp22 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_20", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "0.9", "--max_c", "0.9", "--max_s", "0.9"]

exp23 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_21", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "0.9", "--max_c", "0.8", "--max_s", "0.8"]

exp24 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_22", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "0.9", "--max_c", "0.9", "--max_s", "0.3"]

exp25 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_23", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
         "--dropout_height", "0.9", "--max_c", "0.3", "--max_s", "0.9"]

exp26 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_24", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
         "--dropout_height", "0.9", "--max_c", "0.7", "--max_s", "0.7"]

exp27 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_25_downsample_0.2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "0.9", "--max_c", "0.3", "--max_s", "0.3", "--downsample", "0.2"]

exp28 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_26_downsample_0.3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "0.9", "--max_c", "0.3", "--max_s", "0.3", "--downsample", "0.3"]

exp29 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_27_downsample_0.4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
         "--dropout_height", "0.9", "--max_c", "0.3", "--max_s", "0.3", "--downsample", "0.4"]

exp30 = ["--experiment_name", "mix3_success_on_rand_gen_3_jitter_test_28_downsample_0.5", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
         "--dropout_height", "0.9", "--max_c", "0.3", "--max_s", "0.3", "--downsample", "0.5"]

exp31 = ["--experiment_name", "rand_gen_sampling_debug", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_2", "--save_pred", "--num_epochs", "9"]

exp32 = ["--experiment_name", "rand_gen_sampling_3", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_3", "--save_pred", "--num_epochs", "9"]

exp33 = ["--experiment_name", "mix3_sampling", "--world", "mix3_standability", "--metric", "success",
         "--cached_model", "mix3_sampling", "--save_pred", "--num_epochs", "9"]

exp34 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.2"]

exp35 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.3"]

exp36 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.4"]

exp37 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.5", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.5"]

exp38 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.6", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.6"]

exp39 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.7", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.7"]

exp40 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.8", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.8"]

exp41 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.9", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.9"]

exp42 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.01", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.01"]

exp43 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.02", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.02"]

exp44 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.03", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.03"]

exp45 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.04", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.04"]

exp46 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.05", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.05"]

exp47 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.06", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.06"]

exp48 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.07", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.07"]

exp49 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.08", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.08"]

exp50 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.09", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.09"]

exp51 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.1", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.1"]

exp52 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.12", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.12"]

exp53 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.15", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.15"]

exp54 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.1", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.1",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp55 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.2",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp56 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.3",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp57 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.4",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp58 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.5", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.5",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp59 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.6", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.6",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp60 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.7", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.7",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp61 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.8", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.8",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp62 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.9", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.9",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp63 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.95", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.9",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp64 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.99", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.9",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp65 = ["--experiment_name", "sampled_rand_gen_3", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_3", "--save_pred", "--voxel_size_z", "5.0", "--overlap", "0.2"]

exp66 = ["--experiment_name", "debug", "--world", "rand_gen_3", "--metric", "success", "--use_poses", "--num_samples", "300",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.9",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0."]

exp67 = ["--experiment_name", "sampled_rand_gen_3_debug", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_3", "--save_pred", "--evaluate_only"]

exp68 = ["--experiment_name", "rand_gen_3_sampling_4", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_4", "--save_pred", "--num_epochs", "10"]

exp69 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.1", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.1",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp70 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.2",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp71 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.3",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp72 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.4",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp73 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.5", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.5",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp74 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.6", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.6",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp75 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.7", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.7",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp76 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.75", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp77 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.8", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.8",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp78 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.9", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.9",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp79 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.95", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.95",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp80 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.99", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.99",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp81 = ["--experiment_name", "mix3_success_on_rand_gen_3_dropout_0.85", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.85",
         "--dropout_height", "1.0", "--max_c", "1", "--max_s", "0."]

exp82 = ["--experiment_name", "rand_gen_3_sampling_with_voxels", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_with_voxels", "--save_pred", "--num_epochs", "9"]

exp83 = ["--experiment_name", "rand_gen_3_sampling_with_voxels_debug", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_with_voxels_debug", "--save_pred", "--num_epochs", "9", "--num_samples", "100000"]

exp83 = ["--experiment_name", "rand_gen_3_sampling_with_voxels_3", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_with_voxels_3", "--save_pred", "--num_epochs", "9"]

exp84 = ["--experiment_name", "rand_gen_3_sampling_with_voxels_visual", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_with_voxels_2", "--save_pred", "--evaluate_only", "--epoch_model", "3", "--num_samples", "200000"]

exp85 = ["--experiment_name", "rand_gen_3_sampling_with_voxels_4", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_with_voxels_3", "--save_pred", "--num_epochs", "9"]

exp86 = ["--experiment_name", "rand_gen_3_sampling_with_voxels_evaluation", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_with_voxels_3", "--save_pred", "--num_epochs", "9", "--evaluate_only", "--epoch_model", "8"]

exp87 = ["--experiment_name", "rand_gen_3_sampling_binnata", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_binnata", "--save_pred", "--num_epochs", "9"]

exp87 = ["--experiment_name", "rand_gen_3_sampling_only_pos", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_only_pos", "--save_pred", "--num_epochs", "9"]

exp88 = ["--experiment_name", "mix_3_sampling_only_pos", "--world", "mix3_standability", "--metric", "success",
         "--cached_model", "mix_3_sampling_only_pos", "--save_pred", "--num_epochs", "9"]

exp89 = ["--experiment_name", "mix3_success_with_noise_0.11_clip_0.09", "--world", "mix3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "0.09", "--max_s", "0.11"]

exp90 = ["--experiment_name", "rand_gen_3_sampling_encoder_only", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_encoder_only", "--save_pred", "--num_epochs", "9"]

exp91 = ["--experiment_name", "rand_gen_3_sampling_encoder_only_poisson_loss", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_encoder_only_poisson_loss", "--save_pred", "--num_epochs", "9"]

exp92 = ["--experiment_name", "rand_gen_3_sampling_evaluation_2", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_only_pos", "--save_pred", "--evaluate_only", "--epoch_model", "8"]

exp93 = ["--experiment_name", "rand_gen_3_sampling_create_nav_graph", "--world", "rand_gen_3_standability", "--metric", "success",
         "--cached_model", "rand_gen_3_sampling_only_pos", "--save_pred", "--evaluate_only", "--epoch_model", "7"]

exp94 = ["--experiment_name", "mix3_success_on_rand_gen_3_noise_0.2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.2"]

exp95 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.3"]

exp96 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.4"]

exp97 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.5", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.5"]

exp98 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.6", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.6"]

exp99 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.7", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
         "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
         "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.7"]

exp100 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.8", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.8"]

exp101 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.9", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.9"]

exp102 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.01", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.01"]

exp103 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.02", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.02"]

exp104 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.03", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.03"]

exp105 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.04", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.04"]

exp106 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.05", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.05"]

exp107 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.06", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.06"]

exp108 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.07", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.07"]

exp109 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.08", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.08"]

exp110 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.09", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.09"]

exp111 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.1", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.1"]

exp112 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.12", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.12"]

exp113 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.15", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.15"]

exp114 = ["--experiment_name", "rand_gen_3_success_from_sampled_world", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_standability_learned"]

exp115 = ["--experiment_name", "rand_gen_3_sampling_pos_orient_class", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pos_orient_class", "--save_pred", "--num_epochs", "3", "--voxel_size_z", "2.0", "--overlap", "0.5", "--num_samples", "200000"]

exp116 = ["--experiment_name", "rand_gen_3_sampling_binnata_binary", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_binnata_binary", "--save_pred", "--num_epochs", "9"]

exp117 = ["--experiment_name", "rand_gen_3_sampling_binary_encoder", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_binary_encoder", "--save_pred", "--num_epochs", "9", "--voxel_size_z", "2.0", "--overlap", "0.5", "--num_samples", "50000"]

exp118 = ["--experiment_name", "rand_gen_3_train_map_enc", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_train_map_enc", "--save_pred", "--num_epochs", "6"]

exp119 = ["--experiment_name", "rand_gen_3_sampling_pretrained_encoder_epoch_0_lr_0.001", "--world", "rand_gen_3_standability", "--metric", "success", "--epoch_model", "0",
          "--cached_model", "rand_gen_3_sampling_pretrained_encoder", "--save_pred", "--num_epochs", "3", "--voxel_size_z", "2.0",
          "--overlap", "0.5", "--num_samples", "100000", "--encoder_model", "rand_gen_3_sampling_binary_encoder"]

exp120 = ["--experiment_name", "rand_gen_3_sampling_pretrained_encoder_epoch_0_lr_0.001_plot", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_encoder", "--save_pred", "--num_epochs", "3", "--voxel_size_z", "2.0",
          "--overlap", "0.5", "--num_samples", "100000", "--evaluate_only", "--epoch_model", "2_finetuned"]

exp121 = ["--experiment_name", "rand_gen_3_sampling_generate_graph", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_encoder", "--save_pred", "--num_epochs", "3", "--voxel_size_z", "2.0",
          "--overlap", "0.5", "--evaluate_only", "--epoch_model", "2_finetuned", "--evaluate_only"]

exp122 = ["--experiment_name", "rand_gen_3_success_from_sampled_world_correct", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_standability_learned"]

exp123 = ["--experiment_name", "rand_gen_3_train_map_enc_finetune", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_train_map_enc", "--save_pred", "--num_epochs", "6", "--epoch_model" "5"]

exp124 = ["--experiment_name", "LEE_sampling", "--world", "ETH_LEE_cropped", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_encoder", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.75", "--evaluate_only", "--epoch_model", "2_finetuned", "--evaluate_only"]

exp125 = ["--experiment_name", "LEE_success_from_sampled_world_correct", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "ETH_LEE_cropped_learned"]

exp126 = ["--experiment_name", "mix_3_train_binary_enc_10_voxel", "--world", "mix3_standability", "--metric", "success", "--use_poses",
          "--cached_model", "mix_3_train_binary_enc_10_voxel", "--save_pred", "--num_epochs", "24", "--num_samples", "100000",
          "--voxel_size_z", "10.0", "--voxel_size_z", "10.0", "--voxel_size_z", "10.0", "--overlap", "0.1"]

exp127 = ["--experiment_name", "mix_3_train_map_enc_finetune", "--world", "mix3_standability", "--metric", "success", "--use_poses",
          "--cached_model", "mix_3_train_binary_10_voxel", "--save_pred", "--num_epochs", "3", "--epoch_model", "0", "--num_samples", "100000",
          "--voxel_size_z", "10.0", "--voxel_size_z", "10.0", "--voxel_size_z", "10.0", "--overlap", "0.1",
          "--encoder_model", "mix_3_train_binary_enc_10_voxel"]

exp128 = ["--experiment_name", "mix_3_train_map_enc_finetune_eval", "--world", "mix3_standability", "--metric", "success", "--use_poses",
          "--cached_model", "mix_3_train_binary_10_voxel", "--save_pred", "--num_epochs", "3", "--epoch_model", "0_finetuned", "--num_samples", "100000",
          "--voxel_size_z", "10.0", "--voxel_size_z", "10.0", "--voxel_size_z", "10.0", "--overlap", "0.1", "--evaluate_only"
          ]

exp128 = ["--experiment_name", "mix_3_train_map_enc_finetune_eval", "--world", "mix3_standability", "--metric", "success", "--use_poses",
          "--cached_model", "mix_3_train_binary_10_voxel", "--save_pred", "--num_epochs", "3", "--epoch_model", "0_finetuned", "--num_samples", "100000",
          "--voxel_size_z", "10.0", "--voxel_size_z", "10.0", "--voxel_size_z", "10.0", "--overlap", "0.1", "--evaluate_only"
          ]

exp129 = ["--experiment_name", "mix3_edge_success_4m_voxel", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_4m_voxel", "--save_pred", "--num_epochs", "12"]

exp130 = ["--experiment_name", "mix3_plot4final", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_4m_voxel", "--save_pred", "--num_epochs", "12"]

exp131 = ["--experiment_name", "mix3_edge_success_4m_voxel_on_LEE", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_4m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12"]

exp132 = ["--experiment_name", "mix3_edge_success_4m_voxel_on_HPH", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_4m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12"]

exp133 = ["--experiment_name", "mix3_edge_success_4m_voxel_on_rand_gen_3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_4m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12"]

exp134 = ["--experiment_name", "rand_gen_3_on_ETH_LEE", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_success", "--evaluate_only", "--save_pred", "--num_epochs", "12"]

exp135 = ["--experiment_name", "rand_gen_3_on_ETH_HPH", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_success", "--evaluate_only", "--save_pred", "--num_epochs", "12"]

exp136 = ["--experiment_name", "rand_gen_3_on_mix3", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_success", "--evaluate_only", "--save_pred", "--num_epochs", "12"]

exp136b = ["--experiment_name", "mix3_LEE", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
           "--cached_model", "mix3_success", "--evaluate_only", "--save_pred", "--num_epochs", "12"]

exp137 = ["--experiment_name", "mix3_edge_success_1m_voxel", "--world", "mix3", "--metric", "success", "--use_poses", "--epoch_model", "6"
          "--cached_model", "mix3_edge_success_1m_voxel_b", "--save_pred", "--num_epochs", "6", "--num_samples", "300000"]

exp138 = ["--experiment_name", "mix3_edge_success_1m_voxel_on_LEE", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_1m_voxel_b", "--evaluate_only", "--save_pred"]

exp139 = ["--experiment_name", "mix3_edge_success_1m_voxel_on_HPH", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_1m_voxel_b", "--evaluate_only", "--save_pred"]

exp140 = ["--experiment_name", "mix3_edge_success_1m_voxel_on_rand_gen_3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_1m_voxel_b", "--evaluate_only", "--save_pred"]

exp141 = ["--experiment_name", "mix_3_sampling_pretrained_encoder_4m_lr_0.001", "--world", "mix_3_standability", "--metric", "success", "--epoch_model", "0",
          "--cached_model", "mix_3_sampling_pretrained_encoder_4m", "--save_pred", "--num_epochs", "3", "--voxel_size_x", "4.0", "--voxel_size_y", "4.0", "voxel_size_z", "4.0"
          "--overlap", "0.1", "--num_samples", "100000", "--encoder_model", "mix3_edge_success_4m_voxel"]

exp142 = ["--experiment_name", "mix3_success_downsample_0.75", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--downsample", "0.75",
          "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0"]

exp143 = ["--experiment_name", "mix3_success_downsample_0.80", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--downsample", "0.80",
          "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0"]

exp144 = ["--experiment_name", "mix3_success_downsample_0.85", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--downsample", "0.85",
          "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0"]

exp145a = ["--experiment_name", "mix3_success_downsample_0.9", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
           "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--downsample", "0.9",
           "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0"]

exp145 = ["--experiment_name", "mix3_success_downsample_0.95", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--downsample", "0.95",
          "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0"]

exp146 = ["--experiment_name", "mix3_success_downsample_0.35", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--downsample", "0.35",
          "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0"]

exp147 = ["--experiment_name", "mix3_edge_success_3m_voxel", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_3m_voxel", "--save_pred", "--num_epochs", "5", "--num_samples", "300000",
          "--voxel_size_x", "3.0", "--voxel_size_y", "3.0", "--voxel_size_z", "3.0"]

exp148 = ["--experiment_name", "mix3_edge_success_3m_voxel_on_LEE", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_3m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "3.0", "--voxel_size_y", "3.0", "--voxel_size_z", "3.0"]

exp149 = ["--experiment_name", "mix3_edge_success_3m_voxel_on_HPH", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_3m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "3.0", "--voxel_size_y", "3.0", "--voxel_size_z", "3.0"]

exp150 = ["--experiment_name", "mix3_edge_success_3m_voxel_on_rand_gen_3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_3m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "3.0", "--voxel_size_y", "3.0", "--voxel_size_z", "3.0"]

exp151 = ["--experiment_name", "mix3_edge_success_5m_voxel", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_5m_voxel", "--save_pred", "--num_epochs", "5", "--num_samples", "300000",
          "--voxel_size_x", "5.0", "--voxel_size_y", "5.0", "--voxel_size_z", "5.0"]

exp152 = ["--experiment_name", "mix3_edge_success_5m_voxel_on_LEE", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_5m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "5.0", "--voxel_size_y", "5.0", "--voxel_size_z", "5.0"]

exp153 = ["--experiment_name", "mix3_edge_success_5m_voxel_on_HPH", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_5m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "5.0", "--voxel_size_y", "5.0", "--voxel_size_z", "5.0"]

exp154 = ["--experiment_name", "mix3_edge_success_5m_voxel_on_rand_gen_3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_5m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "5.0", "--voxel_size_y", "5.0", "--voxel_size_z", "5.0"]

exp155 = ["--experiment_name", "mix3_edge_success_0.5m_voxel", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_0.5m_voxel", "--save_pred", "--num_epochs", "5", "--num_samples", "300000",
          "--voxel_size_x", "0.5", "--voxel_size_y", "0.5", "--voxel_size_z", "0.5"]

exp156 = ["--experiment_name", "mix3_edge_success_0.5m_voxel_on_LEE", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_0.5m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "0.5", "--voxel_size_y", "0.5", "--voxel_size_z", "0.5"]

exp157 = ["--experiment_name", "mix3_edge_success_0.5m_voxel_on_HPH", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_0.5m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "0.5", "--voxel_size_y", "0.5", "--voxel_size_z", "0.5"]

exp158 = ["--experiment_name", "mix3_edge_success_0.5m_voxel_on_rand_gen_3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_0.5m_voxel", "--evaluate_only", "--save_pred", "--num_epochs", "12",
          "--voxel_size_x", "0.5", "--voxel_size_y", "0.5", "--voxel_size_z", "0.5"]

exp159 = ["--experiment_name", "mix3_success_with_noise_0.1", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--jitter_train", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1.0", "--max_s", "0.1", "--num_epochs", "5", "--num_samples", "300000"]

exp160 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.01", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.01"]

exp161 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.02", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.02"]

exp162 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.03", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.03"]

exp163 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.04", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.04"]

exp164 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.05", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.05"]

exp165 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.06", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.06"]

exp166 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.07", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.07"]

exp167 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.08", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.08"]

exp168 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.09", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.09"]

exp169 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.1", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.1"]

exp170 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.12", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.12"]

exp171 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.15", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.15"]

exp172 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.2"]

exp173 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.3"]

exp174 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.4"]

exp175 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.5", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.5"]

exp176 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.6", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.6"]

exp177 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.7", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.7"]

exp178 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.8", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.8"]

exp179 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.9", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.9"]

exp180 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_1.0", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.1", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "1.0"]

exp181 = ["--experiment_name", "mix3_success_with_noise_0.15", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--jitter_train", "--max_dropout_ratio", "0.0", "--epoch_model", "0",
          "--dropout_height", "0.9", "--max_c", "1.0", "--max_s", "0.15", "--num_epochs", "4", "--num_samples", "300000"]

exp182 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.01", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.01"]

exp183 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.02", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.02"]

exp184 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.03", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.03"]

exp185 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.04", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.04"]

exp186 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.05", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.05"]

exp187 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.06", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.06"]

exp188 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.07", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.07"]

exp189 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.08", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.08"]

exp190 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.09", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.09"]

exp191 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.1", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.1"]

exp192 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.12", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.12"]

exp193 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.15", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.15"]

exp194 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.2"]

exp195 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.3"]

exp196 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.4"]

exp197 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.5", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.5"]

exp198 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.6", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.6"]

exp199 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.7", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.7"]

exp200 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.8", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.8"]

exp201 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_0.9", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.9"]

exp202 = ["--experiment_name", "mix3_success_noise_train_on_rand_gen_3_noise_1.0", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success_noise_0.15", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "1.0"]

exp203 = ["--experiment_name", "mix3_CoT_on_rand_gen_3", "--world", "rand_gen_3", "--use_poses",
          "--cached_model", "mix3_CoT2", "--save_pred", "--evaluate_only", "--metric", "CoT"]

exp204 = ["--experiment_name", "rand_gen_3_sampling_pretrained_success_class_score", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score", "--save_pred", "--num_epochs", "6", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--overlap", "0.1", "--num_samples", "300000", "--encoder_model", "rand_gen_3_success", "--epoch_model", "8"]

exp205 = ["--experiment_name", "rand_gen_3_sampling_generate_graph_final", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.3", "--evaluate_only", "--epoch_model", "3_finetuned", "--evaluate_only"]

exp206 = ["--experiment_name", "rand_gen_3_sampling_generate_graph_final", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.3", "--evaluate_only", "--epoch_model", "3_finetuned", "--evaluate_only"]

exp207 = ["--experiment_name", "rand_gen_3_sampling_generate_graph_final_success", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "rand_gen_3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_standability_learned",
          "--epoch_model", "8"]

exp208 = ["--experiment_name", "rand_gen_3_sampling_pretrained_success_debug", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score", "--save_pred", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--overlap", "0.1", "--num_samples", "300000", "--evaluate_only"]

exp209 = ["--experiment_name", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--save_pred", "--num_epochs", "6", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--overlap", "0.1", "--num_samples", "300000", "--encoder_model", "rand_gen_3_success", "--epoch_model", "8"]

exp210 = ["--experiment_name", "rand_gen_3_sampling_pretrained_success_with_shift_plot", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--epoch_model", "1_finetuned", "--save_pred", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--overlap", "0.1", "--num_samples", "300000", "--evaluate_only"]

exp211 = ["--experiment_name", "rand_gen_3_sampling_generate_graph_final_shifting", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.3", "--evaluate_only", "--epoch_model", "1_finetuned"]

exp212 = ["--experiment_name", "rand_gen_3_sampling_generate_graph_final_success", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_standability_learned",
          ]

exp213 = ["--experiment_name", "rand_gen_3_sampling_generate_graph_final_shifting_bis", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.75", "--evaluate_only", "--epoch_model", "1_finetuned"]

exp214 = ["--experiment_name", "rand_gen_3_sampling_generate_graph_final_success_FINAL", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_standability_learned_PPT",
          ]

exp215 = ["--experiment_name", "rand_gen_3_sampling_generate_graph_FINAL_finer", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.12", "--evaluate_only", "--epoch_model", "1_finetuned", "--stem", "PPT"]

exp216 = ["--experiment_name", "AAA_mix3_sampled", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "mix3_learned_0",
          ]

exp217 = ["--experiment_name", "AAA_mix3_sampled", "--world", "mix3", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.1", "--evaluate_only", "--epoch_model", "1_finetuned", "--stem", "0"]

exp218 = ["--experiment_name", "AAA_HPH_sampled", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "ETH_HPH_learned_0",
          ]

exp219 = ["--experiment_name", "AAA_HPH_sampled", "--world", "ETH_HPH", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.1", "--evaluate_only", "--epoch_model", "1_finetuned", "--stem", "0"]

exp220 = ["--experiment_name", "AAA_LEE_sampled", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "ETH_LEE_cropped_learned_0",
          ]

exp221 = ["--experiment_name", "AAA_LEE_sampled", "--world", "ETH_LEE_cropped", "--metric", "success",
          "--cached_model", "rand_gen_3_sampling_pretrained_success_class_score_with_shifting", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.1", "--evaluate_only", "--epoch_model", "1_finetuned", "--stem", "0"]

exp222 = ["--experiment_name", "mix3_edge_success_small_repres_2", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_small_repres_2", "--save_pred", "--num_epochs", "1", "--num_samples", "3000"]

exp223 = ["--experiment_name", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--world", "rand_gen_3_standability", "--metric", "success",
          "--cached_model", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--epoch_model", "", "--save_pred", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--overlap", "0.1", "--num_samples", "5000", "--encoder_model", "mix3_edge_success_small_repres_2", "--num_epochs", "1", "--sample"]

exp224 = ["--experiment_name", "AAA_mix3_sampled_2", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "mix3_learned_0",
          ]

exp225 = ["--experiment_name", "AAA_mix3_sampled_2", "--world", "mix3", "--metric", "success",
          "--cached_model", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.5", "--evaluate_only", "--epoch_model", "2_finetuned", "--stem", "0"]

exp226 = ["--experiment_name", "mix3_CoT_on_rand_gen_3", "--world", "mix3", "--use_poses",
          "--cached_model", "mix3_CoT_smaller_repres", "--save_pred", "--metric", "CoT", "--num_epochs", "12", "--num_samples", "500000"]

exp227 = ["--experiment_name", "mix3_edge_success_small_repres", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_small_repres_2", "--save_pred", "--num_epochs", "12", "--num_samples", "500000"]

exp228 = ["--experiment_name", "AAA_rand_gen_3_sampled_2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_learned_mix_3",
          ]

exp229 = ["--experiment_name", "AAA_rand_gen_3_sampled_2", "--world", "rand_gen_3", "--metric", "success",
          "--cached_model", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.92", "--evaluate_only", "--epoch_model", "2_finetuned", "--stem", "mix_3"]

exp230 = ["--experiment_name", "AAA_rand_gen_3_sampled_3", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_learned_mix_3_2",
          ]

exp231 = ["--experiment_name", "AAA_rand_gen_3_sampled_3", "--world", "rand_gen_3", "--metric", "success",
          "--cached_model", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "1.5", "--evaluate_only", "--epoch_model", "2_finetuned", "--stem", "mix_3_2"]

exp232 = ["--experiment_name", "AAA_HPH_sampled_3", "--world", "ETH_HPH", "--metric", "success",
          "--cached_model", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "1.5", "--evaluate_only", "--epoch_model", "2_finetuned", "--stem", "mix_3"]

exp233 = ["--experiment_name", "AAA_LEE_sampled_3", "--world", "ETH_LEE_cropped", "--metric", "success",
          "--cached_model", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "1.5", "--evaluate_only", "--epoch_model", "2_finetuned", "--stem", "mix_3"]

exp234 = ["--experiment_name", "AAA_HPH_sampled_3", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "ETH_HPH_learned_mix_3",
          ]

exp235 = ["--experiment_name", "AAA_LEE_sampled_3", "--world", "ETH_LEE_cropped", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "ETH_LEE_cropped_learned_mix_3",
          ]

exp236 = ["--experiment_name", "AAA_mix3_sampled_3", "--world", "mix3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "mix3_learned_smaller",
          ]

exp237 = ["--experiment_name", "AAA_mix3_sampled_3", "--world", "mix3", "--metric", "success",
          "--cached_model", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "1.25", "--evaluate_only", "--epoch_model", "2_finetuned", "--stem", "smaller"]

exp238 = ["--experiment_name", "AAA_rand_gen_3_sampled_4", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_learned_mix_3_2",
          ]

exp239 = ["--experiment_name", "HPH_mix3", "--world", "ETH_HPH", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_success", "--save_pred", "--evaluate_only",
          ]

exp240 = ["--experiment_name", "CoT_mix3", "--world", "mix3", "--metric", "CoT", "--use_poses",
          "--cached_model", "CoT_mix3", "--save_pred"]

exp241 = ["--experiment_name", "CoT_mix3_on_HPH", "--world", "ETH_HPH", "--metric", "CoT", "--use_poses",
          "--cached_model", "CoT_mix3", "--save_pred", "--evaluate_only"]

exp242 = ["--experiment_name", "CoT_mix3_on_LEE", "--world", "ETH_LEE_cropped", "--metric", "CoT", "--use_poses",
          "--cached_model", "CoT_mix3", "--save_pred", "--evaluate_only"]


## DEBUGGING ##
exp243 = ["--experiment_name", "FINAL_mix3_regression_train", "--world", "mix3", "--metric", "success",
          "--cached_model", "FINAL_mix3_regression_train", "--save_pred", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--num_samples", "5000",  "--num_epochs", "1"]

exp244 = ["--experiment_name", "rand_gen_3_edge_success_small_repres_eval_edge_succ", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "and_gen_3_edge_success_small_repres", "--save_pred", "--num_epochs", "1", "--evaluate_only", ]

exp245 = ["--experiment_name", "mix3debug_success_noise_train_on_rand_gen_3_noise_0.7", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "mix3_edge_success_small_repres_5", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.7"]

exp246 = ["--experiment_name", "AAA_mix3_sampled_3", "--world", "rand_gen_3", "--metric", "success", "--use_poses", "--sample",
          "--cached_model", "mix_3_sampling_debug", "--save_pred", "--evaluate_only", "--overlap", "0.15", "--stem", "rand_gen_3"]

exp247 = ["--experiment_name", "AAA_mix3_sampled_3_edge_graph", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "and_gen_3_edge_success_small_repres", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_learned_rand_gen_3",
          ]

exp248 = ["--experiment_name", "AAA_mix3_sampled_3", "--world", "rand_gen_3_standability", "--metric", "success", "--sample",
          "--cached_model", "mix_3_sampling_pretrained_success_with_shift_smaller_map", "--save_pred", "--voxel_size_z", "2.0",
          "--overlap", "0.15", "--evaluate_only", "--epoch_model", "2_finetuned"]


## Final experiments
exp249 = ["--experiment_name", "FINAL_mix3_regression_train", "--world", "mix3", "--metric", "success",
          "--cached_model", "FINAL_mix3_regression_train", "--save_pred", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--num_samples", "5000",  "--num_epochs", "1"]

exp250 = ["--experiment_name", "FINAL_mix3_regression_train_cot", "--world", "mix3", "--metric", "CoT",
          "--cached_model", "FINAL_mix3_regression_train_cot", "--save_pred", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--num_samples", "5000",  "--num_epochs", "1"]

exp251 = ["--experiment_name", "FINAL_mix3_regression_train_eval_jitter", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "FINAL_mix3_regression_train", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.0",
          "--dropout_height", "0.9", "--max_c", "1", "--max_s", "0.8"]

exp252 = ["--experiment_name", "FINAL_rand_gen_3_sampling_train", "--world", "rand_gen_3_standability", "--metric", "success", "--sample",
          "--cached_model", "FINAL_rand_gen_3_sampling", "--save_pred", "--voxel_size_z", "2.0", "--num_samples", "5000",
          "--overlap", "0.15", "--epoch_model", "", "--encoder_model", "FINAL_mix3_regression_train"]

exp253 = ["--experiment_name", "FINAL_rand_gen_3_sampling_eval", "--world", "rand_gen_3", "--metric", "success", "--use_poses", "--sample",
          "--cached_model", "FINAL_rand_gen_3_sampling", "--save_pred", "--evaluate_only", "--overlap", "0.15", "--stem", "FINAL", "--epoch_model", "0_finetuned",]

exp254 = ["--experiment_name", "FINAL_rand_gen_3_sampling_eval_graph", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
          "--cached_model", "FINAL_mix3_regression_train", "--save_pred", "--evaluate_only", "--cached_dataset", "rand_gen_3_learned_FINAL",
          ]


# subprocess.run(["python", script_name_dbg, *exp122])  # Rivedi!!!!!!!!!!!
# subprocess.run(["python", script_name_dbg, *exp249])
# subprocess.run(["python", script_name_dbg, *exp250])
# subprocess.run(["python", script_name_dbg, *exp251])
# subprocess.run(["python", script_name_dbg, *exp252])
subprocess.run(["python", script_name_dbg, *exp253])
subprocess.run(["python", script_name_dbg, *exp254])

# subprocess.run(["python", script_name, *exp137])  # Cambia voxel dimensions
# subprocess.run(["python", script_name, *exp138])
# subprocess.run(["python", script_name, *exp139])
# subprocess.run(["python", script_name, *exp140])
# subprocess.run(["python", script_name, *exp138])
# subprocess.run(["python", script_name, *exp139])
# subprocess.run(["python", script_name, *exp140])

# subprocess.run(["python", script_name, *exp141])  # Cambia in sampling repo  # da fare
# subprocess.run(["python", script_name, *exp204])  # Cambia in sampling repo

# subprocess.run(["python", script_name, *exp203])  # CoT
# subprocess.run(["python", script_name, *exp145a])

# subprocess.run(["python", script_name, *exp142])  # Cambia voxel dimensions back (2mt) DONE
# subprocess.run(["python", script_name, *exp143])
# subprocess.run(["python", script_name, *exp144])
# subprocess.run(["python", script_name, *exp145])
# subprocess.run(["python", script_name, *exp146])

# subprocess.run(["python", script_name, *exp147])  # 3mt DONE
# subprocess.run(["python", script_name, *exp148])
# subprocess.run(["python", script_name, *exp149])
# subprocess.run(["python", script_name, *exp150])

# subprocess.run(["python", script_name, *exp151])  # 5mt DONE
# subprocess.run(["python", script_name, *exp152])
# subprocess.run(["python", script_name, *exp153])
# subprocess.run(["python", script_name, *exp154])

# subprocess.run(["python", script_name, *exp155])  # 0.5mt DONE
# subprocess.run(["python", script_name, *exp156])
# subprocess.run(["python", script_name, *exp157])
# subprocess.run(["python", script_name, *exp158])

# subprocess.run(["python", script_name, *exp159])  # noise 0.1 DONE
# subprocess.run(["python", script_name, *exp160])
# subprocess.run(["python", script_name, *exp161])
# subprocess.run(["python", script_name, *exp162])
# subprocess.run(["python", script_name, *exp163])
# subprocess.run(["python", script_name, *exp161])
# subprocess.run(["python", script_name, *exp162])
# subprocess.run(["python", script_name, *exp163])
# subprocess.run(["python", script_name, *exp164])
# subprocess.run(["python", script_name, *exp165])
# subprocess.run(["python", script_name, *exp166])
# subprocess.run(["python", script_name, *exp167])
# subprocess.run(["python", script_name, *exp168])
# subprocess.run(["python", script_name, *exp169])
# subprocess.run(["python", script_name, *exp170])
# subprocess.run(["python", script_name, *exp171])
# subprocess.run(["python", script_name, *exp172])
# subprocess.run(["python", script_name, *exp173])
# subprocess.run(["python", script_name, *exp174])
# subprocess.run(["python", script_name, *exp175])
# subprocess.run(["python", script_name, *exp176])
# subprocess.run(["python", script_name, *exp177])
# subprocess.run(["python", script_name, *exp178])
# subprocess.run(["python", script_name, *exp179])
# subprocess.run(["python", script_name, *exp180])

# subprocess.run(["python", script_name, *exp181])  # noise 0.15 DONE
# subprocess.run(["python", script_name, *exp182])
# subprocess.run(["python", script_name, *exp183])
# subprocess.run(["python", script_name, *exp184])
# subprocess.run(["python", script_name, *exp185])
# subprocess.run(["python", script_name, *exp186])
# subprocess.run(["python", script_name, *exp187])
# subprocess.run(["python", script_name, *exp188])
# subprocess.run(["python", script_name, *exp189])
# subprocess.run(["python", script_name, *exp190])
# subprocess.run(["python", script_name, *exp191])
# subprocess.run(["python", script_name, *exp192])
# subprocess.run(["python", script_name, *exp193])
# subprocess.run(["python", script_name, *exp194])
# subprocess.run(["python", script_name, *exp195])
# subprocess.run(["python", script_name, *exp196])
# subprocess.run(["python", script_name, *exp197])
# subprocess.run(["python", script_name, *exp198])
# subprocess.run(["python", script_name, *exp199])
# subprocess.run(["python", script_name, *exp200])
# subprocess.run(["python", script_name, *exp201])
# subprocess.run(["python", script_name, *exp202])


#### DA FARE ####
# subprocess.run(["python", script_name, *exp137])  # voxel dim 1mt (bis)
# subprocess.run(["python", script_name, *exp138])
# subprocess.run(["python", script_name, *exp139])
# subprocess.run(["python", script_name, *exp140])

# subprocess.run(["python", script_name, *exp113])
