# Tone Mapping Application on Medical Images

## Installation
1. Download images.zip from [here](https://drive.google.com/file/d/0Bx9NrJbRmDSpMlQ4VGMyQzZDcGc/view?usp=sharing).
1. Create a subdirectory named "_images_" on ToneMapping directory.
1. Unzip the downloaded images in new directory "_images_".
1. Create "_results_" directory inside each version Directory.
1. In each "_results_" directory create the following directories:
	* "_results_gamma_"
	* "_results_log_"
	* "_results_adap_log_"
1. Compile programs:
	1. Sequential: Go to "_Sequential_" directory and run `make`
	1. Parallel: Go to "_Parallel_" directory and run `make`
	1. MPI: Go to "_MPI/build_" directory and run `make`

## Running the Programs
1. Go to the version Directory that you wish to run, e.g. Parallel.
1. Run the slurm command `sbatch tonemap.sh`
	* For the MPI version, due to a problem with SLURM, run the script "*tonemap_V2.sh*". `chmod +x tonemap_V2.sh` to give execution permissions and then `./tonemap_V2.sh > res_tonemapping_mpi.md`
1. Wait until the program finish, the images will be saved on the previously created "_results_" directory. And a "*res_tonemapping_\<version\>.md*", e.g. res_tonemapping_parallel.md