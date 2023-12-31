# LSF QUERIES
This file explains how to use the HPC server using LSF.

## SETUP
1. Log-in to the server using (when connected to eduroam or VPN):
```bash
ssh studentID@login.hpc.dtu.dk 
```
Replace studentID with your student number, e.g., s183700. When prompted for a password, use your DTU password.

2. IMPORTANT! You are in the login node - move to a node to run queries from using:
```bash
linuxsh
``` 
3. Create a folder for your scripts (if u please) using:
```bash
mkdir FOLDER_NAME
```     
4. Load a python module, e.g.:
```bash
module load python/3.10.12
```  
Use ```module avail``` to see all available modules and ```module list``` to see your modules.
        
5. Install schnetpack:
```bash
pip install schnetpack
pip install tensorboard
```  

## CREATING A JOB
Use the **job_template.sh** located in this folder. Modify using the following steps:
1. Specify the queue in line 4 (default is ```gpuv100``` now). Available resources can be found running ```nodestat -F hpc```.
2. Update the jobname in line 6.
3. Set wall time (default is 15 minutes, so good to remember to update).
4. Update module load in line 31 with your module.
5. Update the script in line 32 with your script.

More settings, like e.g. your mail address (defaul job updates are sent to your student email), can be set in the file as well.

## RUNNING A JOB
1. Create a folder on your local computer with the python script and the .sh script.
2. Copy the folder to the HPC using:
```bash
scp -r PATH_TO_FOLDER studentID@login.hpc.dtu.dk:~/PATH_TO_FOLDER 
``` 
E.g., ```scp -r ~/test_folder s183700@login.hpc.dtu.dk:~/scripts```. To copy a single file, use: ```scp test_job.sh studentID@login.hpc.dtu.dk:~/folderName``` from the folder where the file is located (or add the path).

3. Navigate to the folder on the HPC  and run the .sh script using:
```bash
bsub < SCRIPT_NAME.sh
```
4. Test the status of your queries using:
```bash
bstat
```
5. Copy results to your pc using (run from local machine):
```bash
scp studentID@login.hpc.dtu.dk:~/HPC_PATH LOCAL_PATH
```
E.g., ```scp s183700@login.hpc.dtu.dk:~/scripts/test/test_job.sh ~/Downloads``` or for a whole folder ```scp -r s183700@login.hpc.dtu.dk:~/scripts/training_scripts/homo_training/qm9tut ~/Downloads```.

### GOOD-2-KNOWS:
* Remove non-empty folder using:  
```bash
rm -r -I FOLDER_NAME
```
* Stop a job using:
```bash
bkill <job_id>
```
