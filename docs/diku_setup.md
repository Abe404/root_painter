### DIKU RootPainter Setup

This is is a guide for getting started with the RootPainter software
using the diku servers. RootPainter is described in [this paper.](https://www.biorxiv.org/content/10.1101/2020.04.16.044461v1)

1. You will need to be able to ssh into slurm. If you don't have one already then add an entry for slurm in your ssh config file. Add the following to ~/.ssh/config and replace KUID with your own KU ID.
---
```
Host slurm
  Hostname a00552
  User KUID
  ProxyCommand ssh -q -W %h:%p KUID@ssh-diku-image.science.ku.dk
```
---
There is more information about working with slurm in the [Slurm Wiki](http://image.diku.dk/mediawiki/index.php/Slurm_Cluster).

2. SSH into the server to set up the server component of RootPainter.
```
ssh slurm
```

3. Clone the RootPainter code from the repository and then cd into the trainer directory (the server component).
```
git clone --branch 0.2.3 https://github.com/Abe404/root_painter
cd root_painter/trainer
```

4. To avoid alterating any global packages. I suggest using a virtual environment. Create a virtual environment and activate it.
```
python -m venv env
source ./env/bin/activate
```

5. Install dependencies in the virtual environment. (takes ~3 minutes)
```
pip install -r requirements.txt
```

6. Run root painter to create the sync directory.
```
python main.py
```
You will be prompted to input a location for the sync directory. This is the folder where files are shared between the client and server. I will use ~/root_painter_sync
RootPainter will then create some folders inside ~/root_painter_sync

7. Create a slurm job.
Create a file named job.sh and insert the following. Modify the details based on your preferences.
---
```
#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=20000M
# we run on the gpu partition and we allocate 1 titanrtx gpu
#SBATCH -p gpu --gres=gpu:titanrtx:1
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=3:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
python main.py
```
---

8. Run the slurm job.
```
sbatch job.sh
```

9. To mount the sync directory from your local machine you will need to install sshfs locally (SSH Filesystem client).

Debian / Ubuntu:
```
sudo apt-get install sshfs
```
OSX:
```
brew cask install osxfuse
```
Windows:
[Digital ocean has a guide](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh)

10. Create the directory and mount the drive locally using sshfs. Replace KUID with your own KU ID.
```
mkdir ~/Desktop/root_painter_sync
sshfs -o allow_other,default_permissions KUID@slurm:/home/KUID/root_painter_sync ~/Desktop/root_painter_sync
```
You should now be able to see the folders created by RootPainter (datasets, instructions and projects) inside ~/root_painter_sync on your local machine 
See [lung tutorial](cxr_lung_tutorial.md) for an example of how to use RootPainter to train a model.

