# Learning Based Transmission Optimization

This repository contains code that implements learning based approaches to transmission optimization in a Multiple Input Single Output(MISO) communication system. This was done as part of my final year Dual Degree Project(DDP) at IIT Madras.

I suggest making use of Anaconda 3 for all the software requirements of this project. It can be downloaded from this [link](https://www.anaconda.com/download/). Once it is installed, open *Anaconda Prompt*. Now, you can create a virtual environment within Anaconda where all the software packages required for this project will be installed. You can then run the project scripts within this virtual environment. You should run the following to create a conda virtual environment of the name **my_env**,
```conda
conda create --name my_env --file requirements.txt
```

Once the virtual environment is ready with all the required packages, you can use the following to enter into the virtual environment.
```conda
conda activate my_env
```

This command will take you into the virtual environment, **my_env**. Now, you can run the project scripts as described below.

[Papers/](Papers/) contains some papers that are relevant to the project. The problem setup here is the same as the one used in this [paper](Papers/Conf_Learning_to_Beamform_for_Intelligent_Reflecting_Surface_with_Implicit_Channel_Estimate.pdf). The primary code implementing the dataset generation, model training and baseline calculations are contained in [main.py](main.py). [utils.py](utils.py) contains some support functions that [main.py](main.py) uses. [job.py](job.py) is the python script that needs to be run to dispatch a job. Run `python job.py --help` to view all the available options with which jobs can be dispatched. You can give options appropriate to your needs.

A first job may look something like this,
```cmd
python job.py --generate_user_locations --direc /path_to/main/run-1/ --direc_main /path_to/main/
```

Make sure that the 2 directories(`/path_to/main/run-1/` & `/path_to/main/`) exist before running the above command. For the first job dispatch, you won't have any files relevant to the job in `/path_to/main/run-1/`. So, the `--generate_user_locations` tag needs to be added the first time to generate the user locations. From the second job in the same directory(`/path_to/main/run-1/`), the channels and user locations already saved there from the previous job can be reused. For the second job, the command could be something like,
```cmd
python job.py --import_old_channels --direc /path_to/main/run-1/ --direc_main /path_to/main/
```

This command asks the job to reuse the user locations and channels generated during the first job.
