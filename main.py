# Importing relevant libraries
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torchinfo import summary
from scipy.linalg import fractional_matrix_power
import time
import sys
import logging
from utils import *

#####################################################################
#####################################################################
#####################################################################
def main(**kwargs):
    """
    Function to make the dataset and train the chosen model
    Args:
        kwargs    Dictionary of arguments
    """
    ######################################################################
    # Boolean for debugging
    DEBUG = False

    # Logging configuration
    logging.basicConfig(
        filename=kwargs['direc'] + "Logs.log",
        filemode="a",
        format="%(asctime)s.%(msecs)03d :  %(message)s",
        datefmt="%I:%M:%S",
        level=logging.DEBUG
    )
    logging.getLogger("matplotlib.font_manager").disabled = True

    def logit(message):
        logging.info("%s", message)

    #####################################################################
    logit('')
    logit('')
    logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    logit('Directory ------------------------------ %s' % kwargs['direc'])
    logit('Directory, Main ------------------------ %s' % kwargs['direc_main'])
    logit('Model Type ----------------------------- %s' % kwargs['which_model'])
    logit('Number Of Base Stations ---------------- %d' % kwargs['num_bs'])
    logit('Number Of Users ------------------------ %d' % kwargs['num_users'])
    logit('Number Of Antennas per BS -------------- %d' % kwargs['num_antennas'])
    logit('Number Of IRS Reflective Elements ------ %d' % kwargs['num_reflectors'])
    logit('Total Uplink Power (dBm) --------------- %.3f' % kwargs['up_power'])
    logit('Total Downlink Power (dBm) ------------- %.3f' % kwargs['down_power'])
    logit('Number Of Train Samples ---------------- %d' % kwargs['num_train_samples'])
    logit('Number Of Channels In Train ------------ %d' % (kwargs['num_train_samples']//kwargs['batch_size']))
    logit('Number Of Test Samples ----------------- %d' % kwargs['num_test_samples'])
    logit('Number Of Channels In Test ------------- %d' % (kwargs['num_test_samples']//kwargs['batch_size']))
    logit('Batch Size ----------------------------- %d' % kwargs['batch_size'])
    logit('Pilot Length --------------------------- %d' % kwargs['L'])
    logit('Import Old Datasets -------------------- %s' % kwargs['import_old_datasets'])
    logit('Import Old Channels -------------------- %s' % kwargs['import_old_channels'])
    logit('Generate New User Locations ------------ %s' % kwargs['generate_user_locations'])
    logit('Number Of Epochs ----------------------- %d' % kwargs['num_epochs'])
    logit('Learning Rate -------------------------- %f' % kwargs['learning_rate'])
    logit('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    #####################################################################
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logit('Using %s' % device)

    # Directory where we save data, plots etc.
    direc = kwargs['direc']
    direc_main = kwargs['direc_main']

    # Chosen model type
    which_model = kwargs['which_model']

    # Pilot length
    L = kwargs['L']

    ######################################################################
    ### Some basic global variables that will be useful
    # Number of base stations (B)
    NUM_BS = kwargs['num_bs']
    # Number of users per base station (K)
    NUM_USERS = kwargs['num_users']
    # Number of antennas in the base station (M)
    NUM_ANTENNAS = kwargs['num_antennas']
    # Number of reflective elements in the intelligent reflector (N)
    NUM_REFLECTORS = kwargs['num_reflectors']

    ######################################################################
    # Standard deviation of distribution from which we sample the symbols to be transmitted
    SIGMA_SYMBOL = 1
    # Standard deviation of noise while observing output at any user
    # SIGMA0 = (3.162278e-12)**0.5
    SIGMA0 = (1.8e-12)**0.5         # Matching value from WSR maximization paper
    # Standard deviation of noise vectors while generating pilots
    SIGMA1 = (1e-13)**0.5

    ######################################################################
    # Path for storing or accessing channel from BS to IRS
    PATH_G = direc + "G.npy"
    # Path for storing or accessing channel from BS to Users
    PATH_D = direc + "D.npy"
    # Path for storing or accessing channel from IRS to Users
    PATH_R = direc + "R.npy"

    # Path for storing user locations
    PATH_USER_LOCATIONS = direc + "user_locations.npy"
    # Path for storing the distance of users from the BS
    PATH_DIST_BS = direc + "dist_BS.npy"
    # Path for storing the distance of users from the IRS
    PATH_DIST_IRS = direc + "dist_IRS.npy"

    ######################################################################
    # Model Name
    MODEL_NAME = which_model
    # Number of test samples
    NUM_TEST_SAMPLES = kwargs['num_test_samples'] # 1k
    # Number of train samples
    NUM_TRAIN_SAMPLES = kwargs['num_train_samples'] # 10k
    # Batch size
    BATCH_SIZE = kwargs['batch_size'] # 25

    # Number of epochs
    NUM_EPOCHS = kwargs['num_epochs']
    # Learning Rate
    LEARNING_RATE = kwargs['learning_rate']
    # Boolean to indicate whether we want to use ReduceLROnPlateau
    USE_LR_PLATEAU = False

    # Whether we want to use multi-channel train
    # If True, we will use different channels for each batch
    # Note: Ensure that the test and train set size is a multiple of batch size
    MULTI_CH_TRAIN = True
    assert NUM_TRAIN_SAMPLES % BATCH_SIZE == 0
    assert NUM_TEST_SAMPLES % BATCH_SIZE == 0

    # Number of different channels we will use for training
    NUM_CHANNELS = NUM_TRAIN_SAMPLES // BATCH_SIZE if MULTI_CH_TRAIN else 1
    # Number of different channels we will use for testing
    NUM_TEST_CHANNELS = NUM_TEST_SAMPLES // BATCH_SIZE if MULTI_CH_TRAIN else 1
    # Whether we want to use new channels for testing
    NEW_TEST_CHANNELS = True
    if DEBUG:
        logit(MULTI_CH_TRAIN)
        logit(NUM_CHANNELS)
        logit(NUM_TEST_CHANNELS)
        logit(NEW_TEST_CHANNELS)

    # Model save path
    MODEL_SAVE_PATH = direc + MODEL_NAME + f'_PL_{L}.pth'
    # Model Checkpoint save path
    CHKPT_SAVE_PATH = direc + MODEL_NAME + f'_PL_{L}.pt'

    # Train dataset path
    TRAIN_DATASET_PATH = direc + MODEL_NAME + f"_train_dataset_PL_{L}.npy"
    # Train dataset path
    TEST_DATASET_PATH = direc + MODEL_NAME + f"_test_dataset_PL_{L}.npy"

    ######################################################################
    # Boolean to indicate whether we want to import old datasets
    IMPORT_OLD_DATASETS = kwargs['import_old_datasets']

    # Boolean to indicate whether we want to import old channels
    IMPORT_OLD_CHANNELS = kwargs['import_old_channels']

    # Boolean to choose if we want to generate user locations
    GENERATE_USER_LOCATIONS = kwargs['generate_user_locations']

    ######################################################################
    # Method used to generate channels
    CHANNEL_GENERATION_METHOD = 'custom1'

    # Location of the Base Station
    LOC_BS = 0 + 0j
    # Location of the Intelligent Reflector
    LOC_IRS = 150 + 0j
    # Distance between BS and IRS
    DIST_BS_IRS = np.abs(LOC_BS - LOC_IRS).astype(np.float32)

    # Radius of the circle within which all users are found
    RADIUS_USERS = 1.0        # 10
    # Offset around which all users are found
    OFFSET_USERS = 150 + 30j
    # Location of the users
    if GENERATE_USER_LOCATIONS:
        LOC_USERS = np.zeros(NUM_USERS, dtype=np.complex64)
        for index in range(NUM_USERS):
            while True:
                real_val = 2*np.random.rand(1) - 1
                complex_val = 2*np.random.rand(1) - 1
                if (real_val**2 + complex_val**2) <= 1:
                    LOC_USERS[index] = OFFSET_USERS + RADIUS_USERS*(real_val + complex_val*1j)
                    break
        # Saving the generated user locations
        np.save(PATH_USER_LOCATIONS, LOC_USERS)

        # Calculating the distances between the users and IRS/BS
        DIST_BS = np.abs(LOC_USERS - LOC_BS).astype(np.float32)
        DIST_IRS = np.abs(LOC_USERS - LOC_IRS).astype(np.float32)
        np.save(PATH_DIST_BS, DIST_BS)
        np.save(PATH_DIST_IRS, DIST_IRS)
    else:
        LOC_USERS = np.load(PATH_USER_LOCATIONS)
        DIST_BS = np.load(PATH_DIST_BS)
        DIST_IRS = np.load(PATH_DIST_IRS)

    if DEBUG:
        assert len(LOC_USERS) == NUM_USERS
        assert LOC_USERS.dtype == np.complex64
        logit(DIST_BS[:5])
        logit(DIST_IRS[:5])

    ######################################################################
    # Total downlink power constraint
    TOTAL_POWER_CONSTRAINT_dBm = kwargs['down_power']
    TOTAL_POWER_CONSTRAINT = 1e-3*(10**(TOTAL_POWER_CONSTRAINT_dBm/10))

    # Total uplink power constraint
    TOTAL_UP_POWER_CONSTRAINT_dBm = kwargs['up_power']     # 30
    TOTAL_UP_POWER_CONSTRAINT = 1e-3*(10**(TOTAL_UP_POWER_CONSTRAINT_dBm/10))

    ######################################################################
    ######################################################################
    ### Making a parameters dictionary that we will pass into functions
    PARAMS = {
        "num_bs": NUM_BS,
        "num_users": NUM_USERS,
        "num_antennas": NUM_ANTENNAS,
        "num_reflectors": NUM_REFLECTORS,
        "path_G": PATH_G,
        "path_D": PATH_D,
        "path_R": PATH_R,
        "model_name": MODEL_NAME,
        "model_save_path": MODEL_SAVE_PATH,
        "chkpt_save_path": CHKPT_SAVE_PATH,
        "import_old_channels": IMPORT_OLD_CHANNELS,
        "channel_generation_method": CHANNEL_GENERATION_METHOD,
        "sigma_symbol": SIGMA_SYMBOL,
        "sigma0": SIGMA0,
        "sigma1": SIGMA1,
        "pilot_length": L,
        "loc_BS": LOC_BS,
        "loc_IRS": LOC_IRS,
        "radius_users": RADIUS_USERS,
        "offset_users": OFFSET_USERS,
        "loc_users": LOC_USERS,
        "dist_BS": DIST_BS,
        "dist_IRS": DIST_IRS,
        "dist_BS_IRS": DIST_BS_IRS,
        "total_power_constraint": TOTAL_POWER_CONSTRAINT,
        "total_up_power_constraint": TOTAL_UP_POWER_CONSTRAINT,
        "num_test_samples": NUM_TEST_SAMPLES,
        "num_train_samples": NUM_TRAIN_SAMPLES,
        "batch_size": BATCH_SIZE,
        "train_dataset_path": TRAIN_DATASET_PATH,
        "test_dataset_path": TEST_DATASET_PATH,
        "num_epochs": NUM_EPOCHS,
        "lrate": LEARNING_RATE,
        "use_LR_Plateau": USE_LR_PLATEAU,
        "multi_ch_train": MULTI_CH_TRAIN,
        "num_channels": NUM_CHANNELS,
        "num_test_channels": NUM_TEST_CHANNELS,
        "new_test_channels": NEW_TEST_CHANNELS
    }

    ######################################################################
    # Getting the channels
    channels = get_channels(params=PARAMS, import_old=IMPORT_OLD_CHANNELS, choice=CHANNEL_GENERATION_METHOD)
    PARAMS['channels'] = channels

    ######################################################################
    def get_dataset_MLP(choice, make_new, params=PARAMS):
        """
        Function used to generate the dataset for training and testing
        Arguments:
        choice          Choice as to whether we want to make test or train dataset
        make_new        Boolean to indicate whether we want to create a new dataset or just load an old one
        params          All the relevant parameters

        Returns:
        Dictionary containing {
            "dataset" - Dataset,
            "size" - Number of samples in the dataset
        }
        """
        if make_new:
            if not params['multi_ch_train']:
                # Channels
                G = params['channels']['G'].astype(np.complex64)
                D = params['channels']['D'].astype(np.complex64)
                R = params['channels']['R'].astype(np.complex64)

            if choice == 'train':
                dataset = np.zeros((params['num_train_samples'], 2*params['num_antennas']*params['pilot_length']), dtype = np.float32)
                for idx in range(params['num_train_samples']):
                    if params['multi_ch_train']:
                        # Channels
                        G = params['channels']['G'][idx//params['batch_size']].astype(np.complex64)
                        D = params['channels']['D'][idx//params['batch_size']].astype(np.complex64)
                        R = params['channels']['R'][idx//params['batch_size']].astype(np.complex64)

                    # Symbols we will be sending
                    up_vals = np.eye(params['num_users'], dtype=np.complex64)*np.sqrt(params['num_users']*params['total_up_power_constraint'])
                    symbols = np.tile(up_vals, params['pilot_length']//params['num_users'])

                    # Noise
                    noise = params['sigma1']*(generate_complex_gaussian_array((params['num_antennas'], params['pilot_length'])))

                    # Make sure pilot length is a mulitple of number of users
                    assert params['pilot_length']%params['num_users'] == 0
                    
                    # Randomly sampling angles for IRS phase
                    angles = 2*np.pi*np.random.rand(params['num_reflectors'], params['pilot_length']//params['num_users'])
                    v = np.exp(1j*angles).astype(np.complex64)
                    v = np.repeat(v, params['num_users'], axis=1)
                    assert v.shape == (params['num_reflectors'], params['pilot_length'])

                    # Forming the pilots
                    pilots = np.zeros((params['num_antennas'], params['pilot_length']), dtype = np.complex64)
                    for i in range(params['pilot_length']):
                        pilots[:, i:i+1] = (D + G@np.diag(v[:, i])@R)@symbols[:, i:i+1] + noise[:, i:i+1]

                    # Editing `dataset` to include this data sample
                    dataset[idx] = np.concatenate((pilots.flatten('F').real, pilots.flatten('F').imag))
                # Saving the dataset
                np.save(params['train_dataset_path'], dataset)

            elif choice == 'test':
                dataset = np.zeros((params['num_test_samples'], 2*params['num_antennas']*params['pilot_length']), dtype = np.float32)
                for idx in range(params['num_test_samples']):
                    if params['multi_ch_train']:
                        if params['new_test_channels']:
                            # Channels
                            G = params['channels']['G'][params['num_channels'] + (idx//params['batch_size'])].astype(np.complex64)
                            D = params['channels']['D'][params['num_channels'] + (idx//params['batch_size'])].astype(np.complex64)
                            R = params['channels']['R'][params['num_channels'] + (idx//params['batch_size'])].astype(np.complex64)
                        else:
                            # Channels
                            G = params['channels']['G'][idx//params['batch_size']].astype(np.complex64)
                            D = params['channels']['D'][idx//params['batch_size']].astype(np.complex64)
                            R = params['channels']['R'][idx//params['batch_size']].astype(np.complex64)


                    # Symbols we will be sending
                    up_vals = np.eye(params['num_users'], dtype=np.complex64)*np.sqrt(params['num_users']*params['total_up_power_constraint'])
                    symbols = np.tile(up_vals, params['pilot_length']//params['num_users'])

                    # Noise
                    noise = params['sigma1']*(generate_complex_gaussian_array((params['num_antennas'], params['pilot_length'])))

                    # Randomly sampling angles for IRS phase
                    angles = 2*np.pi*np.random.rand(params['num_reflectors'], params['pilot_length']//params['num_users'])
                    v = np.exp(1j*angles).astype(np.complex64)
                    v = np.repeat(v, params['num_users'], axis=1)
                    assert v.shape == (params['num_reflectors'], params['pilot_length'])

                    # Forming the pilots
                    pilots = np.zeros((params['num_antennas'], params['pilot_length']), dtype = np.complex64)
                    for i in range(params['pilot_length']):
                        pilots[:, i:i+1] = (D + G@np.diag(v[:, i])@R)@symbols[:, i:i+1] + noise[:, i:i+1]

                    # Editing `dataset` to include this data sample
                    dataset[idx] = np.concatenate((pilots.flatten('F').real, pilots.flatten('F').imag))
                # Saving the dataset
                np.save(params['test_dataset_path'], dataset)

            else:
                sys.exit('$$$$$ make_dataset(): Invalid `choice` for dataset. Needs to be "train" or "test".')
        else:
            if choice == 'train':
                dataset = np.load(params['train_dataset_path']).astype(np.float32)
            elif choice == 'test':
                dataset = np.load(params['test_dataset_path']).astype(np.float32)
            else:
                sys.exit('$$$$$ make_dataset(): Invalid `choice` for dataset. Needs to be "train" or "test".')

        return {
            "dataset": dataset,
            "size": dataset.shape[0]
        }

    def get_dataset_RNN(choice, make_new, params=PARAMS):
        """
        Function used to generate the dataset for training and testing
        Arguments:
        choice          Choice as to whether we want to make test or train dataset
        make_new        Boolean to indicate whether we want to create a new dataset or just load an old one
        params          All the relevant parameters

        Returns:
        Dictionary containing {
            "dataset" - Dataset,
            "size" - Number of samples in the dataset
        }
        """
        if make_new:
            if not params['multi_ch_train']:
                # Channels
                G = params['channels']['G'].astype(np.complex64)
                D = params['channels']['D'].astype(np.complex64)
                R = params['channels']['R'].astype(np.complex64)

            if choice == 'train':
                dataset = np.zeros((params['num_train_samples'], params['pilot_length'], 2*params['num_antennas']), dtype = np.float32)
                for idx in range(params['num_train_samples']):
                    if params['multi_ch_train']:
                        # Channels
                        G = params['channels']['G'][idx//params['batch_size']].astype(np.complex64)
                        D = params['channels']['D'][idx//params['batch_size']].astype(np.complex64)
                        R = params['channels']['R'][idx//params['batch_size']].astype(np.complex64)

                    # Symbols we will be sending
                    up_vals = np.eye(params['num_users'], dtype=np.complex64)*np.sqrt(params['num_users']*params['total_up_power_constraint'])
                    symbols = np.tile(up_vals, params['pilot_length']//params['num_users'])

                    # Noise
                    noise = params['sigma1']*(generate_complex_gaussian_array((params['num_antennas'], params['pilot_length'])))

                    # Make sure pilot length is a mulitple of number of users
                    assert params['pilot_length']%params['num_users'] == 0
                    
                    # Randomly sampling angles for IRS phase
                    angles = 2*np.pi*np.random.rand(params['num_reflectors'], params['pilot_length']//params['num_users'])
                    v = np.exp(1j*angles).astype(np.complex64)
                    v = np.repeat(v, params['num_users'], axis=1)
                    assert v.shape == (params['num_reflectors'], params['pilot_length'])

                    # Forming the pilots
                    pilots = np.zeros((params['num_antennas'], params['pilot_length']), dtype = np.complex64)
                    for i in range(params['pilot_length']):
                        pilots[:, i:i+1] = (D + G@np.diag(v[:, i])@R)@symbols[:, i:i+1] + noise[:, i:i+1]

                    # Editing `dataset` to include this data sample
                    pilots = pilots.T
                    dataset[idx] = np.concatenate((pilots.real, pilots.imag), axis=1)
                # Saving the dataset
                np.save(params['train_dataset_path'], dataset)

            elif choice == 'test':
                dataset = np.zeros((params['num_test_samples'], params['pilot_length'], 2*params['num_antennas']), dtype = np.float32)
                for idx in range(params['num_test_samples']):
                    if params['multi_ch_train']:
                        if params['new_test_channels']:
                            # Channels
                            G = params['channels']['G'][params['num_channels'] + (idx//params['batch_size'])].astype(np.complex64)
                            D = params['channels']['D'][params['num_channels'] + (idx//params['batch_size'])].astype(np.complex64)
                            R = params['channels']['R'][params['num_channels'] + (idx//params['batch_size'])].astype(np.complex64)
                        else:
                            # Channels
                            G = params['channels']['G'][idx//params['batch_size']].astype(np.complex64)
                            D = params['channels']['D'][idx//params['batch_size']].astype(np.complex64)
                            R = params['channels']['R'][idx//params['batch_size']].astype(np.complex64)

                    # Symbols we will be sending
                    up_vals = np.eye(params['num_users'], dtype=np.complex64)*np.sqrt(params['num_users']*params['total_up_power_constraint'])
                    symbols = np.tile(up_vals, params['pilot_length']//params['num_users'])

                    # Noise
                    noise = params['sigma1']*(generate_complex_gaussian_array((params['num_antennas'], params['pilot_length'])))

                    # Randomly sampling angles for IRS phase
                    angles = 2*np.pi*np.random.rand(params['num_reflectors'], params['pilot_length']//params['num_users'])
                    v = np.exp(1j*angles).astype(np.complex64)
                    v = np.repeat(v, params['num_users'], axis=1)
                    assert v.shape == (params['num_reflectors'], params['pilot_length'])

                    # Forming the pilots
                    pilots = np.zeros((params['num_antennas'], params['pilot_length']), dtype = np.complex64)
                    for i in range(params['pilot_length']):
                        pilots[:, i:i+1] = (D + G@np.diag(v[:, i])@R)@symbols[:, i:i+1] + noise[:, i:i+1]

                    # Editing `dataset` to include this data sample
                    pilots = pilots.T
                    dataset[idx] = np.concatenate((pilots.real, pilots.imag), axis=1)
                # Saving the dataset
                np.save(params['test_dataset_path'], dataset)

            else:
                sys.exit('$$$$$ make_dataset(): Invalid `choice` for dataset. Needs to be "train" or "test".')
        else:
            if choice == 'train':
                dataset = np.load(params['train_dataset_path']).astype(np.float32)
            elif choice == 'test':
                dataset = np.load(params['test_dataset_path']).astype(np.float32)
            else:
                sys.exit('$$$$$ make_dataset(): Invalid `choice` for dataset. Needs to be "train" or "test".')

        return {
            "dataset": dataset,
            "size": dataset.shape[0]
        }

    MAKE_NEW_DATASETS = not IMPORT_OLD_DATASETS

    # Train dataset
    if which_model == 'MLP':
        train_data = torch.tensor(get_dataset_MLP(choice='train', make_new=MAKE_NEW_DATASETS)['dataset'], device=device)
        logit('Train dataset size: %s' % str(train_data.size()))
    else:
        train_data = torch.tensor(get_dataset_RNN(choice='train', make_new=MAKE_NEW_DATASETS)['dataset'], device=device)
        logit('Train dataset size: %s' % str(train_data.size()))
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=False)

    # Test dataset
    if which_model == 'MLP':
        test_data = torch.tensor(get_dataset_MLP(choice='test', make_new=MAKE_NEW_DATASETS)['dataset'], device=device)
        logit('Test dataset size: %s' % str(test_data.size()))
    else:
        test_data = torch.tensor(get_dataset_RNN(choice='test', make_new=MAKE_NEW_DATASETS)['dataset'], device=device)
        logit('Test dataset size: %s' % str(test_data.size()))
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=PARAMS['batch_size'], shuffle=False)

    loaders = {
        'train' : train_loader,    
        'test'  : test_loader
    }

    ######################################################################
    # Parameters relevant to the model
    if which_model == 'WMMSE':
        MODEL_PARAMS = {
            'type': which_model,
            'RNN_n_neurons': 20,
            'RNN_n_layers': 1,
            'RNN_n_inputs': 2*PARAMS['num_antennas'],
            'RNN_bi_directional': True,
            'RNN_non_linearity': 'tanh',
            'num_its_WMMSE': 4,
            'num_its_PGD': 4,
            'n_outputs_BS': 2*PARAMS['num_antennas']*PARAMS['num_users'],
            'n_outputs_IRS': 2*PARAMS['num_reflectors']
        }
    elif which_model in ['vanilla', 'LSTM', 'GRU']:
        MODEL_PARAMS = {
            'type': which_model,
            'n_neurons': 20,
            'n_layers': 1,
            'n_inputs': 2*PARAMS['num_antennas'],
            'bi_directional': True,
            'non_linearity': 'tanh',
            'n_outputs_BS': 2*PARAMS['num_antennas']*PARAMS['num_users'],
            'n_outputs_IRS': 2*PARAMS['num_reflectors']
        }
    elif which_model == 'MLP':
        MODEL_PARAMS = {
            'type': which_model,
            'n_neurons': 100
        }
    else:
        sys.exit('$$$$$ main(): Invalid `which_model` to create `MODEL_PARAMS`.')

    # Creating the model
    if MODEL_PARAMS['type'] == 'WMMSE':
        model = UnfoldedWMMSE(model_params = MODEL_PARAMS, params = PARAMS)
        model.to(device)
        summary(model, (1, PARAMS['pilot_length'], 2*PARAMS['num_antennas']))
    elif MODEL_PARAMS['type'] == 'vanilla':
        model = VanillaRNN(model_params = MODEL_PARAMS, params = PARAMS)
        model.to(device)
        summary(model, (1, PARAMS['pilot_length'], 2*PARAMS['num_antennas']))
    elif MODEL_PARAMS['type'] == 'LSTM':
        model = LSTM(model_params = MODEL_PARAMS, params = PARAMS)
        model.to(device)
        summary(model, (1, PARAMS['pilot_length'], 2*PARAMS['num_antennas']))
    elif MODEL_PARAMS['type'] == 'GRU':
        model = GRU(model_params = MODEL_PARAMS, params = PARAMS)
        model.to(device)
        summary(model, (1, PARAMS['pilot_length'], 2*PARAMS['num_antennas']))
    elif MODEL_PARAMS['type'] == 'MLP':
        model = BeamFormer(model_params = MODEL_PARAMS, params = PARAMS)
        model.to(device)
        summary(model, (1, 2*PARAMS['num_antennas']*PARAMS['pilot_length']))

    # Loading the model if already trained
    # model = torch.load(PARAMS['model_save_path']).to(device)

    ######################################################################
    # Training, Test loops & Loss function
    def train_loop(loaders, model, loss_fn, optimizer, interval, params=PARAMS):
        """
        Function to train the model and log required information
        Arguments:
        loaders                 dict containing DataLoader objects for the data
        model                   The neural network we want to train
        loss_fn                 The loss function we are trying to minimize
        optimizer               Optimizer that we will use
        interval                Interval between logging of loss & calculating test metrics [default: 40]
        params                  Parameters relevant to the run  [Default: PARAMS]

        Returns:    Dict containing lists of training losses and test losses & accuracies.
        """
        dataloader = loaders['train']
        size = len(dataloader.dataset)
        losses = []
        losses_test = []
        global losses_test_min
        global BATCH_ID

        for batch, X in enumerate(dataloader):
            BATCH_ID = batch
            # Compute prediction and loss
            pred = model(X[0])['out']
            loss = loss_fn(pred)
            losses.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % interval == 0:
                loss, current = loss.item(), batch * len(X[0])
                logit('')
                logit("Loss: %.5f, Sum Rate: %.5f  [%s/%s]" % (loss, -loss, current, size))
                temp1 = test_loop(loaders, model, loss_fn)
                losses_test.append(temp1['loss'])

                # Saving checkpoint if the model achieved here is the best performer on Test so far
                if temp1['loss'] < losses_test_min:
                    # Updating losses_test_min
                    losses_test_min = temp1['loss']
                    logit("Saving Model Checkpoint -------------------- Test Loss: %.5f" % (temp1['loss']))
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': temp1['loss'],
                    }, params['chkpt_save_path'])
            
        return {
            'losses': losses,
            'losses_test': losses_test
        }

    def test_loop(loaders, model, loss_fn, params=PARAMS):
        '''
        Function to calculate loss and accuracy of the model on the test set.
        Arguments:
        loaders                 dict containing DataLoader objects for the data
        model                   The neural network we want to test
        loss_fn                 The loss function we are trying to minimize in training
        Returns:    Dict containing loss and accuracy of the model on the test dataset
        '''
        dataloader = loaders['test']
        num_batches = len(dataloader)
        test_loss = 0
        global BATCH_ID

        with torch.no_grad():
            for batch, X in enumerate(dataloader):
                if params['new_test_channels']:
                    BATCH_ID = batch + params['num_channels']
                else:
                    BATCH_ID = batch
                pred = model(X[0])['out']
                test_loss += loss_fn(pred).item()
        test_loss /= num_batches

        # Printing relevant metrics
        logit("Test Metrics - Avg loss: %.5f, Sum Rate: %.5f" % (test_loss, -test_loss))

        return {
            'loss': test_loss
        }

    def sum_rate_loss(output, params=PARAMS):
        """
        Function to calculate the loss. We will try to minimize this while training the network
        Arguments:
        output              Output returned by the network
        params              All parameters relevant to our run

        Returns
        loss                The calculated loss (Negative sum rate)
        """
        if not params['multi_ch_train']:
            # Getting the channels locally for simplicity of notation
            G = torch.tensor(params['channels']['G'], device=device)
            D = torch.tensor(params['channels']['D'], device=device)
            R = torch.tensor(params['channels']['R'], device=device)
        else:
            # Getting the channels locally for simplicity of notation
            G = torch.tensor(params['channels']['G'][BATCH_ID], device=device)
            D = torch.tensor(params['channels']['D'][BATCH_ID], device=device)
            R = torch.tensor(params['channels']['R'][BATCH_ID], device=device)

        # Number of samples in this batch
        num_samples = output.size(0)

        # Getting the raw representation of the beamformers and the IRS coefficients from network output
        const1 = params['num_antennas']*params['num_users']
        const2 = params['num_reflectors']
        BS_raw = output[:, :2*const1]
        IRS_raw = output[:, -2*const2:]

        # Building the actual complex beamformers and IRS coefficients
        beamformers = BS_raw[:, :const1] + BS_raw[:, -const1:]*1j
        IRS_coeff = IRS_raw[:, :const2] + IRS_raw[:, -const2:]*1j

        # Reshaping the beamformers tensor into a rectangular array whose columns will be beamformers for each user
        beamformers = torch.reshape(beamformers, (-1, params['num_users'], params['num_antennas']))
        beamformers = torch.transpose(beamformers, 1, 2)

        # Initializing variable to hold sum rate for each sample
        sum_rates = torch.zeros(num_samples)

        # Calculating loss/sum-rate for each sample in this batch
        for idx in range(num_samples):
            # Beamformers and IRS coefficients corresponding to this sample
            bf = beamformers[idx]
            irs = IRS_coeff[idx]

            # Reshaping to column vector
            irs = torch.reshape(irs, (-1, 1))
            diagv = torch.diag(torch.squeeze(irs))
            # Multiplying G and diag(v)
            tmp1 = torch.mm(G, diagv)

            # Variable the hold all the rates
            rates = torch.zeros(params['num_users'])

            # Finding the rate for each user
            for i in range(params['num_users']):
                d = D[:, i:i+1]
                r = R[:, i:i+1]
                tmp2 = (d + torch.mm(tmp1, r)).T
                temp = torch.square(torch.abs(torch.squeeze(torch.mm(tmp2, bf))))
                temp_sum = torch.sum(temp)
                # Calculating the rate for the ith user
                rates[i] = torch.log(1 + (temp[i])/(temp_sum - temp[i] + params['sigma0']**2))

            # Finding sum rate
            sum_rates[idx] = torch.sum(rates)

        # Since we want to minimize the loss, we set loss to be - mean sum rate
        loss = -torch.mean(sum_rates)

        return loss

    def sum_rate_loss_conj(output, params=PARAMS):
        """
        Function to calculate the loss. We will try to minimize this while training the network
        Arguments:
        output              Output returned by the network
        params              All parameters relevant to our run

        Returns
        loss                The calculated loss (Negative sum rate)
        """
        if not params['multi_ch_train']:
            # Getting the channels locally for simplicity of notation
            G = torch.tensor(params['channels']['G'], device=device)
            D = torch.tensor(params['channels']['D'], device=device)
            R = torch.tensor(params['channels']['R'], device=device)
        else:
            # Getting the channels locally for simplicity of notation
            G = torch.tensor(params['channels']['G'][BATCH_ID], device=device)
            D = torch.tensor(params['channels']['D'][BATCH_ID], device=device)
            R = torch.tensor(params['channels']['R'][BATCH_ID], device=device)

        # Number of samples in this batch
        num_samples = output.size(0)

        # Getting the raw representation of the beamformers and the IRS coefficients from network output
        const1 = params['num_antennas']*params['num_users']
        const2 = params['num_reflectors']
        BS_raw = output[:, :2*const1]
        IRS_raw = output[:, -2*const2:]

        # Building the actual complex beamformers and IRS coefficients
        beamformers = BS_raw[:, :const1] + BS_raw[:, -const1:]*1j
        IRS_coeff = IRS_raw[:, :const2] + IRS_raw[:, -const2:]*1j

        # Reshaping the beamformers tensor into a rectangular array whose columns will be beamformers for each user
        beamformers = torch.reshape(beamformers, (-1, params['num_users'], params['num_antennas']))
        beamformers = torch.transpose(beamformers, 1, 2)

        # Initializing variable to hold sum rate for each sample
        sum_rates = torch.zeros(num_samples)

        # Calculating loss/sum-rate for each sample in this batch
        for idx in range(num_samples):
            # Beamformers and IRS coefficients corresponding to this sample
            bf = beamformers[idx]
            irs = IRS_coeff[idx]

            # Reshaping to column vector
            irs = torch.reshape(irs, (-1, 1))
            diagv = torch.diag(torch.squeeze(irs))
            # Multiplying G and diag(v)
            tmp1 = torch.mm(G, diagv)

            # Variable the hold all the rates
            rates = torch.zeros(params['num_users'])

            # Finding the rate for each user
            for i in range(params['num_users']):
                d = D[:, i:i+1]
                r = R[:, i:i+1]
                tmp2 = torch.conj((d + torch.mm(tmp1, r)).T)
                temp = torch.square(torch.abs(torch.squeeze(torch.mm(tmp2, bf))))
                temp_sum = torch.sum(temp)
                # Calculating the rate for the ith user
                rates[i] = torch.log(1 + (temp[i])/(temp_sum - temp[i] + params['sigma0']**2))

            # Finding sum rate
            sum_rates[idx] = torch.sum(rates)

        # Since we want to minimize the loss, we set loss to be - mean sum rate
        loss = -torch.mean(sum_rates)

        return loss

    def opt_sum_rate_loss(output, params=PARAMS):
        """
        Function to calculate the loss. We will try to minimize this while training the network
        Arguments:
        output              Output returned by the network
        params              All parameters relevant to our run

        Returns
        loss                The calculated loss (Negative sum rate)
        """
        if not params['multi_ch_train']:
            # Getting the channels locally for simplicity of notation
            G = torch.tensor(params['channels']['G'], device=device)
            D = torch.tensor(params['channels']['D'], device=device)
            R = torch.tensor(params['channels']['R'], device=device)
        else:
            # Getting the channels locally for simplicity of notation
            G = torch.tensor(params['channels']['G'][BATCH_ID], device=device)
            D = torch.tensor(params['channels']['D'][BATCH_ID], device=device)
            R = torch.tensor(params['channels']['R'][BATCH_ID], device=device)

        # Getting the raw representation of the beamformers and the IRS coefficients from network output
        const1 = params['num_antennas']*params['num_users']
        const2 = params['num_reflectors']
        BS_raw = output[:, :2*const1]
        IRS_raw = output[:, -2*const2:]

        # Building the actual complex beamformers and IRS coefficients
        beamformers = BS_raw[:, :const1] + BS_raw[:, -const1:]*1j
        IRS_coeff = IRS_raw[:, :const2] + IRS_raw[:, -const2:]*1j

        # Reshaping the beamformers tensor into a rectangular array whose columns will be beamformers for each user
        beamformers = torch.reshape(beamformers, (-1, params['num_users'], params['num_antennas']))
        beamformers = torch.transpose(beamformers, 1, 2)

        # Finding tensor with the reflection coefficients as the diagonal elements
        # Shape: (Batch, Reflectors, Reflectors)
        diagv = torch.diag_embed(IRS_coeff)

        # Multiplying G and diag(v)
        # Shape: (Batch, Antennas, Reflectors)
        tmp1 = G @ diagv

        # Finding effective channel
        # Shape: (Batch, Antennas, Users)
        H = D + tmp1 @ R
        # Shape: (Batch, Users, Antennas)
        H = torch.transpose(H, 1, 2)

        # Product of effective channel and beamformers
        # Shape: (Batch, Antennas, Users)
        prod = H @ beamformers
        prod_abs_sq = torch.square(torch.abs(prod))
        prod_abs_sq_diag = torch.diagonal(prod_abs_sq, dim1=1, dim2=2)            
        sum1 = torch.sum(prod_abs_sq, dim=2)

        sinr = prod_abs_sq_diag / (sum1 - prod_abs_sq_diag + params['sigma0']**2)
        rates = torch.log(1 + sinr)
        sum_rates = torch.sum(rates, dim=1)
        loss = -torch.mean(sum_rates)

        return loss

    def opt_sum_rate_loss_conj(output, params=PARAMS):
        """
        Function to calculate the loss. We will try to minimize this while training the network
        Arguments:
        output              Output returned by the network
        params              All parameters relevant to our run

        Returns
        loss                The calculated loss (Negative sum rate)
        """
        if not params['multi_ch_train']:
            # Getting the channels locally for simplicity of notation
            G = torch.tensor(params['channels']['G'], device=device)
            D = torch.tensor(params['channels']['D'], device=device)
            R = torch.tensor(params['channels']['R'], device=device)
        else:
            # Getting the channels locally for simplicity of notation
            G = torch.tensor(params['channels']['G'][BATCH_ID], device=device)
            D = torch.tensor(params['channels']['D'][BATCH_ID], device=device)
            R = torch.tensor(params['channels']['R'][BATCH_ID], device=device)

        # Getting the raw representation of the beamformers and the IRS coefficients from network output
        const1 = params['num_antennas']*params['num_users']
        const2 = params['num_reflectors']
        BS_raw = output[:, :2*const1]
        IRS_raw = output[:, -2*const2:]

        # Building the actual complex beamformers and IRS coefficients
        beamformers = BS_raw[:, :const1] + BS_raw[:, -const1:]*1j
        IRS_coeff = IRS_raw[:, :const2] + IRS_raw[:, -const2:]*1j

        # Reshaping the beamformers tensor into a rectangular array whose columns will be beamformers for each user
        beamformers = torch.reshape(beamformers, (-1, params['num_users'], params['num_antennas']))
        beamformers = torch.transpose(beamformers, 1, 2)

        # Finding tensor with the reflection coefficients as the diagonal elements
        # Shape: (Batch, Reflectors, Reflectors)
        diagv = torch.diag_embed(IRS_coeff)

        # Multiplying G and diag(v)
        # Shape: (Batch, Antennas, Reflectors)
        tmp1 = G @ diagv

        # Finding effective channel
        # Shape: (Batch, Antennas, Users)
        H = D + tmp1 @ R
        # Shape: (Batch, Users, Antennas)
        H = torch.transpose(H, 1, 2)
        Hconj = torch.conj(H)

        # Product of effective channel and beamformers
        # Shape: (Batch, Antennas, Users)
        prod = Hconj @ beamformers
        prod_abs_sq = torch.square(torch.abs(prod))
        prod_abs_sq_diag = torch.diagonal(prod_abs_sq, dim1=1, dim2=2)            
        sum1 = torch.sum(prod_abs_sq, dim=2)

        sinr = prod_abs_sq_diag / (sum1 - prod_abs_sq_diag + params['sigma0']**2)
        rates = torch.log(1 + sinr)
        sum_rates = torch.sum(rates, dim=1)
        loss = -torch.mean(sum_rates)

        return loss

    ######################################################################
    # Sum Rate Loss
    # loss_fn = sum_rate_loss
    loss_fn = opt_sum_rate_loss

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['lrate'])
    if PARAMS['use_LR_Plateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = 'min',
            factor = 0.6,
            patience = 1,
            threshold = 1e-2,
            threshold_mode = 'abs',
            verbose = True
        )

    ######################################################################
    # Measuring time taken for the training process
    losses = []
    losses_test = []
    global losses_test_min
    losses_test_min = np.inf

    interval = (PARAMS['num_train_samples']//PARAMS['batch_size'])//5
    # interval = 20
    start = time.time()
    for i in range(PARAMS['num_epochs']):
        logit('')
        logit('------------------------------------------------------')
        logit("Epoch %s -------------------------------" % (i+1))

        # Training the network for this epoch
        temp = train_loop(loaders, model, loss_fn, optimizer, interval)
        if PARAMS['use_LR_Plateau']:
            scheduler.step(temp['losses_test'][-1])
        losses += temp['losses']
        losses_test += temp['losses_test']
    temp = test_loop(loaders, model, loss_fn)
    losses_test.append(temp['loss'])

    end = time.time()
    logit('------------------------------------------------------')
    logit('------------------------------------------------------')
    logit('Total time taken: %.5f seconds' % (end-start))

    ######################################################################
    # Saving the trained network
    torch.save(model, PARAMS['model_save_path'])

    # Sum Rate at the end of training on the test dataset
    logit('Train Sum Rate after training = %.5f' % (-losses[-1]))
    logit('Train Loss after training = %.5f' % (losses[-1]))
    logit('Test Sum Rate after training = %.5f' % (-losses_test[-1]))
    logit('Test Loss after training = %.5f' % (losses_test[-1]))

    # Plotting of training and test metrics
    prefix = direc + PARAMS['model_name'] + f'_PL_{PARAMS["pilot_length"]}'
    plotGraphs(losses, losses_test, interval, prefix)

    ######################################################################
    ######################################################################
    ### Calculating the baselines
    # Calculating capacity on test set
    def transform_output(output, params=PARAMS):
        """
        Function to extract the beamformers and phases from the model output
        """
        # Getting the raw representation of the beamformers and the IRS coefficients from network output
        const1 = params['num_antennas']*params['num_users']
        const2 = params['num_reflectors']
        BS_raw = output[:, :2*const1]
        IRS_raw = output[:, -2*const2:]

        # Building the actual complex beamformers and IRS coefficients
        beamformers = BS_raw[:, :const1] + BS_raw[:, -const1:]*1j
        IRS_coeff = IRS_raw[:, :const2] + IRS_raw[:, -const2:]*1j

        # Reshaping the beamformers tensor into a rectangular array whose columns will be beamformers for each user
        # beamformers = np.reshape(beamformers, (-1, params['num_antennas'], params['num_users']))
        beamformers = np.reshape(beamformers, (-1, params['num_users'], params['num_antennas']))
        beamformers = np.transpose(beamformers, (0, 2, 1))

        return beamformers, IRS_coeff

    def water_fill(heights, power, params=PARAMS):
        assert power > 0
        
        # Sorting the heights in ascending order
        heights = np.sort(heights)
        # print(heights)
        assert heights[0]>0    

        # Starting water level which we will update
        level = heights[0] + power

        # Tolerance to stop the algorithm
        tolerance = 1e-6

        while True:
            diff = level - heights
            mask = (diff>0)
            # Number of relevant bins
            relevant_bins = diff[mask]
            num_relevant = len(relevant_bins)
            curr_power = np.sum(relevant_bins)

            if abs(curr_power - power) < tolerance:
                break
            else:
                level += (power - curr_power)/num_relevant

        return level, num_relevant

    def calc_capacity(H, max_iter=10, params=PARAMS):
        """
        Function to calculate capacity
        """
        # Initial covariance matrices for each of the users
        cov = np.random.rand(params['num_users'], 1, 1).astype(np.complex128)

        for _ in range(max_iter):
            tmp_list = []
            for j in range(params['num_users']):
                tmp_list.append(H[j].conj().T @ cov[j] @ H[j])
            tmp_list_sum = np.sum(tmp_list)

            Dinv_diag_list = []
            U_list = []
            Dinv_list = []
            Uh_list = []
            for i in range(params['num_users']):
                tmp1 = np.eye(params['num_antennas'], dtype=np.complex128)
                tmp2 = H[i] @ fractional_matrix_power(tmp1 + tmp_list_sum - tmp_list[i], -0.5)

                # Computing svd
                U, D_diag, Uh = np.linalg.svd(tmp2 @ (tmp2.conj().T))
                D = np.diag(D_diag)
                U_list.append(U)
                Uh_list.append(Uh)
                Dinv = np.linalg.inv(D)
                Dinv_list.append(Dinv)
                Dinv_diag_list.append(np.diagonal(Dinv))

            # Calculating lambda
            level, _ = water_fill(np.concatenate(Dinv_diag_list).astype(np.float64), params['total_power_constraint'])

            for k in range(params['num_users']):
                Lambda = np.maximum(level*np.eye(1) - Dinv_list[k], np.zeros_like(D))

                # Updating the covariance matrix for this iteration
                cov[k] = U_list[k] @ Lambda @ Uh_list[k]

        # Capcacity calculation for the obtained covariance matrices
        accumulator = np.eye(params['num_antennas']).astype(np.complex128)
        for i in range(params['num_users']):
            accumulator += H[i].conj().T @ cov[i] @ H[i]
        capacity = np.log(np.linalg.det(accumulator))

        return capacity

    # List to hold capacity value calculated for each sample
    cap_vals = []

    # Number of batches we want to use for capacity calculation
    num_batches_cap = 10

    for batch, X in enumerate(loaders['test']):
        # Getting the relevant channel for this batch
        G = PARAMS['channels']['G'][batch + PARAMS['num_channels']]
        D = PARAMS['channels']['D'][batch + PARAMS['num_channels']]
        R = PARAMS['channels']['R'][batch + PARAMS['num_channels']]

        # Forward pass through the model to get the predicted output
        pred = model(X[0])['out'].cpu().detach().numpy()
        _, irs = transform_output(pred)

        for i in range(irs.shape[0]):
            # Net channel we will use for capacity calculation
            # Each row corresponds to H1 for each user
            net_ch = np.expand_dims(((D + G@np.diag(irs[i])@R)/PARAMS['sigma0']).T, axis=1)
            cap_vals.append(calc_capacity(net_ch))
        
        if batch == num_batches_cap-1:
            break

    cap_vals = np.array(cap_vals).real
    logit('')
    logit('')
    logit('BASELINES:')
    logit('Average capacity on the Test set: %.5f, Std: %.5f' % (np.mean(cap_vals), np.std(cap_vals)))

    ######################################################################
    # Random theta and BF
    def calc_rate(beamformers, IRS_coeff, G, D, R, num_samples=1000):
        sum_rates = np.zeros(num_samples)

        # Calculating loss/sum-rate for each sample in this batch
        for idx in range(num_samples):
            # Beamformers and IRS coefficients corresponding to this sample
            bf = beamformers[idx]
            irs = IRS_coeff[idx]

            # Reshaping to column vector
            irs = np.reshape(irs, (-1, 1))
            diagv = np.diag(np.squeeze(irs))
            # Multiplying G and diag(v)
            tmp1 = G@diagv

            # Variable the hold all the rates
            rates = np.zeros(PARAMS['num_users'])

            # Finding the rate for each user
            for i in range(PARAMS['num_users']):
                d = D[:, i:i+1]
                r = R[:, i:i+1]
                tmp2 = (d + tmp1@r).T
                temp = np.square(np.abs(np.squeeze(tmp2@bf)))
                temp_sum = np.sum(temp)
                # Calculating the rate for the ith user
                rates[i] = np.log(1 + (temp[i])/(temp_sum - temp[i] + PARAMS['sigma0']**2))

            # Finding sum rate
            sum_rates[idx] = np.sum(rates)

        # Mean sum rate
        mean_sum_rate = np.mean(sum_rates)
        return mean_sum_rate

    num_samples = 100
    sum_rates = []
    for i in range(len(PARAMS['channels']['G'])):
        G = PARAMS['channels']['G'][i]
        D = PARAMS['channels']['D'][i]
        R = PARAMS['channels']['R'][i]

        beamformers = generate_complex_gaussian_array((num_samples, PARAMS['num_antennas'], PARAMS['num_users']))
        IRS_coeff = generate_complex_gaussian_array((num_samples, PARAMS['num_reflectors']))

        # Normalizing
        IRS_coeff = np.divide(IRS_coeff, np.abs(IRS_coeff))

        frob_norm = np.linalg.norm(beamformers, axis=(1,2))
        BS_normalizing_factor = (PARAMS['total_power_constraint']**0.5)/frob_norm
        beamformers = beamformers*BS_normalizing_factor[:, None, None]

        sum_rates.append(calc_rate(beamformers, IRS_coeff, G, D, R, num_samples))
    
    logit('[Random theta, BF] Average SR: %.5f, Std: %.5f' % (np.mean(sum_rates), np.std(sum_rates)))

    ######################################################################
    # Random theta and Hermitian BF
    num_samples = 100
    sum_rates = []
    for i in range(len(PARAMS['channels']['G'])):
        G = PARAMS['channels']['G'][i]
        D = PARAMS['channels']['D'][i]
        R = PARAMS['channels']['R'][i]

        IRS_coeff = generate_complex_gaussian_array((num_samples, PARAMS['num_reflectors']))
        IRS_coeff = np.divide(IRS_coeff, np.abs(IRS_coeff))

        beamformers = np.zeros((num_samples, PARAMS['num_antennas'], PARAMS['num_users']), dtype=complex)

        for idx in range(num_samples):
            irs = IRS_coeff[idx]

            full_ch = D + G@np.diag(irs)@R
            beamformers[idx] = full_ch.conj()

        frob_norm = np.linalg.norm(beamformers, axis=(1,2))
        BS_normalizing_factor = (PARAMS['total_power_constraint']**0.5)/frob_norm
        beamformers = beamformers*BS_normalizing_factor[:, None, None]

        sum_rates.append(calc_rate(beamformers, IRS_coeff, G, D, R, num_samples))
    
    logit('[Random theta, Hermitian BF] Average SR: %.5f, Std: %.5f' % (np.mean(sum_rates), np.std(sum_rates)))

    ######################################################################
    # Random theta and Zero-Forcing
    num_samples = 100
    sum_rates = []
    for i in range(len(PARAMS['channels']['G'])):
        G = PARAMS['channels']['G'][i]
        D = PARAMS['channels']['D'][i]
        R = PARAMS['channels']['R'][i]

        IRS_coeff = generate_complex_gaussian_array((num_samples, PARAMS['num_reflectors']))
        IRS_coeff = np.divide(IRS_coeff, np.abs(IRS_coeff))

        beamformers = np.zeros((num_samples, PARAMS['num_antennas'], PARAMS['num_users']), dtype=complex)

        for idx in range(num_samples):
            irs = IRS_coeff[idx]

            full_ch = D + G@np.diag(irs)@R
            beamformers[idx] = np.linalg.pinv(full_ch.T)

        frob_norm = np.linalg.norm(beamformers, axis=(1,2))
        BS_normalizing_factor = (PARAMS['total_power_constraint']**0.5)/frob_norm
        beamformers = beamformers*BS_normalizing_factor[:, None, None]

        sum_rates.append(calc_rate(beamformers, IRS_coeff, G, D, R, num_samples))
    
    logit('[Random theta, Zero-Forcing] Average SR: %.5f, Std: %.5f' % (np.mean(sum_rates), np.std(sum_rates)))
