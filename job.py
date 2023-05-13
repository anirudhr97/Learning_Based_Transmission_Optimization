import argparse
from main import main

if __name__ == '__main__':
    # Utilizing argparse to parse command line arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--L", type=int, help="Pilot Length", default=3)
    parser.add_argument("--model_type", type=str, help="Type of model we want to train", choices=['MLP', 'vanilla', 'LSTM', 'GRU', 'WMMSE'], default='MLP')

    # Code only supports B=1 for now. Will need to add code for more than 1 Base Station.
    parser.add_argument("--B", type=int, help="Number of Base Stations (BS)", default=1)
    parser.add_argument("--K", type=int, help="Number of Users", default=3)
    parser.add_argument("--T", type=int, help="Number of Antennas in each BS", default=10)
    parser.add_argument("--R", type=int, help="Number of Reflective elements in IRS", default=100)

    parser.add_argument("--num_train", type=int, help="Number of Training Samples", default=1000)
    parser.add_argument("--num_test", type=int, help="Number of Test Samples", default=1000)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=10)    # 10
    parser.add_argument("--lr", type=float, help="Learning Rate constant used", default=8e-5)   # 8e-5
    parser.add_argument("--num_epochs", type=int, help="Number of Training Epochs", default=20)

    parser.add_argument("--direc_main", type=str, help="Main Directory", default='D:/DDP/alternating_approach/')
    parser.add_argument("--direc", type=str, help="Directory within main directory used for this specific run", default='D:/DDP/alternating_approach/run-15/')

    parser.add_argument("--import_old_datasets", action='store_true', help="Want to use old datasets? Add flag if Yes")
    parser.add_argument("--import_old_channels", action='store_true', help="Want to use old channels? Add flag if Yes")
    parser.add_argument("--generate_user_locations", action='store_true', help="Want to generate new user locations? Add flag if Yes")

    parser.add_argument("--down_power", type=float, help="Total downlink power constraint (dBm)", default=30.0) # 15
    parser.add_argument("--up_power", type=float, help="Total uplink power constraint (dBm)", default=-210.0)   # 15
    # -210 to 90 dBm

    args = parser.parse_args()
    # Execution starts here
    main(
        L = args.L,
        which_model = args.model_type,
        num_bs = args.B,
        num_users = args.K,
        num_antennas = args.T,
        num_reflectors = args.R,
        num_train_samples = args.num_train,
        num_test_samples = args.num_test,
        batch_size = args.batch_size,
        learning_rate = args.lr,
        num_epochs = args.num_epochs,
        direc_main = args.direc_main,
        direc = args.direc,
        import_old_datasets = args.import_old_datasets,
        import_old_channels = args.import_old_channels,
        generate_user_locations = args.generate_user_locations,
        down_power = args.down_power,
        up_power = args.up_power
    )
