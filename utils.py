# Importing relevant libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
######################################################################
class UnfoldedWMMSE(nn.Module):
    def __init__(self, model_params, params):
        super(UnfoldedWMMSE, self).__init__()

        self.model_params = model_params
        self.params = params
        self.rnn = nn.RNN(
            input_size = self.model_params['RNN_n_inputs'],
            hidden_size = self.model_params['RNN_n_neurons'],
            num_layers = self.model_params['RNN_n_layers'],
            nonlinearity = self.model_params['RNN_non_linearity'],
            bidirectional = self.model_params['RNN_bi_directional'],
            batch_first = True
        )
        self.tmp1 = self.params['num_antennas']*self.params['num_reflectors']
        self.tmp2 = self.params['num_antennas']*self.params['num_users']
        self.tmp3 = self.params['num_reflectors']*self.params['num_users']
        self.tmp = self.tmp1 + self.tmp2 + self.tmp3
        if self.model_params['RNN_bi_directional']:
            self.output_IRS = nn.Linear(2*self.model_params['RNN_n_neurons'], self.model_params['n_outputs_IRS'])
            self.output_ch = nn.Linear(2*self.model_params['RNN_n_neurons'], 2*self.tmp)
        else:
            self.output_IRS = nn.Linear(self.model_params['RNN_n_neurons'], self.model_params['n_outputs_IRS'])
            self.output_ch = nn.Linear(self.model_params['RNN_n_neurons'], 2*self.tmp)

        # Total power constraint
        self.total_power_constraint = self.params['total_power_constraint']

        # Initial value for the Beamformers used in the unfolded WMMSE
        temp = generate_complex_gaussian_array((self.params['num_antennas'], self.params['num_users']))
        self.BF_init = torch.tensor((self.total_power_constraint**0.5)*temp/np.linalg.norm(temp), device=device)

        # Learnable parameters for the learning rate
        self.lr = nn.Parameter(torch.Tensor(self.model_params['num_its_WMMSE'], self.model_params['num_its_PGD']))
        nn.init.constant_(self.lr, 1.0)

    def forward(self, x):
        values = {}
        values['rnn'], _ = self.rnn(x)
        values['output_IRS'] = self.output_IRS(values['rnn'][:,-1,:])

        # Normalizing theta values
        temp1 = torch.square(values['output_IRS'][:, :self.params['num_reflectors']])
        temp2 = torch.square(values['output_IRS'][:, -self.params['num_reflectors']:])
        factor = torch.sqrt(temp1 + temp2)
        temp3 = torch.div(values['output_IRS'][:, :self.params['num_reflectors']], factor)
        temp4 = torch.div(values['output_IRS'][:, -self.params['num_reflectors']:], factor)
        values['normalized_IRS'] = torch.cat((temp3, temp4), dim=1)
        values['theta'] = temp3 + 1j*temp4
        values['diag_theta'] = torch.diag_embed(values['theta'])

        # Getting the channel matrices
        values['output_ch'] = self.output_ch(values['rnn'][:,-1,:])
        G_raw = values['output_ch'][:, :2*self.tmp1]
        D_raw = values['output_ch'][:, 2*self.tmp1 : 2*(self.tmp1+self.tmp2)]
        R_raw = values['output_ch'][:, -2*self.tmp3:]
        # Forming the complex channels
        G = G_raw[:, :self.tmp1] + 1j*G_raw[:, -self.tmp1:]
        D = D_raw[:, :self.tmp2] + 1j*D_raw[:, -self.tmp2:]
        R = R_raw[:, :self.tmp3] + 1j*R_raw[:, -self.tmp3:]
        # Reshaping the channels
        G = G.reshape((-1, self.params['num_antennas'], self.params['num_reflectors']))
        D = D.reshape((-1, self.params['num_antennas'], self.params['num_users']))
        R = R.reshape((-1, self.params['num_reflectors'], self.params['num_users']))

        # Effective channel
        H = D + G@values['diag_theta']@R
        # Shape (Batch, Users, Antennas)
        H = torch.transpose(H, 1, 2)
        Hconj = torch.conj(H)
        # Transpose of H : Shape (Batch, Antennas, Users)
        H_T = torch.transpose(H, 1, 2)

        # Unfolded WMMSE
        BF = self.BF_init
        for i in range(self.model_params['num_its_WMMSE']):
            prod = Hconj@BF
            prod_diag = torch.diagonal(prod, dim1=1, dim2=2)
            prod_abs_sq = torch.square(torch.abs(prod))
            prod_abs_sq_diag = torch.diagonal(prod_abs_sq, dim1=1, dim2=2)            
            sum1 = torch.sum(prod_abs_sq, dim=2)

            # Updating the Receiver gains (Ui)
            Ui = prod_diag/(sum1 + self.params['sigma0']**2)
            Ui_abs_sq = torch.square(torch.abs(Ui))

            # Updating the weights (Wi)
            Wi = (sum1 + self.params['sigma0']**2)/(sum1 - prod_abs_sq_diag + self.params['sigma0']**2)

            # Calculating A matrix : Shape (Batch, Antennas, Antennas)
            temp = (Ui_abs_sq * Wi)[:, None, :]
            H_T_mod = H_T * temp
            A = H_T_mod @ Hconj

            # Updating the Beamformers (BF) with PGD
            for j in range(self.model_params['num_its_PGD']):
                # Calculating the gradient
                grad = 2*(A@BF - H_T*((Ui * Wi)[:, None, :]))

                # Updating the beamformer
                BF = BF - self.lr[i, j]*grad

                # Projection onto the constraint space
                sqrtP = self.total_power_constraint**0.5
                norm = torch.linalg.matrix_norm(BF)
                BF = (sqrtP * BF)/(sqrtP + torch.maximum(torch.zeros(1, device=device), norm - sqrtP))[:,None,None]

        # Flattening out the beamformers
        # Shape (Batch, Users, Antennas)
        BF = torch.transpose(BF, 1, 2)
        BF = BF.reshape((-1, self.params['num_users'] * self.params['num_antennas']))
        BFreal = torch.real(BF)
        BFimag = torch.imag(BF)
        values['normalized_BS'] = torch.cat((BFreal, BFimag), dim=1)

        # Concatenating
        values['out'] = torch.cat((values['normalized_BS'], values['normalized_IRS']), dim=1)

        return {
            "values": values,
            "out_BS": values['normalized_BS'],
            "out_IRS": values['normalized_IRS'],
            "out": values['out'],
            "in": x
        }

######################################################################
class BeamFormer(torch.nn.Module):
    def __init__(self, model_params, params):
        super(BeamFormer, self).__init__()

        self.model_params = model_params
        self.params = params

        # Declaring the layers
        self.fc1 = nn.Sequential(
            nn.Linear(2*self.params['num_antennas']*self.params['pilot_length'], model_params['n_neurons']),
            nn.LeakyReLU(negative_slope = 0.05)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(model_params['n_neurons'], model_params['n_neurons']), 
            nn.LeakyReLU(negative_slope = 0.05)
        )
        self.fc3 = nn.Linear(model_params['n_neurons'], model_params['n_neurons'])
        self.output_BS = nn.Linear(model_params['n_neurons'], 2*self.params['num_antennas']*self.params['num_users'])
        self.output_IRS = nn.Linear(model_params['n_neurons'], 2*self.params['num_reflectors'])

        # Total power constraint
        self.total_power_constraint = self.params['total_power_constraint']

    def forward(self, x):
        values = {}
        values['fc1'] = self.fc1(x)
        values['fc2'] = self.fc2(values['fc1'])
        values['fc3'] = self.fc3(values['fc2'])
        values['output_BS'] = self.output_BS(values['fc3'])
        values['output_IRS'] = self.output_IRS(values['fc3'])

        # Normalizing
        frob_norm = torch.sqrt(torch.sum(torch.square(values['output_BS']), dim=1))
        BS_normalizing_factor = (self.total_power_constraint**0.5)/frob_norm
        values['normalized_BS'] = values['output_BS']*BS_normalizing_factor[:, None]

        temp1 = torch.square(values['output_IRS'][:, :self.params['num_reflectors']])
        temp2 = torch.square(values['output_IRS'][:, -self.params['num_reflectors']:])
        factor = torch.sqrt(temp1 + temp2)
        temp3 = torch.div(values['output_IRS'][:, :self.params['num_reflectors']], factor)
        temp4 = torch.div(values['output_IRS'][:, -self.params['num_reflectors']:], factor)
        values['normalized_IRS'] = torch.cat((temp3, temp4), dim=1)

        # Concatenating
        values['out'] = torch.cat((values['normalized_BS'], values['normalized_IRS']), dim=1)

        return {
            "values": values,
            "out_BS": values['normalized_BS'],
            "out_IRS": values['normalized_IRS'],
            "out": values['out'],
            "in": x
        }

class VanillaRNN(nn.Module):
    def __init__(self, model_params, params):
        super(VanillaRNN, self).__init__()

        self.model_params = model_params
        self.params = params
        self.rnn = nn.RNN(
            input_size = self.model_params['n_inputs'],
            hidden_size = self.model_params['n_neurons'],
            num_layers = self.model_params['n_layers'],
            nonlinearity = self.model_params['non_linearity'],
            bidirectional = self.model_params['bi_directional'],
            batch_first = True
        )
        if self.model_params['bi_directional']:
            self.output_BS = nn.Linear(2*self.model_params['n_neurons'], self.model_params['n_outputs_BS'])
            self.output_IRS = nn.Linear(2*self.model_params['n_neurons'], self.model_params['n_outputs_IRS'])
        else:
            self.output_BS = nn.Linear(self.model_params['n_neurons'], self.model_params['n_outputs_BS'])
            self.output_IRS = nn.Linear(self.model_params['n_neurons'], self.model_params['n_outputs_IRS'])

        # Total power constraint
        self.total_power_constraint = self.params['total_power_constraint']

    def forward(self, x):
        values = {}
        values['rnn'], _ = self.rnn(x)
        values['output_BS'] = self.output_BS(values['rnn'][:,-1,:])
        values['output_IRS'] = self.output_IRS(values['rnn'][:,-1,:])

        # Normalizing
        frob_norm = torch.sqrt(torch.sum(torch.square(values['output_BS']), dim=1))
        BS_normalizing_factor = (self.total_power_constraint**0.5)/frob_norm
        values['normalized_BS'] = values['output_BS']*BS_normalizing_factor[:, None]

        temp1 = torch.square(values['output_IRS'][:, :self.params['num_reflectors']])
        temp2 = torch.square(values['output_IRS'][:, -self.params['num_reflectors']:])
        factor = torch.sqrt(temp1 + temp2)
        temp3 = torch.div(values['output_IRS'][:, :self.params['num_reflectors']], factor)
        temp4 = torch.div(values['output_IRS'][:, -self.params['num_reflectors']:], factor)
        values['normalized_IRS'] = torch.cat((temp3, temp4), dim=1)

        # Concatenating
        values['out'] = torch.cat((values['normalized_BS'], values['normalized_IRS']), dim=1)

        return {
            "values": values,
            "out_BS": values['normalized_BS'],
            "out_IRS": values['normalized_IRS'],
            "out": values['out'],
            "in": x
        }

class LSTM(nn.Module):
    def __init__(self, model_params, params):
        super(LSTM, self).__init__()

        self.model_params = model_params
        self.params = params
        self.rnn = nn.LSTM(
            input_size = self.model_params['n_inputs'],
            hidden_size = self.model_params['n_neurons'],
            num_layers = self.model_params['n_layers'],
            bidirectional = self.model_params['bi_directional'],
            batch_first = True
        )
        if self.model_params['bi_directional']:
            self.output_BS = nn.Linear(2*self.model_params['n_neurons'], self.model_params['n_outputs_BS'])
            self.output_IRS = nn.Linear(2*self.model_params['n_neurons'], self.model_params['n_outputs_IRS'])
        else:
            self.output_BS = nn.Linear(self.model_params['n_neurons'], self.model_params['n_outputs_BS'])
            self.output_IRS = nn.Linear(self.model_params['n_neurons'], self.model_params['n_outputs_IRS'])

        # Total power constraint
        self.total_power_constraint = self.params['total_power_constraint']

    def forward(self, x):
        values = {}
        values['rnn'], _ = self.rnn(x)
        values['output_BS'] = self.output_BS(values['rnn'][:,-1,:])
        values['output_IRS'] = self.output_IRS(values['rnn'][:,-1,:])

        # Normalizing
        frob_norm = torch.sqrt(torch.sum(torch.square(values['output_BS']), dim=1))
        BS_normalizing_factor = (self.total_power_constraint**0.5)/frob_norm
        values['normalized_BS'] = values['output_BS']*BS_normalizing_factor[:, None]

        temp1 = torch.square(values['output_IRS'][:, :self.params['num_reflectors']])
        temp2 = torch.square(values['output_IRS'][:, -self.params['num_reflectors']:])
        factor = torch.sqrt(temp1 + temp2)
        temp3 = torch.div(values['output_IRS'][:, :self.params['num_reflectors']], factor)
        temp4 = torch.div(values['output_IRS'][:, -self.params['num_reflectors']:], factor)
        values['normalized_IRS'] = torch.cat((temp3, temp4), dim=1)

        # Concatenating
        values['out'] = torch.cat((values['normalized_BS'], values['normalized_IRS']), dim=1)

        return {
            "values": values,
            "out_BS": values['normalized_BS'],
            "out_IRS": values['normalized_IRS'],
            "out": values['out'],
            "in": x
        }

class GRU(nn.Module):
    def __init__(self, model_params, params):
        super(GRU, self).__init__()

        self.model_params = model_params
        self.params = params
        self.rnn = nn.GRU(
            input_size = self.model_params['n_inputs'],
            hidden_size = self.model_params['n_neurons'],
            num_layers = self.model_params['n_layers'],
            bidirectional = self.model_params['bi_directional'],
            batch_first = True
        )
        if self.model_params['bi_directional']:
            self.output_BS = nn.Linear(2*self.model_params['n_neurons'], self.model_params['n_outputs_BS'])
            self.output_IRS = nn.Linear(2*self.model_params['n_neurons'], self.model_params['n_outputs_IRS'])
        else:
            self.output_BS = nn.Linear(self.model_params['n_neurons'], self.model_params['n_outputs_BS'])
            self.output_IRS = nn.Linear(self.model_params['n_neurons'], self.model_params['n_outputs_IRS'])

        # Total power constraint
        self.total_power_constraint = self.params['total_power_constraint']

    def forward(self, x):
        values = {}
        values['rnn'], _ = self.rnn(x)
        values['output_BS'] = self.output_BS(values['rnn'][:,-1,:])
        values['output_IRS'] = self.output_IRS(values['rnn'][:,-1,:])

        # Normalizing
        frob_norm = torch.sqrt(torch.sum(torch.square(values['output_BS']), dim=1))
        BS_normalizing_factor = (self.total_power_constraint**0.5)/frob_norm
        values['normalized_BS'] = values['output_BS']*BS_normalizing_factor[:, None]

        temp1 = torch.square(values['output_IRS'][:, :self.params['num_reflectors']])
        temp2 = torch.square(values['output_IRS'][:, -self.params['num_reflectors']:])
        factor = torch.sqrt(temp1 + temp2)
        temp3 = torch.div(values['output_IRS'][:, :self.params['num_reflectors']], factor)
        temp4 = torch.div(values['output_IRS'][:, -self.params['num_reflectors']:], factor)
        values['normalized_IRS'] = torch.cat((temp3, temp4), dim=1)

        # Concatenating
        values['out'] = torch.cat((values['normalized_BS'], values['normalized_IRS']), dim=1)

        return {
            "values": values,
            "out_BS": values['normalized_BS'],
            "out_IRS": values['normalized_IRS'],
            "out": values['out'],
            "in": x
        }

######################################################################
######################################################################
def generate_complex_gaussian_array(shape):
    """
    Function to generate an array of given shape sampled from the complex Gaussian distribution
    Arguments:
    shape           List/tuple containing the shape of the array

    Returns:
    matrix          numpy array sampled from complex Gaussian distribution of given `shape`
    """
    # Forming matrix
    matrix = (np.random.randn(*shape) + np.random.randn(*shape)*1j)/(2**0.5)

    return matrix.astype(np.complex64)

def generate_channels(params, choice='randNormalC', save_channels=True):
    """
    Function to generate the channels
    Arguments:
    params              Dictionary containing all the relevant parameters
    choice              Method of channel generation used
    save_channels       Boolean to indicate whether we want to save the generated channels

    Returns:
    Dictionary containing {
        "G": Channel from BS to IRS,
        "D": Channel from BS to Users,
        "R": Channel from IRS to Users
    }
    """
    # Generating channels as per `choice`
    if choice == 'rand':
        G = np.random.rand(params['num_antennas'], params['num_reflectors']).astype(np.float32)
        D = np.random.rand(params['num_antennas'], params['num_users']).astype(np.float32)
        R = np.random.rand(params['num_reflectors'], params['num_users']).astype(np.float32)
    elif choice == 'randNormal':
        G = np.random.randn(params['num_antennas'], params['num_reflectors']).astype(np.float32)
        D = np.random.randn(params['num_antennas'], params['num_users']).astype(np.float32)
        R = np.random.randn(params['num_reflectors'], params['num_users']).astype(np.float32)
    elif choice == 'randC':
        G = np.random.rand(params['num_antennas'], params['num_reflectors']) + np.random.rand(params['num_antennas'], params['num_reflectors'])*1j
        D = np.random.rand(params['num_antennas'], params['num_users']) + np.random.rand(params['num_antennas'], params['num_users'])*1j
        R = np.random.rand(params['num_reflectors'], params['num_users']) + np.random.rand(params['num_reflectors'], params['num_users'])*1j
        G = G.astype(np.complex64)
        D = D.astype(np.complex64)
        R = R.astype(np.complex64)
    elif choice == 'randNormalC':
        G = generate_complex_gaussian_array((params['num_antennas'], params['num_reflectors']))
        D = generate_complex_gaussian_array((params['num_antennas'], params['num_users']))
        R = generate_complex_gaussian_array((params['num_reflectors'], params['num_users']))
    elif choice == 'custom1':
        ### Rayleigh Fading
        D = generate_complex_gaussian_array((params['num_antennas'], params['num_users']))
        # Calculating the path-loss value
        path_loss_D = 32.6 + 36.7*np.log10(params['dist_BS'])
        path_loss_D = 10**(-path_loss_D/20)
        # Multiplying path loss value to the complex gaussian sampling
        D = path_loss_D * D

        ### Rician Fading
        # Rician factor (epsilon)
        epsilon = 10
        tmp1 = (1/(1+epsilon))**0.5
        tmp2 = (epsilon/(1+epsilon))**0.5
        # Generating G0 and R0 
        G0 = generate_complex_gaussian_array((params['num_antennas'], params['num_reflectors']))
        R0 = generate_complex_gaussian_array((params['num_reflectors'], params['num_users']))
        # Calculating the path-loss values
        path_loss_G = 30 + 22*np.log10(params['dist_BS_IRS'])
        path_loss_R = 30 + 22*np.log10(params['dist_IRS'])
        path_loss_G = 10**(-path_loss_G/20)
        path_loss_R = 10**(-path_loss_R/20) 

        # Steering Terms (Assuming there is no physical distance between the antennas and reflective elements, 
        # steering arrays will just be ones. We can change if needed)
        # steering_term_G = np.ones((params['num_antennas'], params['num_reflectors']))
        # steering_term_R = np.ones((params['num_reflectors'], params['num_users']))

        # Forming the steering vector for R
        steering_term_R = np.zeros((params['num_reflectors'], params['num_users']), dtype=np.complex64)
        ratio = np.squeeze(np.divide(params['loc_users'].imag - params['loc_IRS'].imag, params['dist_IRS']))
        mult_factor = np.mod(np.arange(params['num_reflectors']), 10)
        for idx in range(params['num_users']):
            steering_term_R[:, idx] = np.exp(1j*np.pi * ratio[idx] * mult_factor)

        # Forming the steering vector for G
        ratio_x = (params['loc_IRS'].real - params['loc_BS'].real)/params['dist_BS_IRS']
        ratio_y = (params['loc_BS'].imag - params['loc_IRS'].imag)/params['dist_BS_IRS']
        a_BS = np.squeeze(np.exp(1j*np.pi * ratio_x * np.arange(params['num_antennas'])))
        a_IRS = np.squeeze(np.exp(1j*np.pi * ratio_y * mult_factor))
        steering_term_G = np.outer(a_BS, a_IRS.conj())

        # Making G
        G = path_loss_G * (tmp2*steering_term_G + tmp1*G0)
        # Making R
        R = path_loss_R * (tmp2*steering_term_R + tmp1*R0)

        # Converting all 3 channels to np.complex64
        G = G.astype(np.complex64)
        D = D.astype(np.complex64)
        R = R.astype(np.complex64)
    else:
        sys.exit('$$$$$ generate_channels(): Invalid `choice` for channel generation')

    # Saving the generated channels if `save_channels` is True
    if save_channels:
        np.save(params['path_G'], G)
        np.save(params['path_D'], D)
        np.save(params['path_R'], R)
    
    return {
        "G": G,
        "D": D,
        "R": R
    }

def get_channels(params, import_old=False, choice='randNormalC'):
    """
    Function to get the channels. We generate them or import them based on the boolean `import_old`
    Arguments:
    params              Dictionary containing all the relevant parameters
    import_old          Boolean to indicate whether we want to import old channels
    choice              Method of channel generation used if `import_old` is False

    Returns:
    Dictionary containing {
        "G": Channel from BS to IRS,
        "D": Channel from BS to Users,
        "R": Channel from IRS to Users
    }
    """
    if params['multi_ch_train']:
        if import_old:
            channels = {
                "G": np.load(params['path_G']),
                "D": np.load(params['path_D']),
                "R": np.load(params['path_R'])
            }
        else:
            channels = {
                "G": [],
                "D": [],
                "R": []
            }
            for _ in range(1, params['num_channels']+1):
                tmp = generate_channels(params, choice, False)
                channels['G'].append(tmp['G'])
                channels['D'].append(tmp['D'])
                channels['R'].append(tmp['R'])            
            if params['new_test_channels']:
                for _ in range(1, params['num_test_channels']+1):
                    tmp = generate_channels(params, choice, False)
                    channels['G'].append(tmp['G'])
                    channels['D'].append(tmp['D'])
                    channels['R'].append(tmp['R'])
            np.save(params['path_G'], np.array(channels['G']))
            np.save(params['path_D'], np.array(channels['D']))
            np.save(params['path_R'], np.array(channels['R']))
    else:
        if import_old:
            channels = {
                "G": np.load(params['path_G']),
                "D": np.load(params['path_D']),
                "R": np.load(params['path_R'])
            }
        else:
            channels = generate_channels(params, choice, True)

    return channels

######################################################################
def plotGraphs(losses, losses_test, interval, prefix=''):
    '''
    Function to do all the relevant plotting of loss, accuracy vs iterations
    Arguments:
    losses          List containing loss on the training set every iteration
    losses_test     List containing loss on the test set every 'interval' iterations
    interval        Number of iterations between test set evaluations during training
    prefix          Prefix to be added to plot names when saving     (default: '')
    '''
    # Number of iterations carried out during training
    num_iters = len(losses)
    plt.figure()
    plt.plot(np.arange(num_iters), -np.array(losses))
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Sum Rate')
    plt.title('Train Sum Rate Plot')
    plt.savefig(prefix + '_train_sum_rate.png', bbox_inches='tight', dpi=300)

    plt.figure()
    plt.plot(list(np.arange(0, num_iters, interval)) + [num_iters], -np.array(losses_test))
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Sum Rate')
    plt.title('Test Sum Rate Plot')
    plt.savefig(prefix + '_test_sum_rate.png', bbox_inches='tight', dpi=300)


######################################################################
######################################################################
######################################################################
# self.size_in, self.size_out = size_in, size_out
# weights = torch.Tensor(size_out, size_in)
# self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
# bias = torch.Tensor(size_out)
# self.bias = nn.Parameter(bias)

# # initialize weights and biases
# nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
# fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
# bound = 1 / math.sqrt(fan_in)
# nn.init.uniform_(self.bias, -bound, bound)  # bias init