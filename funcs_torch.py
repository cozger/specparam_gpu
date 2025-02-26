import torch
import numpy as np

# Helper: Convert input to a torch.Tensor of type double.
def _to_tensor(x, device=None):
    """
    Convert input x to a torch tensor with dtype=torch.float64.
    If x is a NumPy array, convert it.
    
    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input array.
    device : torch.device, optional
        If specified, move the tensor to this device.
        
    Returns
    -------
    torch.Tensor
        Tensor with dtype torch.float64.
    """
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x).double()
    elif torch.is_tensor(x):
        tensor = x.double()
    else:
        raise ValueError("Input must be a numpy array or torch tensor.")
    
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def torch_compute_fwhm(std):
    """Compute the full-width half-max, given the gaussian standard deviation.

    Parameters
    ----------
    std : float
        Gaussian standard deviation.

    Returns
    -------
    float
        Calculated full-width half-max.
    """

    return 2 * torch.sqrt(2 * torch.log(2)) * std


def torch_compute_gauss_std(fwhm):
    """Compute the gaussian standard deviation, given the full-width half-max.

    Parameters
    ----------
    fwhm : float
        Full-width half-max.

    Returns
    -------
    float
        Calculated standard deviation of a gaussian.
    """

    return fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))


#residual function for GPU batches that mimics the torch_expo_nk_function
def batched_residual_function(params_flat,current_batch_size,x_batch,y_batch):
    # Reshape the flat parameter vector to (current_batch_size, 2)
    params = params_flat.view(current_batch_size, 2)
    # For each trial in the batch:
    #   predictions_i = params[i,0] - log10(x_batch[i] ** params[i,1])
    # x_batch has shape (current_batch_size, 55) and we need to broadcast the per-trial parameters.
    predictions = params[:, 0].unsqueeze(1) - torch.log10(torch.pow(x_batch, params[:, 1].unsqueeze(1)))
    # residuals: difference between predictions and the actual y_batch
    residuals = predictions - y_batch
    # Flatten the residuals to a 1D tensor.
    return residuals.view(-1)
def batched_residual_function_generic(params_flat, current_batch_size, x_batch, y_batch, num_params, model_func):
    # Reshape the flat parameter vector to (current_batch_size, num_params)
    params = params_flat.view(current_batch_size, num_params)
    # Compute predictions:
    # For instance, if num_params==3 then for each trial i:
    #    prediction_i = model_func(x_batch[i], params[i,0], params[i,1], params[i,2])
    # We can write this in vectorized form using our model_func.
    predictions = model_func(x_batch, *[params[:, i] for i in range(num_params)])
    # Compute residuals
    residuals = predictions - y_batch
    # Return residuals as a flattened vector
    return residuals.view(-1)

def torch_gaussian_function(xs, *params):
    """Gaussian fitting function (PyTorch version).

    Parameters
    ----------
    xs : 1D or 2D torch.Tensor
        Input x-axis values. For batched evaluation, xs is expected to have shape 
        (batch_size, n_freq).
    *params : float or torch scalars
        Parameters that define the Gaussian function in groups of three:
        each group is (center, height, width), where for batched evaluation each
        parameter is of shape (batch_size,).

    Returns
    -------
    ys : torch.Tensor
        Output values for the Gaussian function, with the same shape as xs.
    """
    device = xs.device if torch.is_tensor(xs) else None
    xs = _to_tensor(xs, device=device)
    ys = torch.zeros_like(xs)
    # Process parameters in groups of three.
    for ctr, hgt, wid in zip(*[iter(params)] * 3):
        # Unsqueeze each parameter so that they have shape (batch_size, 1)
        # This way, the subtraction xs - ctr will broadcast correctly.
        ctr = ctr.unsqueeze(1)
        hgt = hgt.unsqueeze(1)
        wid = wid.unsqueeze(1)
        # print(ctr)
        # print(hgt)
        # print(wid)
        ys = ys + hgt * torch.exp(-((xs - ctr) ** 2) / (2 * wid ** 2))
    return ys


def torch_expo_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component with a 'knee' (PyTorch version).

    NOTE: This function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1D array (numpy or torch.Tensor)
        Input x-axis values.
    *params : float or torch scalars
        Parameters (offset, knee, exp) that define the function:
            y = offset - log10(knee + xs**exp)

    Returns
    -------
    ys : torch.Tensor
        Output values for the exponential function.
    """
    device = None
    if torch.is_tensor(xs):
        device = xs.device
    xs = _to_tensor(xs, device=device)
    # Convert parameters to tensors:
    offset, knee, exp = (_to_tensor(np.array(p), device=device) for p in params)
    ys = offset - torch.log10(knee + torch.pow(xs, exp))
    return ys


def torch_expo_nk_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component without a 'knee' (PyTorch version).

    NOTE: This function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1D array (numpy or torch.Tensor)
        Input x-axis values.
    *params : float or torch scalars
        Parameters (offset, exp) that define the function:
            y = offset - log10(xs**exp)

    Returns
    -------
    ys : torch.Tensor
        Output values for the exponential function without a knee.
    """
    # Determine the device and convert xs.
    device = xs.device if torch.is_tensor(xs) else None
    xs = _to_tensor(xs, device=device)  # now xs is a torch tensor
    
    # Ensure that each parameter is a torch tensor.
    params_t = [p if torch.is_tensor(p) else _to_tensor(p, device=device) for p in params]
    # For each parameter, if it's 1D (shape: (batch_size,)), unsqueeze to shape (batch_size, 1)
    offset, exp = [p.unsqueeze(1) if p.dim() == 1 else p for p in params_t]
    
    # Now xs is (batch_size, n_freq), and offset, exp are (batch_size, 1)
    # This broadcasting works as desired.
    ys = offset - torch.log10(torch.pow(xs, exp))
    return ys



def torch_linear_function(xs, *params):
    """Linear fitting function (PyTorch version).

    Parameters
    ----------
    xs : 1D array (numpy or torch.Tensor)
        Input x-axis values.
    *params : float or torch scalars
        Parameters (offset, slope) that define the linear function:
            y = offset + xs * slope

    Returns
    -------
    ys : torch.Tensor
        Output values for the linear function.
    """
    device = None
    if torch.is_tensor(xs):
        device = xs.device
    xs = _to_tensor(xs, device=device)
    offset, slope = (_to_tensor(np.array(p), device=device) for p in params)
    ys = offset + xs * slope
    return ys


def torch_quadratic_function(xs, *params):
    """Quadratic fitting function (PyTorch version).

    Parameters
    ----------
    xs : 1D array (numpy or torch.Tensor)
        Input x-axis values.
    *params : float or torch scalars
        Parameters (offset, slope, curve) that define the quadratic function:
            y = offset + xs * slope + xs**2 * curve

    Returns
    -------
    ys : torch.Tensor
        Output values for the quadratic function.
    """
    device = None
    if torch.is_tensor(xs):
        device = xs.device
    xs = _to_tensor(xs, device=device)
    offset, slope, curve = (_to_tensor(np.array(p), device=device) for p in params)
    ys = offset + xs * slope + (xs ** 2) * curve
    return ys


def torch_get_pe_func(periodic_mode):
    """Select and return the specified function for the periodic component (PyTorch version).

    Parameters
    ----------
    periodic_mode : {'gaussian'}
        Which periodic fitting function to return.

    Returns
    -------
    pe_func : function
        The PyTorch function for the periodic component.

    Raises
    ------
    ValueError
        If the specified periodic mode label is not understood.
    """
    if periodic_mode == 'gaussian':
        pe_func = torch_gaussian_function
    else:
        raise ValueError("Requested periodic mode not understood.")
    return pe_func


def torch_get_ap_func(aperiodic_mode):
    """Select and return the specified function for the aperiodic component (PyTorch version).

    Parameters
    ----------
    aperiodic_mode : {'fixed', 'knee'}
        Which aperiodic fitting function to return.

    Returns
    -------
    ap_func : function
        The PyTorch function for the aperiodic component.

    Raises
    ------
    ValueError
        If the specified aperiodic mode label is not understood.
    """
    if aperiodic_mode == 'fixed':
        ap_func = torch_expo_nk_function
    elif aperiodic_mode == 'knee':
        ap_func = torch_expo_function
    else:
        raise ValueError("Requested aperiodic mode not understood.")
    return ap_func


def torch_infer_ap_func(aperiodic_params):
    """Infer which aperiodic function was used, from parameters (PyTorch version).

    Parameters
    ----------
    aperiodic_params : list or tuple of float
        Parameters that describe the aperiodic component of a power spectrum.

    Returns
    -------
    aperiodic_mode : {'fixed', 'knee'}
        The kind of aperiodic fitting function the given parameters are consistent with.

    Raises
    ------
    Exception
        If the given parameters are inconsistent with available options.
    """
    if len(aperiodic_params) == 2:
        aperiodic_mode = 'fixed'
    elif len(aperiodic_params) == 3:
        aperiodic_mode = 'knee'
    else:
        raise Exception("The given aperiodic parameters are inconsistent with available options.")
    return aperiodic_mode
