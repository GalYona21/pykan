import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import sympy
import trimesh
import scipy.spatial
import plotly.graph_objects as go

import diff_operators
from my_functions import calculate_shape_operator_and_principal_directions
import polyscope as ps

# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)
SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x),
                 'x^2': (lambda x: x**2, lambda x: x**2),
                 'x^3': (lambda x: x**3, lambda x: x**3),
                 'x^4': (lambda x: x**4, lambda x: x**4),
                 '1/x': (lambda x: 1/x, lambda x: 1/x),
                 '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2),
                 '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3),
                 '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4),
                 'sqrt': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x)),
                 '1/sqrt(x)': (lambda x: 1/torch.sqrt(x), lambda x: 1/sympy.sqrt(x)),
                 'exp': (lambda x: torch.exp(x), lambda x: sympy.exp(x)),
                 'log': (lambda x: torch.log(x), lambda x: sympy.log(x)),
                 'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x)),
                 'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x)),
                 'tan': (lambda x: torch.tan(x), lambda x: sympy.tan(x)),
                 'tanh': (lambda x: torch.tanh(x), lambda x: sympy.tanh(x)),
                 'sigmoid': (lambda x: torch.sigmoid(x), sympy.Function('sigmoid')),
                 #'relu': (lambda x: torch.relu(x), relu),
                 'sgn': (lambda x: torch.sign(x), lambda x: sympy.sign(x)),
                 'arcsin': (lambda x: torch.arcsin(x), lambda x: sympy.arcsin(x)),
                 'arctan': (lambda x: torch.arctan(x), lambda x: sympy.atan(x)),
                 'arctanh': (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x)),
                 '0': (lambda x: x*0, lambda x: x*0),
                 'gaussian': (lambda x: torch.exp(-x**2), lambda x: sympy.exp(-x**2)),
                 'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x)),
                 #'logcosh': (lambda x: torch.log(torch.cosh(x)), lambda x: sympy.log(sympy.cosh(x))),
                 #'cosh^2': (lambda x: torch.cosh(x)**2, lambda x: sympy.cosh(x)**2),
                'x^5': (lambda x: x**5, lambda x: x**5)
}

def create_dataset(f, 
                   n_var=2, 
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        
        
    train_label = f(train_input)
    test_label = f(test_input)
        
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset

def create_dataset_from_mesh(mesh_path,sampled_index=0, ball_radius=0.1, test_num=1000, normalize_input=False, normalize_label=False,show_sample=False, seed=0, down_sampling_ratio=1.0, device='cpu'):
    # load a mesh and sample a point cloud from the mesh in ball_radius
    # mesh_path: str, path to the mesh
    mesh = trimesh.load(mesh_path)
    vertices = np.array(mesh.vertices)
    normals = np.array(mesh.vertex_normals)
    neighbors = scipy.spatial.cKDTree(vertices)
    # Sample a point and find its neighbors within the ball radius
    sampled_point = vertices[sampled_index]
    sampled_normal = normals[sampled_index]

    indices = neighbors.query_ball_point(sampled_point, ball_radius)
    sampled_points = vertices[indices]

    if down_sampling_ratio < 1.0:
        # always keep the sampled point
        np.random.seed(seed)
        sampled_points = sampled_points[np.random.choice(len(sampled_points), int(len(sampled_points)*down_sampling_ratio), replace=False)]
        sampled_points = np.concatenate([sampled_point.reshape(1, -1), sampled_points], axis=0)

    vector_to_points = sampled_points - sampled_point
    # Compute an orthonormal basis for the tangent plane
    tangent1 = np.cross(sampled_normal, [1, 0, 0])
    if np.linalg.norm(tangent1) < 1e-6:
        tangent1 = np.cross(sampled_normal, [0, 1, 0])
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    tangent2 = np.cross(sampled_normal, tangent1)

    # Project the points onto the tangent plane
    projected_points = np.dot(vector_to_points, tangent1.reshape(-1, 1)) * tangent1 + np.dot(vector_to_points, tangent2.reshape(-1,1)) * tangent2

    train_labels = np.dot(vector_to_points, sampled_normal)

    # projected_points = vector_to_points - np.dot(vector_to_points, sampled_normal.reshape(-1, 1)) * sampled_normal.reshape(1, -1)
    # Create input and label tensors

    # Create train input tensor (UV coordinates on the tangent plane)
    train_input_tensor = torch.from_numpy(projected_points[:, :2]).float().to(device)

    # Create train label tensor (Z coordinate after projection)
    train_label_tensor = torch.from_numpy(train_labels).float().unsqueeze(1).to(device)

    # Randomly sample UV coordinates for test input
    uv_min, uv_max = torch.min(train_input_tensor, dim=0)[0], torch.max(train_input_tensor, dim=0)[0]
    test_input_tensor = torch.rand(test_num, 2).to(device) * (uv_max - uv_min) + uv_min

    # Evaluate the test labels using the train data
    distances = torch.cdist(test_input_tensor, train_input_tensor)
    _, nearest_indices = torch.min(distances, dim=1)
    test_label_tensor = train_label_tensor[nearest_indices]

    # Normalize input if required
    if normalize_input:
        mean_input = torch.mean(train_input_tensor, dim=0, keepdim=True)
        std_input = torch.std(train_input_tensor, dim=0, keepdim=True)
        train_input_tensor = (train_input_tensor - mean_input) / std_input
        test_input_tensor = (test_input_tensor - mean_input) / std_input

    # Normalize label if required
    if normalize_label:
        mean_label = torch.mean(train_label_tensor, dim=0, keepdim=True)
        std_label = torch.std(train_label_tensor, dim=0, keepdim=True)
        train_label_tensor = (train_label_tensor - mean_label) / std_label
        test_label_tensor = (test_label_tensor - mean_label) / std_label

    dataset = {
        'train_input': train_input_tensor,
        'train_label': train_label_tensor,
        'test_input': test_input_tensor,
        'test_label': test_label_tensor
    }
    if show_sample:

        fig = go.Figure(data=[go.Scatter3d(x=sampled_points[:, 0], y=sampled_points[:, 1], z=sampled_points[:, 2], mode='markers', marker=dict(size=3), name='gt_sampled_points'),
                                go.Scatter3d(x=[sampled_point[0]], y=[sampled_point[1]], z=[sampled_point[2]], mode='markers', marker=dict(size=5, color='red'), name='gt_sampled_point'),
                              go.Cone(x=[sampled_point[0]], y=[sampled_point[1]], z=[sampled_point[2]], u=[sampled_normal[0]], v=[sampled_normal[1]], w=[sampled_normal[2]], showscale=False, sizeref=0.01,  name='Sampled Normal', visible=True)
                                ])
        projected_origin_point = np.argmin(np.linalg.norm(projected_points, axis=1))
        fig2 = go.Figure(data=[go.Scatter3d(x=projected_points[:, 0], y=projected_points[:, 1], z=train_labels, mode='markers', marker=dict(size=3, color='green'), name='projected_sampled_points'),
                               go.Scatter3d(x=[projected_points[projected_origin_point, 0]], y=[projected_points[projected_origin_point, 1]], z=[train_labels[projected_origin_point]],
                                            mode='markers', marker=dict(size=5, color='blue'), name='projected_sampled_point'),
                               go.Cone(x=[projected_points[projected_origin_point, 0]], y=[projected_points[projected_origin_point, 1]], z=[train_labels[projected_origin_point]],
                                       u=[0], v=[0], w=[1], showscale=False, sizeref=0.01, name='Projected Normal',
                                       visible=True)])

        # fig_bunny = go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode='markers', marker=dict(size=3))

        # fig = go.Figure(data=[go.Surface(x=XY[:,0].reshape(30, 30).detach().numpy(),
        #                                  y=XY[:,1].reshape(30, 30).detach().numpy(),
        #                                  z=F.detach().numpy())])
        # Create a grid of points for the XY plane
        x_range = np.linspace(projected_points[:, 0].min(), projected_points[:, 0].max(), 100)
        y_range = np.linspace(projected_points[:, 1].min(), projected_points[:, 1].max(), 100)
        x, y = np.meshgrid(x_range, y_range)
        z = np.zeros_like(x)  # Set z-coordinates to zero for the XY plane

        # Create the transparent XY plane trace
        xy_plane = go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False)

        # Add the XY plane trace to fig2
        fig2.add_trace(xy_plane)

        fig.update_layout(scene=dict(xaxis_title='x',
                                     yaxis_title='y',
                                     zaxis_title='z'))
        fig2.update_layout(scene=dict(xaxis_title='x',
                                     yaxis_title='y',
                                     zaxis_title='f(x, y)'))
        fig.show()
        fig2.show()

    # if show_sample:
    #     # can do the same as above but with polyscope
    #     ps.init()
    #     ps_points = ps.register_point_cloud("points", sampled_points)
    #     ps_sampled_point = ps.register_point_cloud("sampled_point", sampled_point.reshape(1, -1))
    #     ps.get_point_cloud("sampled_point").add_vector_quantity("normal", sampled_normal.reshape(1, -1), enabled=True)
    #     ps.show()
    #     ps.remove_point_cloud("points")
    #     ps.remove_point_cloud("sampled_point")
    #
    #     ps.init()
    #     ps_projected_points = ps.register_point_cloud("projected_points", np.concatenate([projected_points[:,:2], train_labels[:,np.newaxis]], axis=1))
    #     ps_projected_sampled_point = ps.register_point_cloud("projected_sampled_point", projected_points[-1:].reshape(1, -1))
    #     ps.get_point_cloud("projected_sampled_point").add_vector_quantity("normal_projected", np.array([0, 0, 1]).reshape(1, -1), enabled=True)
    #     ps.show()

    return dataset

def gaussian_weighted_mse(x, y, sigma=0.01):
    squared_diff = (x - y) ** 2

    # Compute the Gaussian weights based on the ground truth values (y)
    gaussian_weights = torch.exp(-y ** 2 / (2 * sigma ** 2))

    # Apply the scaled weights to the squared differences
    weighted_squared_diff = squared_diff * gaussian_weights

    # Compute the mean of the weighted squared differences
    loss = torch.mean(weighted_squared_diff)

    return loss

def loss_func_codazzi(pred, y, input):
    mse = torch.mean((pred - y) ** 2)
    # min euclidean norm of input, preferably 0,0,0 point
    mid_point_index = torch.argmin(torch.linalg.norm(input, dim=1))
    output = {'model_out': pred, 'model_in': input}
    d1, d2, k1, k2 = calculate_shape_operator_and_principal_directions(output, mid_point=mid_point_index)

    grad_k1 = diff_operators.gradient(k1, input)
    grad_k2 = diff_operators.gradient(k2, input)

    k1_1 = (grad_k1 * d1).sum(dim=-1).T
    k1_2 = (grad_k1 * d2).sum(dim=-1).T
    k2_1 = (grad_k2 * d1).sum(dim=-1).T
    k2_2 = (grad_k2 * d2).sum(dim=-1).T

    # Compute gradients for k1_2 and k2_1 separately
    grad_k1_2 = diff_operators.gradient(k1_2, input)
    grad_k2_1 = diff_operators.gradient(k2_1, input)

    k1_22 = (grad_k1_2 * d2).sum(dim=-1).T
    k2_11 = (grad_k2_1 * d1).sum(dim=-1).T

    k_diff = k1 - k2
    k_diff_sq = k_diff ** 2

    term1 = (k1_22 - k2_11) * k_diff
    term2 = k1_1 * k2_1
    term3 = k1_2 * k2_2
    term4 = 2 * k2_1 ** 2
    term5 = 2 * k1_2 ** 2
    term6 = k1 * k2 * k_diff_sq

    loss = (term1 + term2 + term3 - term4 - term5 - term6).abs_().mean()
    loss = loss + mse
    return loss


def fit_params(x, y, fun, a_range=(-10, 10), b_range=(-10, 10), grid_number=101, iteration=3, verbose=True,
               device='cpu'):
    '''
    fit a, b, c, d such that

    .. math::
        |y-(cf(ax+b)+d)|^2

    is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.

    Args:
    -----
        x : 1D array
            x values
        y : 1D array
            y values
        fun : function
            symbolic function
        a_range : tuple
            sweeping range of a
        b_range : tuple
            sweeping range of b
        grid_num : int
            number of steps along a and b
        iteration : int
            number of zooming in
        verbose : bool
            print extra information if True
        device : str
            device

    Returns:
    --------
        a_best : float
            best fitted a
        b_best : float
            best fitted b
        c_best : float
            best fitted c
        d_best : float
            best fitted d
        r2_best : float
            best r2 (coefficient of determination)

    Example
    -------
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    '''
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij')
        post_fun = fun(a_grid[None, :, :] * x[:, None, None] + b_grid[None, :, :])
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = torch.sum((post_fun - x_mean) * (y - y_mean)[:, None, None], dim=0) ** 2
        denominator = torch.sum((post_fun - x_mean) ** 2, dim=0) * torch.sum((y - y_mean)[:, None, None] ** 2, dim=0)
        r2 = numerator / (denominator + 1e-4)
        r2 = torch.nan_to_num(r2)

        best_id = torch.argmax(r2)
        a_id, b_id = torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number

        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose == True:
                print('Best value at boundary.')
            if a_id == 0:
                a_arange = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_arange = [a_[-2], a_[-1]]
            if b_id == 0:
                b_arange = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_arange = [b_[-2], b_[-1]]

        else:
            a_range = [a_[a_id - 1], a_[a_id + 1]]
            b_range = [b_[b_id - 1], b_[b_id + 1]]

    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]

    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(f'r2 is not very high, please double check if you are choosing the correct symbolic function.')

    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(post_fun[:, None].detach().cpu().numpy(), y.detach().cpu().numpy())
    c_best = torch.from_numpy(reg.coef_)[0].to(device)
    d_best = torch.from_numpy(np.array(reg.intercept_)).to(device)
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best


def add_symbolic(name, fun):
    '''
    add a symbolic function to library
    
    Args:
    -----
        name : str
            name of the function
        fun : fun
            torch function or lambda function
    
    Returns:
    --------
        None
    
    Example
    -------
    >>> print(SYMBOLIC_LIB['Bessel'])
    KeyError: 'Bessel'
    >>> add_symbolic('Bessel', torch.special.bessel_j0)
    >>> print(SYMBOLIC_LIB['Bessel'])
    (<built-in function special_bessel_j0>, Bessel)
    '''
    exec(f"globals()['{name}'] = sympy.Function('{name}')")
    SYMBOLIC_LIB[name] = (fun, globals()[name])


