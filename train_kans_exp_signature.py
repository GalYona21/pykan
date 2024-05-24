from kan import *
from my_functions import calculate_signature, calculate_signature_over_all_surface, format_signature
import fpsample


mesh_path = "data/armadillo_curvs.ply"
mesh = trimesh.load(mesh_path)
vertices = np.array(mesh.vertices)
sampled_indices = fpsample.fps_sampling(vertices, 200, start_idx=0) # the start index is for reproducibility
print(sampled_indices)


grid_range = 1.0
ball_radius = 0.09 # for bunny worked good with 0.09 around 250 points neighbors
lr = 1.0
lamb=100.0

signatures_full_sampling = torch.zeros((len(sampled_indices), 6), dtype=torch.float32)
signatures_down_sampled = torch.zeros((len(sampled_indices), 6), dtype=torch.float32)
for i,sampled_index in enumerate(sampled_indices):
    dataset_full_sample = create_dataset_from_mesh(mesh_path=mesh_path, sampled_index=sampled_index, show_sample=False, ball_radius=ball_radius, seed=i, down_sampling_ratio=1.0)
    print("dataset full size:" , len(dataset_full_sample["train_input"]))
    model_full_sampled = KAN(width=[2, 5, 1], grid=1, k=5, grid_eps=1.0, seed=0, grid_range=[-grid_range, grid_range], noise_scale=0.1, noise_scale_base=0.1, base_fun=torch.nn.SiLU(), learn_rotation_mat=False)
    model_full_sampled.train(dataset_full_sample, opt="LBFGS", steps=20, lr=lr, lamb=lamb, lamb_l1=1.0, lamb_entropy=1.0, lamb_coef=0.0, lamb_coefdiff=0.0, sglr_avoid=True, loss_fn=gaussian_weighted_mse)
    model_full_sampled = model_full_sampled.prune(threshold=3e-3, mode='manual', active_neurons_id=[[0, 1], [0]])
    model_full_sampled.train(dataset_full_sample, opt="LBFGS", steps=50, lr=lr, sglr_avoid=True, loss_fn=gaussian_weighted_mse)
     # manual mode
    # if nan continue

    # fix all
    try:
        model_full_sampled.fix_symbolic(0, 0, 0, 'x^4')
        model_full_sampled.fix_symbolic(0, 1, 0, 'x^3')
        model_full_sampled.fix_symbolic(1, 0, 0, 'x^2')
    except:
        print("nan encountered")
        continue


    model_full_sampled.train(dataset_full_sample, opt="LBFGS", steps=50, lr=lr, sglr_avoid=True, lamb_coef=0.10, lamb_coefdiff=0.10, loss_fn=gaussian_weighted_mse)

    print(model_full_sampled.symbolic_formula()[0][0])
    x = torch.tensor([[0.0], [0.0]], requires_grad=True).T

    signature_full = calculate_signature(model_full_sampled, x) # only calc sig in the origin


    # down sampling

    dataset_down_sampled = create_dataset_from_mesh(mesh_path=mesh_path, sampled_index=sampled_index, show_sample=False, ball_radius=ball_radius, seed=i, down_sampling_ratio=0.8)
    print("dataset down sampled size:" , len(dataset_down_sampled["train_input"]))
    model_down_sampled = KAN(width=[2, 5, 1], grid=1, k=5, grid_eps=1.0, seed=0, grid_range=[-grid_range, grid_range], noise_scale=0.1, noise_scale_base=0.1, base_fun=torch.nn.SiLU(), learn_rotation_mat=False)
    model_down_sampled.train(dataset_down_sampled, opt="LBFGS", steps=20, lr=lr, lamb=lamb, lamb_l1=1.0, lamb_entropy=1.0, lamb_coef=0.0, lamb_coefdiff=0.0, sglr_avoid=True, loss_fn=gaussian_weighted_mse)
    model_down_sampled = model_full_sampled.prune(threshold=3e-3, mode='manual', active_neurons_id=[[0, 1], [0]])
    model_down_sampled.train(dataset_down_sampled, opt="LBFGS", steps=50, lr=lr, sglr_avoid=True, loss_fn=gaussian_weighted_mse)
     # manual mode

    # fix all
    try:
        model_down_sampled.fix_symbolic(0, 0, 0, 'x^4')
        model_down_sampled.fix_symbolic(0, 1, 0, 'x^3')
        model_down_sampled.fix_symbolic(1, 0, 0, 'x^2')
    except:
        print("nan encountered")
        continue

    model_down_sampled.train(dataset_down_sampled, opt="LBFGS", steps=50, lr=lr, sglr_avoid=True, lamb_coef=0.10, lamb_coefdiff=0.10, loss_fn=gaussian_weighted_mse)

    print(model_down_sampled.symbolic_formula()[0][0])
    x = torch.tensor([[0.0], [0.0]], requires_grad=True).T

    signature_down_sampled = calculate_signature(model_down_sampled, x) # only calc sig in the origin
    # check if signature_full or signature_down_sample is nan
    if torch.isnan(signature_full).any(dim=1) or torch.isnan(signature_down_sampled).any(dim=1):
        print("nan encountered")
        continue

    print("sampled_index", i, "signature ", signature_full)
    signatures_full_sampling[i,:] = signature_full
    print("sampled_index", i, "signature ", signature_down_sampled)
    signatures_down_sampled[i,:] = signature_down_sampled



indices_full_thrown = torch.isnan(signatures_full_sampling).any(dim=1)
indices_full_thrown =torch.concatenate([indices_full_thrown, signatures_full_sampling.norm(dim=1) == 0])
indices_down_sampled_thrown = torch.isnan(signatures_down_sampled).any(dim=1)
indices_down_sampled_thrown = torch.concatenate([indices_down_sampled_thrown, signatures_down_sampled.norm(dim=1) == 0])

# throw 0's
signatures_full_sampling = signatures_full_sampling[signatures_full_sampling.norm(dim=1) != 0]
signatures_down_sampled = signatures_down_sampled[signatures_down_sampled.norm(dim=1) != 0]
# throw nan's
signatures_full_sampling = signatures_full_sampling[~torch.isnan(signatures_full_sampling).any(dim=1)]
signatures_down_sampled = signatures_down_sampled[~torch.isnan(signatures_down_sampled).any(dim=1)]


# indice_thrown indices, not logical
# indices_full_thrown = torch.concatenate(indices_full_thrown)
print(indices_full_thrown.shape)
print(torch.arange(len(indices_full_thrown))[indices_full_thrown])

print(indices_down_sampled_thrown.shape)
print(torch.arange(len(indices_down_sampled_thrown))[indices_down_sampled_thrown])


# normalize each entry to 1 in each of the 6 columns of the signature
signatures_full_sampling = signatures_full_sampling / torch.norm(signatures_full_sampling, dim=0)
signatures_down_sampled = signatures_down_sampled / torch.norm(signatures_down_sampled, dim=0)
print(signatures_full_sampling)
print(signatures_down_sampled)


# measure correspondence according distance between signatures
# before normalized each entry to 1

distances = torch.cdist(signatures_full_sampling, signatures_down_sampled, p=2)
print(distances)
# take minimum distance for each row and assign index which is the closest for each surface
min_distances, min_indices = torch.min(distances, dim=1)
print(min_distances)
print(min_indices)

# print("correspondences:"+str(torch.sum(min_indices == torch.arange(len(min_indices)))))

for i in range(6):
    distances = torch.cdist(signatures_full_sampling[:, :i+1], signatures_down_sampled[:, :i+1], p=2)
    min_distances, min_indices = torch.min(distances, dim=1)
    print("i", i, "min_distances", min_distances, "min_indices", min_indices)
    print("i", i, "sum", torch.sum(min_indices == torch.arange(len(min_indices))))

for i in range(6):
    distances = torch.cdist(signatures_full_sampling[:, i:i+1], signatures_down_sampled[:, i:i+1], p=2)
    min_distances, min_indices = torch.min(distances, dim=1)
    print("i", i, "min_distances", min_distances, "min_indices", min_indices)
    print("i", i, "sum", torch.sum(min_indices == torch.arange(len(min_indices))))


torch.save(signatures_full_sampling, mesh_path+ 'signatures_full_sampling_200_surfaces.pt')
torch.save(signatures_down_sampled, mesh_path+'signatures_down_sampled_200_surfaces.pt')