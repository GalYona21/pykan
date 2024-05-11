import torch

import diff_operators
from diff_operators import mean_curvature


def calculate_shape_operator_and_principal_directions(output, mid_point, return_pushed_forward_vectors= False):
    grad_f = torch.autograd.grad(output['model_out'], output["model_in"], grad_outputs=torch.ones_like(output['model_out']), create_graph=True)[0]

    dxdf = grad_f[:, 0]
    dydf = grad_f[:, 1]
    dxxdf_and_dxydf = torch.autograd.grad(dxdf, output["model_in"], grad_outputs=torch.ones_like(dxdf), create_graph=True)[0]
    dxydf_and_dyydf = torch.autograd.grad(dydf, output["model_in"], grad_outputs=torch.ones_like(dydf), create_graph=True)[0]
    # Hf = torch.stack([dxxdf_and_dxydf, dxydf_and_dyydf], dim=1)

    # calculate first fundamental form
    f_x = torch.stack([torch.ones_like(grad_f[:, 0]), torch.zeros_like(grad_f[:, 0]), grad_f[:,0]], dim=0).T
    f_y = torch.stack([torch.zeros_like(grad_f[:, 1]), torch.ones_like(grad_f[:, 1]), grad_f[:,1]], dim=0).T

    # Calculate the first fundamental form I for each point vectorized

    E = torch.sum(f_x * f_x, dim=1)
    F = torch.sum(f_x * f_y, dim=1)
    G = torch.sum(f_y * f_y, dim=1)
    I = torch.stack([E, F, F, G], dim=1).reshape(-1, 2, 2)

    # calculate the second fundamental form II
    # N = torch.nn.functional.normalize(torch.cross(f_x, f_y), dim=0)
    # mesh_o3d = o3d.geometry.TriangleMesh()
    # mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.v)
    # mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.f)
    # mesh_o3d.compute_vertex_normals()
    # normals = torch.tensor(mesh_o3d.vertex_normals, dtype=torch.float32)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(mesh.v)
    #
    # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # point_cloud.orient_normals_consistent_tangent_plane(6)
    # # point_cloud.orient_normals_towards_camera_location([0, 0, 1])
    # normals = np.asarray(point_cloud.normals)
    # normals = torch.tensor(normals, dtype=torch.float32)
    normals = torch.nn.functional.normalize(torch.cross(f_x, f_y), dim=1)

    f_xx = torch.stack([torch.zeros_like(dxxdf_and_dxydf[:, 0]), torch.zeros_like(dxxdf_and_dxydf[:, 0]), dxxdf_and_dxydf[:, 0]], dim=0).T
    f_xy = torch.stack([torch.zeros_like(dxydf_and_dyydf[:, 1]), torch.zeros_like(dxydf_and_dyydf[:, 1]), dxxdf_and_dxydf[:, 1]], dim=0).T
    f_yy = torch.stack([torch.zeros_like(dxydf_and_dyydf[:, 1]), torch.zeros_like(dxydf_and_dyydf[:, 1]), dxydf_and_dyydf[:, 1]], dim=0).T

    L = torch.sum(f_xx * normals, dim=1)
    M = torch.sum(f_xy * normals, dim=1)
    N = torch.sum(f_yy * normals, dim=1)
    II = torch.stack([L, M, M, N], dim=1).reshape(-1, 2, 2)

    # Calculate the shape operator S
    S = -torch.inverse(I) @ II

    # calculate the principal directions
    eigenvalues, eigenvectors = torch.linalg.eigh(S)

    # plot the eigenvectors of each point on a 2d plot using plt
    # eigenvectors = eigenvectors.reshape(-1, 2, 2)
    # eigenvalues = eigenvalues.reshape(-1, 2)
    # eigenvectors = eigenvectors.detach().numpy()
    # fig, ax = plt.subplots()
    # ax.quiver(mesh.v[:, 0], mesh.v[:, 1], eigenvectors[:, 0, 0], eigenvectors[:, 0, 1], color='r')
    # ax.quiver(mesh.v[:, 0], mesh.v[:, 1], eigenvectors[:, 1, 0], eigenvectors[:, 1, 1], color='b')
    # ax.set_aspect('equal')
    # plt.show()

    # calculate the differential that pushes the principal directions to 3D
    # stack f_x and f_y to get the basis of the tangent space
    tangent_basis = torch.stack([f_x, f_y], dim=1)

    e1_e2 = torch.einsum('ijk,ikl->ijl', eigenvectors, tangent_basis)
    e1 = e1_e2[:, 0, :]
    e2 = e1_e2[:, 1, :]

    e1 = e1 / torch.linalg.norm(e1, axis=1)[:, None]
    e2 = e2 / torch.linalg.norm(e2, axis=1)[:, None]
    k1 = eigenvalues[:, 0]
    k2 = eigenvalues[:, 1]
    # center point index
    center_point_index = mid_point
    # igl_e1, igl_e2, igl_k1, igl_k2 = igl.principal_curvature(mesh.v, mesh.f)
    # # change from float64 to float32 in np
    # igl_e1 = igl_e1.astype(np.float32)
    # igl_e2 = igl_e2.astype(np.float32)

    # inr surface reconstruction
    # rand_sampled_uv = torch.rand(1000, 2) * 2 - 1
    # rand_sampled_f_uv = model(rand_sampled_uv)['model_out']
    # rand_output_to_vis = np.stack([rand_sampled_uv[:, 0].detach().numpy(), rand_sampled_uv[:, 1].detach().numpy(),
    #                                rand_sampled_f_uv.detach().numpy().flatten()], axis=1)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(rand_output_to_vis)
    #
    # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # surface_reconstruction = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, 0.5)
    # create
    # visualize with wireframe of triangle mesh and lightning and shader
    # o3d.visualization.draw_geometries([surface_reconstruction], mesh_show_wireframe=True, mesh_show_back_face=True, mesh_show_normal=True)
    # mesh_to_plot = (np.array(surface_reconstruction.vertices, dtype=np.float32), np.array(surface_reconstruction.triangles, dtype=np.float32))


    # visualize_pointclouds2(mesh.v, vector_fields_to_visualize=e1, vector_field_to_visualize2=e2, fps_indices=np.arange(0, len(mesh.v)), arrow_scale=0.1)
    # visualize_pointclouds2(mesh.v, vector_fields_to_visualize=[e1[center_point_index], e2[center_point_index], igl_e1[center_point_index], igl_e2[center_point_index]], fps_indices=center_point_index, arrow_scale=0.1)
    # visualize_meshes.visualize_meshes_func(mesh_to_plot, vector_fields_to_visualize=[e1[center_point_index], e2[center_point_index], igl_e1[center_point_index], igl_e2[center_point_index]], fps_indices=center_point_index, arrow_scale=0.1)
    # using igl to calculate principal curvatures and directions
    if return_pushed_forward_vectors:
        return e1[center_point_index], e2[center_point_index], grad_f

    # we don't concern with the sign of the direction of e1 and e2
    eigenvector1_absolute = torch.abs(eigenvectors[center_point_index, :, 0])
    eigenvector2_absolute = torch.abs(eigenvectors[center_point_index, :, 1])
    return  eigenvector1_absolute, eigenvector2_absolute, k1[center_point_index], k2[center_point_index]

def mean_curvature(model, x):
    y = model(x)
    gradient = diff_operators.gradient(y, x)
    mean_curvature = diff_operators.mean_curvature(y, x, gradient)
    return mean_curvature


def calculate_signature(model, x, is_siren=False):
    y = model(x)
    if is_siren:
        output = y
    else:
        output = {'model_in': x, 'model_out': y}
    d1, d2, k1, k2 = calculate_shape_operator_and_principal_directions(output, mid_point=0)
    H = (k1 + k2)/2
    K = k1 * k2
    H = H.unsqueeze(0)
    K = K.unsqueeze(0)
    grad_H = diff_operators.gradient(H, output['model_in'])
    H_1 = torch.sum(grad_H * d1, dim=1)
    H_2 = torch.sum(grad_H * d2, dim=1)
    grad_K = diff_operators.gradient(K, output["model_in"])
    K_1 = torch.sum(grad_K * d1, dim=1)
    K_2 = torch.sum(grad_K * d2, dim=1)
    signature = torch.stack([H, K, H_1, H_2, K_1, K_2], dim=1)
    return signature

def calculate_signature_over_all_surface(model, x):
    y = model(x)
    output = {'model_in': x, 'model_out': y}
    all_point_indices = torch.arange(x.shape[0])
    d1, d2, k1, k2 = calculate_shape_operator_and_principal_directions(output, mid_point=all_point_indices)
    H = (k1 + k2)/2
    K = k1 * k2
    H = H.unsqueeze(0)
    K = K.unsqueeze(0)
    grad_H = diff_operators.gradient(H, x)
    H_1 = torch.sum(grad_H * d1, dim=1)
    H_2 = torch.sum(grad_H * d2, dim=1)
    grad_K = diff_operators.gradient(K, x)
    K_1 = torch.sum(grad_K * d1, dim=1)
    K_2 = torch.sum(grad_K * d2, dim=1)

    signature = torch.stack([H.squeeze(), K.squeeze(), H_1, H_2, K_1, K_2], dim=1)
    # average over all points
    signature = torch.mean(signature, dim=0)
    return signature

def format_signature(signature):
    H, K, H_1, H_2, K_1, K_2 = signature.squeeze().tolist()
    formatted_signature = (
        f"H: {H:.4f}, K: {K:.4f}, "
        f"H_1: {H_1:.4f}, H_2: {H_2:.4f}, "
        f"K_1: {K_1:.4f}, K_2: {K_2:.4f}"
    )
    return formatted_signature
