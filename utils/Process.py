import torch


def EProjSimplex_new_matrix(v, k=1):
    row_means = torch.mean(v, dim=1)
    v0 = v - row_means.view(-1, 1) + k / v.size(1)
    vmin = torch.min(v0, dim=1).values

    negative_mask = vmin < 0
    positive_mask = ~negative_mask

    v1 = v0.clone()

    ft = torch.ones_like(vmin)
    lambda_m = torch.zeros_like(vmin)
    iteration = 100
    for _ in range(iteration):
        v1[negative_mask] = v0[negative_mask] - lambda_m[negative_mask].unsqueeze(1)
        posidx = v1 > 0
        npos = torch.sum(posidx, dim=1)
        g = -npos
        f = torch.sum(torch.where(posidx, v1, torch.zeros_like(v1)), dim=1) - k
        lambda_m = lambda_m - f / g
        ft += 1
        x = torch.max(v1, torch.zeros_like(v1))
        x[ft > iteration] = torch.max(
            v1[ft > iteration], torch.zeros_like(v1[ft > iteration])
        )

    x_list = torch.where(negative_mask.unsqueeze(1), x, v0)
    x_list = torch.where(positive_mask.unsqueeze(1), x_list, v1)

    return torch.clamp(x_list, min=0)


def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1)  # [B, N, f]
    A = -2 * torch.bmm(y, x)  # [B, N, N]
    A += torch.sum(y**2, dim=2, keepdim=True)  # [B, N, 1]
    A += torch.sum(x**2, dim=1, keepdim=True)  # [B, 1, N]
    return A.squeeze()
