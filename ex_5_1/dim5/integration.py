import torch


# TNN的积分运算
def normalization(w, phi):
    return torch.prod(
        torch.sqrt(torch.sum(w * phi**2, dim=2)), dim=0
    ), phi / torch.sqrt(torch.sum(w * phi**2, dim=2)).unsqueeze(dim=-1)


def int1_tnn(w, alpha, phi, if_sum=True):
    """
    单个TNN的积分.

    参数:
        w: 积分权重 [N]
        alpha: 缩放参数 [p]
        phi: TNN在积分点上的值 [dim, p, N]
    返回:
        [1] 如果if_sum=True
        [p] 如果if_sum=False
    """
    if if_sum:
        return torch.sum(alpha * torch.prod(torch.sum(w * phi, dim=2), dim=0))
    else:
        return alpha * torch.prod(torch.sum(w * phi, dim=2), dim=0)


def int2_tnn(w, alpha1, phi1, alpha2, phi2, if_sum=True):
    """
    两个TNN乘积的积分 (两个TNN的L2内积).

    参数:
        w: 积分权重 [N]
        alpha1: TNN1的缩放参数 [p1]
        phi1: TNN1在积分点上的值 [dim, p1, N]
        alpha2: TNN2的缩放参数 [p2]
        phi2: TNN2在积分点上的值 [dim, p2, N]
    返回:
        [1] 如果if_sum=True
        [p1,p2] 如果if_sum=False
    """
    if if_sum:
        return torch.sum(
            torch.outer(alpha1, alpha2)
            * torch.prod((w * phi1) @ phi2.transpose(1, 2), dim=0)
        )
    else:
        return torch.outer(alpha1, alpha2) * torch.prod(
            (w * phi1) @ phi2.transpose(1, 2), dim=0
        )


def int2_tnn_amend_1d(
    w1, w2, alpha1, phi1, alpha2, phi2, grad_phi1, grad_phi2, if_sum=True
):
    """
    两个TNN乘积的积分并分别修正各维度 (两个TNN的H1内积).

    参数:
        w: 积分权重 [N]
        alpha1: TNN1的缩放参数 [p1]
        phi1: TNN1在积分点上的值 [dim,p1,N]
        alpha2: TNN2的缩放参数 [p2]
        phi2: TNN2在积分点上的值 [dim,p2,N]
        grad_phi1: TNN1的梯度值 [dim,p1,N]
        grad_phi2: TNN2的梯度值 [dim,p2,N]
    返回:
        [1] 如果if_sum=True
        [p1,p2] 如果if_sum=False
    """
    # if if_sum:
    #     return torch.sum(int2_tnn(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0))
    # else:
    #     return int2_tnn(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0)

    if if_sum:
        dim = phi1.size(0)
        a = (w1 * phi1) @ phi2.transpose(1, 2).unsqueeze(dim=0).expand(
            dim, -1, -1, -1
        ).clone()
        b = (w2 * grad_phi1) @ grad_phi2.transpose(1, 2)
        a[torch.arange(dim), torch.arange(dim), :, :] = b
        # print(torch.sum(int2_tnn(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0))-torch.sum(torch.outer(alpha1,alpha2)*torch.prod(a,dim=1)))
        return torch.sum(torch.outer(alpha1, alpha2) * torch.prod(a, dim=1))
    else:
        dim = phi1.size(0)
        a = (w1 * phi1) @ phi2.transpose(1, 2).unsqueeze(dim=0).expand(
            dim, -1, -1, -1
        ).clone()
        b = (w2 * grad_phi1) @ grad_phi2.transpose(1, 2)
        a[torch.arange(dim), torch.arange(dim), :, :] = b
        # print(int2_tnn(w1, alpha1, phi1, alpha2, phi2, if_sum=False) * torch.sum(((w2*grad_phi1)@grad_phi2.transpose(1,2))/((w1*phi1)@phi2.transpose(1,2)),dim=0)-torch.sum(torch.outer(alpha1,alpha2)*torch.prod(a,dim=1),dim=0))
        return torch.sum(
            torch.outer(alpha1, alpha2) * torch.prod(a, dim=1), dim=0
        )


def int3_tnn(w, alpha1, phi1, alpha2, phi2, alpha3, phi3, if_sum=True):
    """
    三个TNN乘积的积分.

    参数:
        w: 积分权重 [N]
        alpha1: TNN1的缩放参数 [p1]
        phi1: TNN1在积分点上的值 [dim,p1,N]
        alpha2: TNN2的缩放参数 [p2]
        phi2: TNN2在积分点上的值 [dim,p2,N]
        alpha3: TNN3的缩放参数 [p3]
        phi3: TNN3在积分点上的值 [dim,p3,N]
    返回:
        [1] 如果if_sum=True
        [p1,p2,p3] 如果if_sum=False
    """
    if if_sum:
        return torch.sum(
            torch.einsum("i,j,k->ijk", alpha1, alpha2, alpha3)
            * torch.prod(
                torch.einsum("din,djn,dkn->dijk", w * phi1, phi2, phi3), dim=0
            )
        )
    else:
        return torch.einsum("i,j,k->ijk", alpha1, alpha2, alpha3) * torch.prod(
            torch.einsum("din,djn,dkn->dijk", w * phi1, phi2, phi3), dim=0
        )


def int4_tnn(
    w, alpha1, phi1, alpha2, phi2, alpha3, phi3, alpha4, phi4, if_sum=True
):
    """
    四个TNN乘积的积分.

    参数:
        w: 积分权重 [N]
        alpha1: TNN1的缩放参数 [p1]
        phi1: TNN1在积分点上的值 [dim,p1,N]
        alpha2: TNN2的缩放参数 [p2]
        phi2: TNN2在积分点上的值 [dim,p2,N]
        alpha3: TNN3的缩放参数 [p3]
        phi3: TNN3在积分点上的值 [dim,p3,N]
        alpha4: TNN4的缩放参数 [p4]
        phi4: TNN4在积分点上的值 [dim,p4,N]
    返回:
        [1] 如果if_sum=True
        [p1,p2,p3,p4] 如果if_sum=False
    """
    if if_sum:
        return torch.sum(
            torch.einsum("i,j,k,l->ijkl", alpha1, alpha2, alpha3, alpha4)
            * torch.prod(
                torch.einsum(
                    "din,djn,dkn,dln->dijkl", w * phi1, phi2, phi3, phi4
                ),
                dim=0,
            )
        )
    else:
        return torch.einsum(
            "i,j,k,l->ijkl", alpha1, alpha2, alpha3, alpha4
        ) * torch.prod(
            torch.einsum("din,djn,dkn,dln->dijkl", w * phi1, phi2, phi3, phi4),
            dim=0,
        )


# ********** 误差估计器 **********
def error0_estimate(w, alpha_f, f, alpha, phi, projection=True):
    inner0_phi_phi = int2_tnn(w, alpha, phi, alpha, phi)
    inner0_f_phi = int2_tnn(w, alpha_f, f, alpha, phi)
    inner0_f_f = int2_tnn(w, alpha_f, f, alpha_f, f)
    if projection:
        return torch.sqrt(
            1
            - torch.sum(inner0_f_phi) ** 2
            / (torch.sum(inner0_phi_phi) * torch.sum(inner0_f_f))
        )
    else:
        return torch.sqrt(
            torch.sum(inner0_phi_phi)
            - 2 * torch.sum(inner0_f_phi)
            + torch.sum(inner0_f_f)
        )


def error1_estimate(
    w, alpha_f, f, alpha, phi, grad_f, grad_phi, projection=True
):
    inner1_phi_phi = int2_tnn_amend_1d(
        w, w, alpha, phi, alpha, phi, grad_phi, grad_phi
    )
    inner1_f_phi = int2_tnn_amend_1d(
        w, w, alpha_f, f, alpha, phi, grad_f, grad_phi
    )
    inner1_f_f = int2_tnn_amend_1d(
        w, w, alpha_f, f, alpha_f, f, grad_f, grad_f
    )
    if projection:
        return torch.sqrt(
            1
            - torch.sum(inner1_f_phi) ** 2
            / (torch.sum(inner1_phi_phi) * torch.sum(inner1_f_f))
        )
    else:
        return torch.sqrt(
            torch.sum(inner1_phi_phi)
            - 2 * torch.sum(inner1_f_phi)
            + torch.sum(inner1_f_f)
        )
