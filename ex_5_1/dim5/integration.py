import torch


def normalization(w, phi):
    """
    对TNN进行归一化处理

    计算TNN的L2范数并进行归一化, 确保每个基函数在L2意义下的单位化.

    Args:
        w: 积分权重, 形状为 [N]
            - N: 积分点总数量, 用于数值积分的求积点数量
        phi: TNN的子网络在积分点上的值, 形状为 [dim, p, N]
            - dim: 张量神经网络的维度, 表示问题的空间维度数(如五维问题中dim=5)
            - p: 张量神经网络的秩(rank), 即基函数的数量, 对应每个维度FNN的输出层神经元数量
            - N: 积分点总数量, 与权重w的长度相同

    Returns:
        tuple: (范数值, 归一化后的phi)
            - 范数值: 形状为 [p]
            - 归一化后的phi: 形状为 [dim, p, N]
    """
    return torch.prod(
        torch.sqrt(torch.sum(w * phi**2, dim=2)), dim=0
    ), phi / torch.sqrt(torch.sum(w * phi**2, dim=2)).unsqueeze(dim=-1)


def int1_tnn(w, alpha, phi, if_sum=True):
    """
    计算单个TNN的积分

    对单个张量神经网络在给定积分点上进行数值积分计算.
    积分形式为: ∫ α·φ(x) dx, 其中φ(x)是张量分解形式的函数.

    Args:
        w: 积分权重, 形状为 [N]
            - N: 积分点总数量, 用于数值积分的求积点数量
        alpha: 缩放参数向量, 形状为 [p]
            - p: 张量神经网络的秩(rank), 即基函数的数量
        phi: TNN的子网络在积分点上的函数值, 形状为 [dim, p, N]
            - dim: 张量神经网络的维度, 表示问题的空间维度数
            - p: 张量神经网络的秩(rank), 与alpha的长度相同
            - N: 积分点总数量, 与权重w的长度相同
        if_sum: 是否对所有基函数求和, 默认为True

    Returns:
        torch.Tensor:
            - 如果if_sum=True, 返回标量积分值, 形状为 [1]
            - 如果if_sum=False, 返回每个基函数的积分值, 形状为 [p]
    """
    # 计算每个维度上的积分: ∫ φ_i(x_i) dx_i
    # phi的形状为[dim, p, N], 对最后一个维度(积分点)进行加权求和
    dim_integrals = torch.sum(w * phi, dim=2)  # 形状: [dim, p]

    # 计算张量积: 对每个基函数j, 计算所有维度的乘积 ∏_i ∫ φ_{i,j}(x_i) dx_i
    tensor_products = torch.prod(dim_integrals, dim=0)  # 形状: [p]

    # 应用缩放参数并根据需要求和
    if if_sum:
        # 返回所有基函数的加权和: Σ_j α_j * ∏_i ∫ φ_{i,j}(x_i) dx_i
        return torch.sum(alpha * tensor_products)
    else:
        # 返回每个基函数的积分值: α_j * ∏_i ∫ φ_{i,j}(x_i) dx_i
        return alpha * tensor_products


def int2_tnn(w, alpha1, phi1, alpha2, phi2, if_sum=True):
    """
    计算两个TNN乘积的积分 (L2内积)

    计算两个张量神经网络的L2内积: ∫ (α1·φ1(x)) * (α2·φ2(x)) dx
    这是求解偏微分方程时计算质量矩阵和刚度矩阵的基础运算.

    Args:
        w: 积分权重, 形状为 [N]
            - N: 积分点总数量, 用于数值积分的求积点数量
        alpha1: 第一个TNN的缩放参数, 形状为 [p1]
            - p1: 第一个TNN的秩(rank), 即基函数的数量
        phi1: 第一个TNN的子网络在积分点上的值, 形状为 [dim, p1, N]
            - dim: 张量神经网络的维度, 表示问题的空间维度数
            - p1: 第一个TNN的秩, 与alpha1的长度相同
            - N: 积分点总数量, 与权重w的长度相同
        alpha2: 第二个TNN的缩放参数, 形状为 [p2]
            - p2: 第二个TNN的秩(rank), 即基函数的数量
        phi2: 第二个TNN的子网络在积分点上的值, 形状为 [dim, p2, N]
            - dim: 张量神经网络的维度, 与phi1的第一个维度相同
            - p2: 第二个TNN的秩, 与alpha2的长度相同
            - N: 积分点总数量, 与权重w和phi1的最后一个维度相同
        if_sum: 是否对所有基函数组合求和, 默认为True

    Returns:
        torch.Tensor:
            - 如果if_sum=True, 返回总的积分值, 形状为 [1]
            - 如果if_sum=False, 返回所有基函数组合的积分矩阵, 形状为 [p1, p2]
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
