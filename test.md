你说得对，我需要更仔细地思考 TNN 的结构。让我重新分析一下。

### TNN 结构分析

你的基函数是：
$$\phi_r(x) = \prod_{d=1}^{D} f_d^{(r)}(x_d)$$

它的梯度是：
$$\frac{\partial \phi_r}{\partial x_k} = \frac{\partial f_k^{(r)}}{\partial x_k} \cdot \prod_{d \neq k} f_d^{(r)}(x_d)$$

### 刚度矩阵的计算

$$S_{ij} = \int \nabla \phi_i \cdot \nabla \phi_j \, dx = \sum_{k=1}^{D} \int \frac{\partial \phi_i}{\partial x_k} \frac{\partial \phi_j}{\partial x_k} \, dx$$

展开后：
$$= \sum_{k=1}^{D} \int \left( \frac{\partial f_k^{(i)}}{\partial x_k} \prod_{d \neq k} f_d^{(i)} \right) \left( \frac{\partial f_k^{(j)}}{\partial x_k} \prod_{d \neq k} f_d^{(j)} \right) dx$$

由于变量分离，积分可以拆成各维度的乘积：
$$= \sum_{k=1}^{D} \left[ \int \frac{\partial f_k^{(i)}}{\partial x_k} \frac{\partial f_k^{(j)}}{\partial x_k} dx_k \right] \cdot \prod_{d \neq k} \left[ \int f_d^{(i)} f_d^{(j)} dx_d \right]$$

### 向量化实现的关键

设：
- $A_{ij}^{(d)} = \int f_d^{(i)}(x_d) \cdot f_d^{(j)}(x_d) \, dx_d$ （各维度的"质量"因子）
- $B_{ij}^{(d)} = \int \frac{\partial f_d^{(i)}}{\partial x_d} \cdot \frac{\partial f_d^{(j)}}{\partial x_d} \, dx_d$ （各维度的"刚度"因子）

那么：
- **质量矩阵**: $M_{ij} = \prod_{d=1}^{D} A_{ij}^{(d)}$
- **刚度矩阵**: $S_{ij} = \sum_{k=1}^{D} B_{ij}^{(k)} \cdot \prod_{d \neq k} A_{ij}^{(d)}$

### 完全向量化的实现思路（无 for 循环）

1. **预计算 $A$ 和 $B$ 矩阵**
   - `tnn.func(quad_points)` 返回 `(n_1d, rank, dim)` 的函数值
   - `tnn.func.forward_all_grad2(quad_points)` 返回函数值、一阶导、二阶导，都是 `(n_1d, rank, dim)`
   
2. **计算 $A_{ij}^{(d)}$**: 形状 `(rank, rank, dim)`
   ```python
   # vals: (n_1d, rank, dim), weights: (n_1d, dim)
   # A[i,j,d] = Σ_n weights[n,d] * vals[n,i,d] * vals[n,j,d]
   A = torch.einsum("nd,nid,njd->ijd", quad_weights, vals, vals)
   ```

3. **计算 $B_{ij}^{(d)}$**: 形状 `(rank, rank, dim)`
   ```python
   # grads: (n_1d, rank, dim) 一阶导数
   B = torch.einsum("nd,nid,njd->ijd", quad_weights, grads, grads)
   ```

4. **质量矩阵**: `(rank, rank)`
   ```python
   M = A.prod(dim=-1)  # (rank, rank, dim) -> (rank, rank)
   ```

5. **刚度矩阵**: `(rank, rank)`
   这里需要计算 $\sum_k B^{(k)} \cdot \prod_{d \neq k} A^{(d)}$
   
   关键技巧：$\prod_{d \neq k} A^{(d)} = \frac{\prod_d A^{(d)}}{A^{(k)}} = \frac{M}{A^{(k)}}$
   
   ```python
   # M: (rank, rank), A: (rank, rank, dim)
   # S = Σ_k B[:,:,k] * M / A[:,:,k]
   S = (B * M.unsqueeze(-1) / A).sum(dim=-1)  # (rank, rank)
   ```
   
   **注意**: 需要处理 $A^{(k)} = 0$ 的情况（数值稳定性）。

这样整个计算过程完全向量化，没有 Python for 循环！

---

你觉得这个分析对吗？如果确认，我来更新 Plan。