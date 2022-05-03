import numpy as np
import time

class ILQR_Own():
    def __init__(self):
        self.ref_path = None
        self.pos_param = 5 
        self.u_param = 5

        self.diff = None
    
    def set_ref_path(self, ref_path):
        path = ref_path[-50:]
        sampled = path[1::5]
        self.ref_path = sampled 
    
    def get_cost(self, X, U):

        sample_rate = len(X) / len(self.ref_path)
        X_sampled = X[1::sample_rate]
        X_sampled = X_sampled[:len(self.ref_path)]

        pos_error = np.sum((X_sampled - self.ref_path)**2) 
        u_error = np.sum(U**2)

        cost = self.pos_param*pos_error + self.u_param*u_error

        return cost

    def get_derivatives(self, nominal_states, nominal_controls):
        

        q = 
        Q = 

        r = 2 * nominal_controls
        R = 2 * np.eye(len(nominal_controls))

        S = np.zeros((2,4))

        return q, Q, r, R, S

         

    def forward_pass(self, nominal_states, nominal_controls, K_closed_loop,
                     k_open_loop, alpha):
        # t0 = time.time()
        X = np.zeros_like(nominal_states)
        U = np.zeros_like(nominal_controls)

        X[:, 0] = nominal_states[:, 0]
        for i in range(self.N - 1):
            K = K_closed_loop[:, :, i]
            k = k_open_loop[:, i]
            u = (nominal_controls[:, i] + alpha * k +
                 K @ (X[:, i] - nominal_states[:, i]))
            X[:, i + 1], U[:, i] = self.dynamics.forward_step(X[:, i], u)
        # print("forward pass integration use: ", time.time()-t0)
        
        t1 = time.time()
        J = self.get_cost(X, U)

        return X, U, J

    def backward_pass(self, nominal_states, nominal_controls):
        # t0 = time.time()
        L_x, L_xx, L_u, L_uu, L_ux = self.get_derivatives(
            nominal_states, nominal_controls)
        # print("backward pass derivative use: ", time.time()-t0)
        fx, fu = self.dynamics.get_AB_matrix(nominal_states, nominal_controls)

        k_open_loop = np.zeros((self.dim_u, self.N - 1))
        K_closed_loop = np.zeros((self.dim_u, self.dim_x, self.N - 1))
        # derivative of value function at final step
        V_x = L_x[:, -1]
        V_xx = L_xx[:, :, -1]

        reg_mat = self.lambad * np.eye(self.dim_u)

        Q_u_hist = np.zeros([self.dim_u, self.N - 1])
        Q_uu_hist = np.zeros([self.dim_u, self.dim_u, self.N - 1])
        for i in range(self.N - 2, -1, -1):
            Q_x = L_x[:, i] + fx[:, :, i].T @ V_x
            Q_u = L_u[:, i] + fu[:, :, i].T @ V_x
            Q_xx = L_xx[:, :, i] + fx[:, :, i].T @ V_xx @ fx[:, :, i]
            Q_ux = fu[:, :, i].T @ V_xx @ fx[:, :, i] + L_ux[:, :, i]
            Q_uu = L_uu[:, :, i] + fu[:, :, i].T @ V_xx @ fu[:, :, i]

            Q_uu_inv = np.linalg.inv(Q_uu + reg_mat)
            k_open_loop[:, i] = -Q_uu_inv @ Q_u
            K_closed_loop[:, :, i] = -Q_uu_inv @ Q_ux

            # Update value function derivative for the previous time step
            V_x = Q_x - K_closed_loop[:, :, i].T @ Q_uu @ k_open_loop[:, i]
            V_xx = Q_xx - K_closed_loop[:, :, i].T @ Q_uu @ K_closed_loop[:, :,
                                                                          i]

            Q_u_hist[:, i] = Q_u
            Q_uu_hist[:, :, i] = Q_uu
        # print("backward pass use: ", time.time()-t0)

        return K_closed_loop, k_open_loop

    def solve(self, cur_state, controls=None, obs_list=[], record=False):
        status = 0
        self.lambad = 10

        time0 = time.time()

        if controls is None:
            controls = np.zeros((self.dim_u, self.N))
        states = np.zeros((self.dim_x, self.N))
        states[:, 0] = cur_state

        for i in range(1, self.N):
            states[:,
                   i], _ = self.dynamics.forward_step(states[:, i - 1],
                                                      controls[:, i - 1])

        J = self.get_cost(states, controls)

        converged = False

        # have_not_updated = 0
        for i in range(self.steps):
            K_closed_loop, k_open_loop = self.backward_pass(
                states, controls)
            updated = False
            for alpha in self.alphas:
                X_new, U_new, J_new = (
                    self.forward_pass(states, controls, K_closed_loop,
                                      k_open_loop, alpha))
                if J_new <= J:
                    if np.abs((J - J_new) / J) < self.tol:
                        converged = True
                    J = J_new
                    states = X_new
                    controls = U_new
                    updated = True
                    break
            if updated:
                self.lambad *= 0.7
            else:
                status = 2
                break
            # self.lambad = min(max(self.lambad_min, self.lambad), self.lambad_max)
            self.lambad = max(self.lambad_min, self.lambad)

            if converged:
                status = 1
                break
        t_process = time.time() - time0
        # print("step, ", i, "alpha:", alpha)

        if record:
            # get parameters for FRS
            K_closed_loop, _ = self.backward_pass(states, controls)
            fx, fu = self.dynamics.get_AB_matrix(states, controls)
        else:
            K_closed_loop = None
            fx = None
            fu = None
        return states, controls, t_process, status, K_closed_loop, fx, fu