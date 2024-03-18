import numpy as np
import pickle
import cvxpy as cp
import multiprocessing as mp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from default_args import *

n_process = 100

def generate_opt(defaults, opt_problem, instance_para_list, input_bound, output_bound, paralell=True, solver='gurobi'):
    print(opt_problem, instance_para_list)
    seed = defaults['seed']
    num_var = instance_para_list[0]
    num_ineq = instance_para_list[1]
    num_eq = instance_para_list[2]
    test_size = instance_para_list[3]#defaults['dataSize']
    if opt_problem == 'qp':
        data = make_qp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver)
    elif opt_problem == 'socp':
        data = make_socp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver)
    elif opt_problem == 'convex_qcqp':
        data = make_convex_qcqp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver)
    elif opt_problem == 'sdp':
        data = make_sdp(seed, int(num_var**0.5), num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver)
    elif opt_problem == 'jccim':
        num_scenario = instance_para_list[4]
        data = make_jcc_im(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver, num_scenario)
    else:
        NotImplementedError
    if not os.path.exists('datasets/{}'.format(opt_problem)):
        os.makedirs('datasets/{}'.format(opt_problem))
    with open("datasets/{}/random_{}_{}_dataset_var{}_ineq{}_eq{}_ex{}".format(opt_problem, seed, opt_problem, num_var, num_ineq, num_eq, test_size), 'wb') as f:
        pickle.dump(data, f)

def find_partial_variable(A, num_var, num_eq):
    i = 0
    det_min = 0
    best_partial = 0
    while i < 1000:
        # np.random.seed(i)
        partial_vars_idx = np.random.choice(num_var, num_var - num_eq, replace=False)
        other_vars = np.setdiff1d(np.arange(num_var), partial_vars_idx)
        _, det = np.linalg.slogdet(A[:, other_vars])
        if det>det_min:
            det_min = det
            print('best_det', det_min, end='\r')
            best_partial = partial_vars_idx
        i += 1
    print('best_det', det_min)
    return best_partial

def make_qp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver):
    np.random.seed(seed)
    XL, XU = input_bound
    YL, YU = output_bound
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)
    data = {'Q': Q, 'p': p, 'A': A, 'X': X, 'G': G, 'h': h, 'XL': XL, 'XU': XU, 'YL': YL, 'YU': YU, }
    if paralell:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, p, G, h, YU, YL, A, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_qp, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_qp((i, Xi, num_var, Q, p, G, h, YU, YL, A, solver))
            Y.append(yt)
    data['Y'] = np.array(Y)
    data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data

def solve_qp(args):
    n, Xi, num_var, Q, p, G, h, YU, YL, A, solver = args
    y = cp.Variable(num_var)
    constraints = [G @ y <= h, y <= YU, y >= YL, A @ y == Xi]
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y), constraints)
    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")

    return y.value

def make_socp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver):
    np.random.seed(seed)
    """input-output para"""
    XL = input_bound[0]
    XU = input_bound[1]
    YL = output_bound[0]
    YU = output_bound[1]
    """Obj para"""
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)
    """Eq para"""
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    """Ineq para"""
    x0 = np.random.uniform(-1, 1, size=(num_var))
    G = np.random.uniform(-1, 1, size=(num_ineq, num_ineq, num_var))
    h = np.random.uniform(-1, 1, size=(num_ineq, num_ineq))
    C = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    d = np.linalg.norm(G @ x0 + h, ord=2, axis=1) - C @ x0
    """data set"""
    data = {'Q':Q, 'p':p,
            'A':A, 'X':X,
            'G':np.array(G), 'h': np.array(h), 
            'C': np.array(C), 'd': np.array(d),
            'XL':XL, 'XU':XU,
            'YL':YL, 'YU':YU,
            'Y': []}
    if paralell:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, p, G, h, C, d, YU, YL, A, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_socp, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_socp((i, Xi, num_var, Q, p, G, h, C, d, YU, YL, A, solver))
            Y.append(yt)

    data['Y'] = np.array(Y)
    data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data

def solve_socp(args):
    n, Xi, num_var, Q, p, G, h, C, d, YU, YL, A, solver = args
    y = cp.Variable(num_var)
    soc_constraints = [cp.SOC(C[i].T @ y + d[i], G[i] @ y + h[i]) for i in range(C.shape[0])]
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                      soc_constraints + [A @ y == Xi, y <= YU, y >= YL])

    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")

    return y.value

def make_convex_qcqp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver):
    np.random.seed(seed)
    """input-output para"""
    XL = input_bound[0]
    XU = input_bound[1]
    YL = output_bound[0]
    YU = output_bound[1]
    """Obj para"""
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)
    """Eq para"""
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    """Ineq para"""
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)
    H = np.random.uniform(0, 0.1,  size=(num_ineq, num_var))
    H = [np.diag(H[i]) for i in range(num_ineq)]
    H = np.array(H)
    """data set"""
    data = {'Q':Q, 'p':p,
            'A':A, 'X':X,
            'G':G, 'H':H, 'h':h,
            'XL':XL, 'XU':XU,
            'YL':YL, 'YU':YU,
            'Y': []}
    if paralell:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, p, G, H, h, YU, YL, A, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_qcqp, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_qcqp((i, Xi, num_var, Q, p, G, H, h, YU, YL, A, solver))
            Y.append(yt)

    data['Y'] = np.array(Y)
    data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data

def solve_qcqp(args):
    n, Xi, num_var, Q, p, G, H, h, YU, YL, A, solver = args
    y = cp.Variable(num_var)
    constraints = [0.5 * cp.quad_form(y, H[i]) + G[i].T @ y <= h[i] for i in range(H.shape[0])]
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                      constraints + [A @ y == Xi, y <= YU, y >= YL])

    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")

    return y.value

def make_sdp(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver):
    np.random.seed(seed)
    """input-output para"""
    XL = input_bound[0]
    XU = input_bound[1]
    YL = output_bound[0]
    YU = output_bound[1]
    """Obj para"""
    Q = np.random.uniform(-1,1, size=(num_var, num_var))
    Q = (Q+Q.T)/2
    """Eq para"""
    A = np.random.uniform(-1,1, size=(num_eq, num_var, num_var))
    A = (A + A.transpose((0, 2, 1))) / 2
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    """Ineq para"""
    y0 = np.random.uniform(-1,1, size=(num_var, num_var))
    G = np.random.uniform(-1,1, size=(num_ineq, num_var, num_var))
    G = (G + G.transpose((0, 2, 1))) / 2
    h = np.trace(G@y0, axis1=1, axis2=2)
    """Ineq para"""
    # G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    # h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)
    """data set"""
    data = {'Q':Q,
            'A':A, 'X':X,
            'G':G, 'h':h,
            'XL':XL, 'XU':XU,
            'YL':YL, 'YU':YU,
            'Y': []}
    if paralell:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, G, h, YU, YL, A, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_sdp, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_sdp((i, Xi, num_var, Q, G, h, YU, YL, A, solver))
            Y.append(yt)
    A_extend = [np.tril(At) + np.triu(At, 1).T for At in A]
    A_extend = np.array([At[np.tril_indices(num_var)] for At in A_extend])
    G_extend = [np.tril(Gt) + np.triu(Gt, 1).T for Gt in G]
    G_extend = np.array([Gt[np.tril_indices(num_var)] for Gt in G_extend])
    Y_extend = np.array([Yt[np.tril_indices(num_var)] for Yt in Y])
    data['Y'] = np.array(Y)
    data['A'] = np.array(A)
    data['Ye'] = Y_extend
    data['Ae'] = A_extend
    data['Ge'] = G_extend
    data['best_partial'] = find_partial_variable(A_extend, int(num_var*(num_var+1)/2), num_eq)
    # A_extend = np.array([A[i].flatten() for i in range(num_eq)])
    # data['best_partial'] = find_partial_variable(A_extend, num_var**2, num_eq)
    return data

def solve_sdp(args):
    n, Xi, num_var, Q, G, h, YU, YL, A, solver = args
    y = cp.Variable((num_var, num_var), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.trace(Q @ y)),
                      [cp.trace(A[i] @ y) == Xi[i] for i in range(A.shape[0])] +
                      # [cp.trace(G[i] @ y) <= h[i] for i in range(G.shape[0])] +
                      [y >> 0, y <= YU, y >= YL])
    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0,0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")

    return y.value

def make_jcc_im(seed, num_var, num_ineq, num_eq, test_size, input_bound, output_bound, paralell, solver, num_scenario=100):
    np.random.seed(seed)
    XL, XU = input_bound
    YL, YU = output_bound
    Q = np.diag(np.random.rand(num_var)) * 0.5
    p = np.random.uniform(-1, 1, num_var)
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    # A = np.zeros([num_eq, num_var])
    # for col in range(num_var):
    #     row = np.random.choice(num_eq)
    #     A[row, col] = 1
    X = np.random.uniform(XL, XU, size=(test_size, num_eq))
    W = np.random.rand(num_scenario, num_eq) * 0.1
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    # G = np.zeros([num_ineq, num_var])
    # for col in range(num_var):
    #     row = np.random.choice(num_ineq)
    #     G[row, col] = 1
    h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)
    data = {'Q': Q, 'p': p, 'A': A, 'W': W, 'X': X, 'G': G, 'h': h, 'XL': XL, 'XU': XU, 'YL': YL, 'YU': YU, }
    if paralell:
        with mp.Pool(processes=n_process) as pool:
            params = [(i, Xi, num_var, Q, p, G, h, YU, YL, A, W, solver) for i, Xi in enumerate(X)]
            Y = list(pool.map(solve_jcc_im, params))
    else:
        Y = []
        for i, Xi in enumerate(X):
            yt = solve_jcc_im((i, Xi, num_var, Q, p, G, h, YU, YL, A, W, solver))
            Y.append(yt)
    data['Y'] = np.array(Y)
    # data['best_partial'] = find_partial_variable(A, num_var, num_eq)
    return data

def solve_jcc_im(args):
    n, Xi, num_var, Q, p, G, h, YU, YL, A, W, solver = args
    num_scenario = W.shape[0]
    y = cp.Variable(num_var)
    # z = cp.Variable(num_scenario, boolean=True)
    # - (1 - z[i]) * 1e5
    constraints = [y <= YU, y >= YL]
    constraints += [A @ y >= Xi+W[i] for i in range(num_scenario)]
    constraints += [G @ y <= h]
    # constraints.append(cp.sum(z) / num_scenario >= 0.9)
    prob = cp.Problem(cp.Minimize( p.T @ y), constraints)
    try:
        if solver == 'gurobi':
            prob.solve(solver=cp.GUROBI)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)
        else:
            prob.solve()
        print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    except Exception as e:
        print(f"Error solving problem for n={n}: {e}")
    return y.value

# def __main__():
#     defaults = config()
#     # solver: mosek, gurobi
#     # paras: [num_var, n_ineq, n_eq, n_samples, n_scenario]
#     generate_opt(defaults, 'qp', [400, 100, 100, 10000], [-1,1], [-3, 3], paralell=True, solver='mosek')
#     generate_opt(defaults, 'convex_qcqp', [400, 100, 100, 10000], [-1,1], [-3, 3], paralell=True, solver='mosek')
#     generate_opt(defaults, 'socp', [400, 100, 100, 10000], [-1, 1], [-3, 3], paralell=True, solver='mosek')
#     generate_opt(defaults, 'sdp', [1600, 50, 50, 10000], [-1, 1], [-3, 3], paralell=True, solver='mosek')
#     generate_opt(defaults, 'jccim', [400, 100, 100, 10000, 100], [-1,1], [-3, 3], paralell=True, solver='mosek')

# if __name__ == '__main__':
#     __main__()