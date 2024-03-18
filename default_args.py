def config():
    defaults = {}
    defaults['predType'] = ['NN', 'NN_Eq'][1]
    defaults['projType'] = ['WS', 'Proj', 'D_Proj', 'G_Bis', 'H_Bis'][4]
    defaults['probType'] = ['qp', 'socp', 'convex_qcqp', 'sdp', 'acopf'][4]
    defaults['probSize'] = [[100, 50, 50, 10000],
                            [200, 100, 100, 20000],
                            [100, 10, 10, 10000],
                            [400, 20, 20, 20000]][2]
    defaults['opfSize'] = [[30,  10000], [118, 20000]][1]
    defaults['testSize'] = 1024
    defaults['saveAllStats'] = False
    defaults['resultsSaveFreq'] = 1000
    defaults['seed'] = 2023

    defaults['mapping_para'] = \
        {'training': True, 'testing': False,
        'n_samples': 10000, 'c_samples': 10000,
         'total_iteration': 10000, 'batch_size': 1024,
         'shape': 'square', 'bound': [0, 1], 'scale_ratio': 1,
         'lr': 1e-4, 'lr_decay': 0.9, 'lr_decay_step': 1000,
        'num_layer': 3, 'bilip': True, 'L': 2,
        'penalty_coefficient': 10, 'distortion_coefficient': 1, 'transport_coefficient': 0,
        'testing_samples': 1024}

    defaults['nn_para'] = \
        {'training': True, 'testing': True,
         'approach': 'supervise',
        'total_iteration': 10000, 'batch_size': 256,
        'num_layer': 3, 'lr': 1e-3, 'lr_decay': 0.9, 'lr_decay_step': 1000,
        'objWeight': 0.001, 'softWeightInEqFrac': 0.1, 'softWeightEqFrac': 0.1}

    defaults['proj_para'] = \
        {'useTestCorr': False,    # post-process for infeasible solutions
        'corrMode': 'partial',    # equality completion
        'corrTestMaxSteps': 100,  # steps for D-Proj
        'corrBis': 0.9,           # steps for bisection
        'corrEps': 1e-5,          # tolerance for constraint violation
        'corrLr': 1e-5,           # stepsize for gradient descent in D-Proj
        'corrMomentum': 0.1, }    # momentum parameter in D-Proj

    return defaults

