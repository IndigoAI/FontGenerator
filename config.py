PARAMS = {
    'Attr2Font': {
        'gen_params': {
            'in_channels': 3,
            'style_out': 256,
            'out_channels': 3,
            'n_attr': 37,
            'attention': True
        },
        'discr_params': {
            'in_channels': 3,
            'attr_channels': 37,
            'return_attr': True
        },
        'optim_params': {
            'lr': 2e-4,
            'beta1': 0.5,
            'beta2': 0.99
        },
        'lambds': {
            'lambd_adv': 5,
            'lambd_pixel': 50,
            'lambd_char': 3,
            'lamdb_cx': 6,
            'lamdb_attr': 20
        }
    },
    'StarGAN': {
        'gen_params': {
            'c_dim': 37,
            'n_res': 6
        },
        'discr_params': {
            'img_size': 64,
            'c_dim': 37,
            'n_hidden': 5
        },
        'optim_params': {
            'lr': 1e-4,
            'beta1': 0.5,
            'beta2': 0.99,
            'step_size': 1,
            'gamma': 0.8
        },
        'lambds': {
            'lambda_clf': 1,
            'lambda_gp': 10,
            'lambda_rec': 10
        }
    }
}