class Config:
    feature_size = 64
    feature_length = 2
    feature_extra_length = 7

    use_lstm = False

    max_sequence_length = 36
    plan_use_node_attrs = True
    plan_use_leaf_node_attrs = True
    plan_use_leaf_node_attrs_with_sql_info = True
    plan_use_branch_node_attrs = True
    plan_relative_dist_maximum = 8
    plan_use_height = False
    plan_height_maximum = 8
    plan_aggr_node_bidirectional = False
    plan_aggr_node_link_to_root_nodes_only = False
    plan_aggr_node_relative_dist_type = 'height'
    plan_table_unordered = True

    plan_features_on_cuda = False
    plan_explain_use_partial_plan = True
    plan_explain_use_cache = True
    plan_explain_use_asyncio = False

    topk_explore_prob_coef = 0.3
    topk_explore_lambda = 2.0

    extractor_num_table_layers = 1
    extractor_autoencoder_attr_size = 3
    extractor_use_autoencoder = True
    extractor_use_normalization = True
    extractor_use_no_gat = False

    transformer_hidden_size = 192
    transformer_attention_heads = 8
    transformer_fc_dropout_rate = 0.1
    transformer_attention_dropout_rate = 0.1
    transformer_attention_layers = 6
    transformer_attention_dropout_until_epoch = None

    transformer_use_height = True
    transformer_relative_dist_maximum = 8
    transformer_super_node_height = True
    transformer_super_node_link_to_root_nodes_only = False
    transformer_link_to_super_node = True
    # none, dist, height, dist_different
    transformer_super_node_relative_dist_bias_type = 'height'
    # batch, layer
    transformer_node_attrs_preprocess_norm_type = 'batch'
    # none, completed, all
    transformer_node_attrs_preprocess_zscore = 'all'

    rl_initial_exploration_prob = 0.8
    rl_reward_weighting = 0.1
    rl_sample_weighting_exponent = 0.0
    rl_use_ranking_loss = True
    rl_ranking_loss_start_epoch = 8
    rl_use_complementary_loss = True
    rl_complementary_loss_equal_suppress = 0
    rl_use_other_states = True
    rl_ranking_loss_weight = 0.2
    rl_complementary_loss_weight = 0.2
    rl_grad_clip_ratio = 1.0

    adam_weight_decay = 0.

    epochs = 200
    epoch_parameter_reset_start_epoch = 20
    epoch_parameter_reset_interval = None
    batch_size = 128
    sql_timeout_limit = 4
    bushy = True
    bushy_start_epoch = 4
    beam_width = 4
    priority_memory_ratio = 0.125
    priority_memory_start_epoch = 16
    memory_size = 10000000
    validate_start = 0
    validate_interval = 4
    save_checkpoint_interval = 4

    resample_weight_cap = (0.5, 2)
    resample_mode = 'augment'
    resample_start_epoch = 10
    resample_alpha = 0.2
    resample_interval = 1
    resample_count = 0

    path_checkpoints = 'checkpoints'
    path_results = 'results'
    config_file_name = 'checkpoints/config_{}.txt'
    checkpoint_file_name = '{}.checkpoint.pkl'
    cache_file_name = 'checkpoints/cache_{}.pkl'
    log_name = 'GLO'
    log_file_name = 'log/{}.log'
    log_debug = True
    cache_save_interval = 400

    experiment_id = '(default)'
    # automatically send experiment results to the receiver account
    email = None
    email_password = None
    email_receiver = None
    email_product_name = 'GLO'
    seed = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        res = {}
        for attr in dir(self):
            value = getattr(self, attr)
            if not callable(value):
                res[attr] = value
        return res
