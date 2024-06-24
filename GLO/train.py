import os, sys
from pathlib import Path
import argparse
import typing
import math
import random
import json
import logging
from enum import Enum

import pandas as pd
from tqdm import tqdm
import torch

from GLO.utils.exception import get_exception_str, get_traceback
from lib.torch.seed import get_random_state, set_random_state, seed
from lib.torch.safe_save import load_torch, save_torch
from lib.torch.torch_summary import sum_batch
from lib.tools import Logger, timer, smtp
from lib.tools.interfaces import StateDictInterface
from lib.tools.randomizer import Randomizer
from model.rl.explorer import HalfTimeExplorer

from .core.sql_featurizer import database, Sql
from .model.rl import ValueBasedRL
from .model.train import load_dataset, CacheManager, PlanManager, plan_latency

def dict_list_to_dataframe(dicts):
    df = {k: [] for k in dicts[0].keys()}
    for dic in dicts:
        for k, v in dic.items():
            df[k].append(v)
    return pd.DataFrame(df)

def warm_up(dataset : typing.Iterable[Sql], iterations=1):
    data = []
    for i in range(iterations):
        data.extend(dataset)
    random.shuffle(data)
    gen = tqdm(data, desc='Warm up')
    for sql in gen:
        gen.set_postfix({'file': sql.filename})
        database.latency(str(sql), cache=False)

class Trainer(StateDictInterface):
    class history_type(Enum):
        add_memory = 0
        train = 1
        validate = 2
        set_sample_weight = 3

    def __init__(
        self,
        train_dataset : typing.Iterable[Sql],
        test_dataset : typing.Iterable[Sql],
        device,
        checkpoint_file=None,
        load_checkpoint=True,
    ):
        self.__device = device
        self.train_dataset = list(train_dataset)
        self.test_dataset = list(test_dataset)
        self.reset()

        default_checkpoint_file = Path(database.config.path_checkpoints) / f'{database.config.experiment_id}.checkpoint.pkl'
        if checkpoint_file is None or not os.path.isfile(checkpoint_file):
            self.checkpoint_file = default_checkpoint_file
        else:
            self.checkpoint_file = Path(checkpoint_file)

        if load_checkpoint and os.path.isfile(self.checkpoint_file):
            self.load_checkpoint(self.checkpoint_file)
        else:
            self.init()

    @property
    def device(self):
        return self.__device

    def reset(self):
        seed(database.config.seed)
        self.experiment_id = database.config.experiment_id
        log_file = database.config.log_file_name.format(self.experiment_id)
        log_level = logging.DEBUG if database.config.log_debug else logging.INFO
        self.log = Logger(database.config.log_name, level=log_level, file=log_file, to_stderr=False, to_stdout=True)
        self.log.format('[%(levelname)s %(asctime)s.%(msecs)d] %(message)s', '%Y-%m-%d %H:%M:%S')

        self.randomizers = {
            'sample': Randomizer(random.getrandbits(64)),
            'exploration': Randomizer(random.getrandbits(64)),
        }
        self.baseline_explorer = HalfTimeExplorer(0.5, 0.2, 80)
        self.rl_model = ValueBasedRL(
            self.__device,
            num_table_layers=database.config.extractor_num_table_layers,
            initial_exploration_prob=database.config.rl_initial_exploration_prob,
        )
        self.baseline_plan_manager = PlanManager()
        self.plan_manager = PlanManager()
        self.cache_manager = CacheManager(database.config.cache_file_name.format(self.experiment_id), auto_save_interval=database.config.cache_save_interval)

        self.epoch_history = None
        self.history = []
        self.sample_weights = None
        self.prob_bias = None
        self.validate_results = []
        self.time_records = []
        self.epoch = 0
        self.bushy = None
        self.initialized = False

        self.register('rl_model')
        self.register('plan_manager')
        self.register('baseline_plan_manager')
        self.register('sample_weights')
        self.register('prob_bias')
        self.register('baseline_explorer')
        self.register('validate_results')
        self.register('time_records')
        self.register('history')
        self.register('initialized')

    def load_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            checkpoint_file = self.checkpoint_file
        self.log(f'Loading checkpoint from {checkpoint_file}')
        dic = load_torch(checkpoint_file)
        self.load_state_dict(dic)
        self.log('Loaded')

    def init(self):
        self.log('Initializing models')
        self.baseline_plan_manager.init(
            (*self.train_dataset, *self.test_dataset),
            self.cache_manager,
            set_timeout=False,
            verbose=True,
        )
        self.plan_manager.init(
            self.train_dataset,
            self.cache_manager,
            set_timeout=True,
            verbose=True,
        )
        self.cache_manager.flush()

        self.log('Adding initial experiences')
        preprocess_plans = []
        sql_graphs = [sql.graph for sql in self.train_dataset]
        if sql_graphs:
            self.rl_model.model_extractor.fit_norm(sql_graphs)

        for sql in self.train_dataset:
            plan_results = self.plan_manager.get(sql, detail=True)
            if plan_results is None or plan_results['actions'] is None:
                continue
            best_order, best_value = plan_results['actions'], plan_results['value']
            with torch.no_grad():
                plan_set = []
                plan = self.rl_model.plan_init(sql)
                while not plan.completed:
                    action = best_order[plan.total_branch_nodes]
                    self.rl_model.step(plan, action)
                    self.rl_model.add_memory(plan.clone(), best_value)
                    plan_set.append(plan.clone())
                    if database.config.transformer_node_attrs_preprocess_zscore in ('all',):
                        preprocess_plans.append(plan.clone())
            self.rl_model.add_pair_memory(plan.clone(), best_value)
            self.rl_model.add_priority_memory(plan_set, best_value)

            if database.config.transformer_node_attrs_preprocess_zscore in ('all', 'completed'):
                preprocess_plans.append(plan.clone())

        if preprocess_plans:
            self.log('Calculating mean and std for Z-score normalization')
            prep_features = [plan.to_sequence() for plan in preprocess_plans]
            prep_node_attrs = [features['node_attr'][features['node_attr_mask'] != 0] for features in prep_features]
            prep_node_attrs = torch.cat(prep_node_attrs, dim=0)[..., :3]
            if torch.numel(prep_node_attrs) == 0:
                prep_node_attrs = torch.zeros(1, 3, device=self.__device)
            if not database.config.use_lstm:
                self.rl_model.model_transformer.node_attr_zscore.fit(prep_node_attrs)

        initial_checkpoint = Path(database.config.path_checkpoints) / f'{database.config.experiment_id}.init.pkl'
        self.log(f'Saving initial checkpoint to {initial_checkpoint}')
        save_torch(self.state_dict(), initial_checkpoint)
        self.log('Saved')

        self.initialized = True

    def state_dict(self):
        res = super().state_dict()
        res.update({
            'epoch': self.epoch,
            'timeout_limit': database.timeout,
            'random_state': get_random_state(),
            'randomizers': {k : v.state_dict() for k, v in self.randomizers.items()},
        })
        return res

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            self.rl_model.schedule(self.epoch)
        if 'timeout_limit' in state_dict:
            database.timeout = state_dict['timeout_limit']
        if 'random_state' in state_dict:
            set_random_state(state_dict['random_state'])
        if 'randomizers' in state_dict:
            for k, v in state_dict['randomizers'].items():
                if k not in self.randomizers:
                    self.randomizers[k] = Randomizer(0)
                self.randomizers[k].load_state_dict(v)

    def send_email(self, title, message, desc=None):
        if database.config.email is not None and database.config.email_password is not None:
            self.log(f'Sending email as {database.config.email} with title {title}')
            title = f'{database.config.email_product_name} | {title}'
            if desc is None:
                from_ = None
            else:
                from_ = title
                title = desc
            html_args = {
                'product_name': database.config.email_product_name,
                'message': message,
            }
            receiver = database.config.email_receiver if database.config.email_receiver else None
            with smtp.login(database.config.email, database.config.email_password, ssl=True) as m:
                if m is not None:
                    success = smtp.mail(m, smtp.mime_from_file(title, 'html/remind.html', replace=html_args, from_=from_), receiver=receiver)
                else:
                    success = False
            if not success:
                self.log(f'Failed to send mail as {database.config.email}', level=logging.WARNING)
            else:
                self.log(f'Sent email as {database.config.email}')

    def _train(self, train_method, validate_method, save_method, after_validation=None):
        total_epochs = database.config.epochs
        for epoch in range(self.epoch, total_epochs):
            self.epoch_history = []
            self.bushy = database.config.bushy and epoch >= database.config.bushy_start_epoch
            train_method()
            epoch = epoch + 1
            if epoch > database.config.validate_start and epoch % database.config.validate_interval == 0:
                validate_method()
            self.cache_manager.flush()
            self.rl_model.schedule()
            if callable(after_validation):
                after_validation()
            self.epoch = epoch
            self.history.append(self.epoch_history)
            save_method()

    def save_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            checkpoint_file = self.checkpoint_file
        self.log(f'Saving checkpoint to {checkpoint_file}')
        res = save_torch(self.state_dict(), checkpoint_file)
        self.log('Saved')
        return res

    def train(self):
        def save_checkpoint():
            if self.epoch % database.config.save_checkpoint_interval == 0:
                self.save_checkpoint()
        return self._train(self._train_one_epoch, self._validate, save_checkpoint, self._train_after_validation)

    def _validate_dataset(self, dataset : typing.Iterable[Sql]):
        dataset = tuple(dataset)
        if len(dataset) == 0:
            raise RuntimeError('empty dataset')

        self.rl_model.eval_mode()
        res = []
        with torch.no_grad():
            gen = tqdm(dataset)
            for sql in gen:
                with timer:
                    plan_result = self.rl_model.get_plan(sql, bushy=True, beam_size=database.config.beam_width, detail=True)
                    plan = plan_result['result']
                    detail_time = plan_result['time']
                inference_time = timer.time * 1000
                detailed_execution_time = plan_latency(plan, self.cache_manager, detail=True)
                origin_execution_time = plan_latency(sql, self.cache_manager, detail=True)

                relative_latency = detailed_execution_time['latency'] / origin_execution_time['latency']
                timer_relative_latency = detailed_execution_time['timer_latency'] / origin_execution_time['timer_latency']

                origin_plan_order = self.baseline_plan_manager.get(sql)
                if origin_plan_order is None:
                    comparison_result = None
                    comparison_success = None
                    prediction = self.rl_model.predict_plan(plan, detail=True, gt=detailed_execution_time['latency'])
                    loss, origin_loss = prediction['loss'], None
                else:
                    origin_plan = self.rl_model.plan_init(sql)
                    while not origin_plan.completed:
                        self.rl_model.step(origin_plan, origin_plan_order[origin_plan.total_branch_nodes])
                    prediction = self.rl_model.compare_plan(
                        plan,
                        origin_plan,
                        detail=True,
                        gt1=detailed_execution_time['latency'],
                        gt2=origin_execution_time['latency'],
                    )
                    comparison_result = prediction['comparison']
                    loss, origin_loss = prediction['loss1'], prediction['loss2']
                    comparison_success = None if comparison_result == 0 else ((relative_latency < 1) ^ (comparison_result > 0))

                res.append({
                    'file': sql.filename,
                    'latency': detailed_execution_time['latency'],
                    'baseline_latency': origin_execution_time['latency'],
                    'relative_latency': relative_latency,
                    'loss': loss,
                    'baseline_loss': origin_loss,
                    'timer_latency': detailed_execution_time['timer_latency'],
                    'baseline_timer_latency': origin_execution_time['timer_latency'],
                    'timer_relative_latency': timer_relative_latency,
                    'inference_time': inference_time,
                    'comparison_result': comparison_result,
                    'comparison_success': comparison_success,
                    'comparison_latency': detailed_execution_time['latency'] if comparison_result <= 0 else origin_execution_time['latency'],
                    'baseline': origin_plan.hints(),
                    'plan': plan.hints(),
                    'time_init': detail_time['init'],
                    'time_gpu': detail_time['gpu'],
                    'time_explain': detail_time['explain'],
                    'time_to_sequence': detail_time['to_sequence'],
                    'time_join': detail_time['join'],
                    'time_other_cpu': detail_time['total'] - detail_time['init'] - detail_time['gpu'] - detail_time['explain'] - detail_time['to_sequence'] - detail_time['join'],
                })
                gen.set_postfix({
                    'relative': relative_latency,
                })
        return dict_list_to_dataframe(res)

    def _validate_df_collect_data(self, df):
        gmrl = math.exp(df['relative_latency'].map(math.log).sum() / len(df))
        gmrl_comparison = math.exp(
            (df['comparison_latency'] / df['baseline_latency']).map(math.log).sum() / len(df))
        speedup = df['baseline_latency'].sum() / df['latency'].sum()
        speedup_comparison = df['baseline_latency'].sum() / df['comparison_latency'].sum()
        comparison_accuracy = df['comparison_success'].astype(float).dropna().mean()
        return {
            'gmrl': gmrl,
            'gmrl_comparison': gmrl_comparison,
            'speedup': speedup,
            'speedup_comparison': speedup_comparison,
            'comparison_accuracy': comparison_accuracy,
        }

    def _to_email_df(self, df : pd.DataFrame, rename_mapping : dict):
        rename_mapping = tuple((k, rename_mapping[k]) for k in filter(lambda x: x in rename_mapping, df.keys()))
        new_df = df[[*tuple(zip(*rename_mapping))[0]]].rename({k : v for k, v in rename_mapping}, axis=1)
        new_df = new_df.applymap(lambda x: round(x, 3) if isinstance(x, float) else x)
        return new_df

    def _validate(self, add_history=True, save_path=None, prefix=None):
        if save_path is None:
            save_path = database.config.path_results
        if prefix is not None:
            prefix = f'{prefix}.'
        else:
            prefix = ''

        if add_history:
            self.epoch_history.append((
                self.history_type.validate,
                (),
            ))

        path_results = Path(save_path) / f'{prefix}{self.experiment_id}'
        os.makedirs(path_results, exist_ok=True)

        self.log('Validating with test dataset')
        _df_test = self._validate_dataset(self.test_dataset)
        df_test = _df_test.sort_values(by='file')
        test_res = self._validate_df_collect_data(df_test)
        self.log(f"Test set GMRL: {test_res['gmrl']:.03f} / {test_res['gmrl_comparison']:.03f} speedup: {test_res['speedup']:.03f} / {test_res['speedup_comparison']:.03f}")
        self.log(f"Test set comparison accuracy: {100 * test_res['comparison_accuracy']:.03f}%")
        df_test.to_csv(path_results / f'test.{self.epoch:03d}.csv', index=False)

        self.log('Validating with train dataset')
        _df_train = self._validate_dataset(self.train_dataset)
        df_train = _df_train.sort_values(by='file')
        train_res = self._validate_df_collect_data(df_train)
        self.log(f"Train set GMRL: {train_res['gmrl']:.03f} / {train_res['gmrl_comparison']:.03f} speedup: {train_res['speedup']:.03f} / {train_res['speedup_comparison']:.03f}")
        self.log(f"Train set comparison accuracy: {100 * train_res['comparison_accuracy']:.03f}%")
        df_train.to_csv(path_results / f'train.{self.epoch:03d}.csv', index=False)

        if self.epoch >= 10:
            _raw_tops = _df_train['latency'].nlargest(max(round(len(_df_train) * 0.125), 1)).index.astype(int).tolist()
            raw_to_total_portion = [0 for i in range(len(_df_train))]
            for index in _raw_tops:
                raw_to_total_portion[index] += 1 / len(_raw_tops)
            self.prob_bias = list(map(lambda x: min(max(x - train_res['gmrl'], 0), 200.0), _df_train['relative_latency']))
            _total_weight = sum(self.prob_bias)
            alpha_absolute = database.config.resample_alpha
            self.sample_weights = [
                (1 - alpha_absolute) * (rel / _total_weight) + alpha_absolute * por
                for rel, por in zip(self.prob_bias, raw_to_total_portion)
            ]
        else:
            self.sample_weights = None

        validate_res = {
            'epoch': self.epoch,
            **{'test_' + k : v for k, v in test_res.items()},
            **{'train_' + k: v for k, v in train_res.items()},
        }
        self.validate_results.append(validate_res)
        validate_df = dict_list_to_dataframe(self.validate_results)
        validate_df.to_csv(path_results / f'results.csv', index=False)

        rename_mapping = {
            'file': 'file',
            'latency': 'time',
            'baseline_latency': 'ori',
            'relative_latency': 'rel',
            'loss': 'loss',
            'baseline_loss': 'oloss',
            'comparison_result': 'cmp',
            'comparison_success': 'correct?',
            'baseline': 'base',
            'plan': 'plan',
        }

        validate_df_rename_mapping = {
            'epoch': 'epoch',
            'test_gmrl': 'GMRL',
            'test_gmrl_comparison': 'GMRL (cmp)',
            'test_speedup': 'Speed',
            'test_speedup_comparison': 'Speed (cmp)',
            'test_comparison_accuracy': 'Acc',
            'train_gmrl': 'tGMRL',
            'train_gmrl_comparison': 'tGMRL (cmp)',
            'train_speedup': 'tSpeed',
            'train_speedup_comparison': 'tSpeed (cmp)',
            'train_comparison_accuracy': 'tAcc',
        }

        self.send_email(
            f'{prefix}{self.experiment_id}',
            f"""
<span>Test set speedup: {test_res['speedup']:.03f} / {test_res['speedup_comparison']:.03f}; GMRL: {test_res['gmrl']:.03f} / {test_res['gmrl_comparison']:.03f}; comparison accuracy: {100 * test_res['comparison_accuracy']:.03f}%</span><br />
<span>Train set speedup: {train_res['speedup']:.03f} / {train_res['speedup_comparison']:.03f}; GMRL: {train_res['gmrl']:.03f} / {train_res['gmrl_comparison']:.03f}; comparison accuracy: {100 * train_res['comparison_accuracy']:.03f}%</span><br />
<hr />
<h3>All results</h3>
{self._to_email_df(validate_df, validate_df_rename_mapping).to_html(index=False)}
<h3>Test dataset</h3>
{self._to_email_df(df_test, rename_mapping).to_html(index=False)}
<h3>Train dataset</h3>
{self._to_email_df(df_train, rename_mapping).to_html(index=False)}
""",
            f"Epoch {self.epoch:03d} ({test_res['speedup_comparison']:.02f}, {test_res['gmrl_comparison']:.02f}, {100 * test_res['comparison_accuracy']:.02f}%)",
        )

        if add_history:
            self.epoch_history.append((
                self.history_type.set_sample_weight,
                (self.sample_weights, )
            ))

    class _plan_search_res:
        def __init__(self, state, action=None, parent_state=None, parent_index=None, value=None, is_exploration=False):
            self.state = state
            self.action = action
            self.value = value
            self.parent_state = parent_state
            self.parent_index = parent_index
            self.is_exploration = is_exploration

    def _sample_dataset(self, dataset, sample_weights=None, prob_bias=None):
        if prob_bias is None:
            dataset = list(map(lambda x: (x, None), dataset))
        else:
            dataset = list(zip(dataset, prob_bias))
        if self.epoch >= database.config.resample_start_epoch and self.epoch % database.config.resample_interval == 0:
            resample_mode = database.config.resample_mode
            if resample_mode == 'replace':
                if len(dataset) > database.config.resample_count:
                    new_train_dataset = self.randomizers['sample'].sample(dataset, k=len(dataset) - database.config.resample_count)
                else:
                    new_train_dataset = []
                new_train_dataset.extend(self.randomizers['sample'].choices(dataset, weights=sample_weights, k=database.config.resample_count))
            elif resample_mode == 'augment':
                dataset = [*dataset, *self.randomizers['sample'].choices(dataset, weights=sample_weights, k=database.config.resample_count)]
            else:
                # augment_deterministic
                if sample_weights is not None:
                    train_dataset_with_weights = sorted(dataset, key=lambda x: x[1], reverse=True)
                    dataset = [*dataset, *train_dataset_with_weights[:database.config.resample_count]]
        self.randomizers['sample'].shuffle(dataset)
        return dataset

    def _train_search_plan(
        self,
        sql,
        beam_size=1,
        use_best=False,
        bushy=True,
        prob_bias=None,
    ):
        with torch.no_grad():
            new_exploration_probs = []
            best_order_detail = self.plan_manager.get(sql, detail=True)
            use_best = use_best and best_order_detail and (bushy or best_order_detail['is_left_deep']) \
                and best_order_detail['actions'] is not None
            if use_best:
                best_order = best_order_detail['actions']
            else:
                best_order = None
            best_prev_value = best_order_detail['value'] if best_order_detail else None

            init_state = self.rl_model.plan_init(sql)
            search_plans = [self._plan_search_res(init_state, None, None, None, None, False)]
            search_paths = []
            display_records = []
            while not search_plans[0].state.completed:
                step = search_plans[0].state.total_branch_nodes
                search_paths.append(search_plans)

                if use_best:
                    parent_state = search_plans[0].state
                    state = parent_state.clone()
                    self.rl_model.step(state, best_order[step])
                    search_plans = [self._plan_search_res(state, best_order[step], parent_state, 0, None, False)]

                    if self.randomizers['exploration'].random() < (1 - 0.4 ** (1 / max(len(state.sql.aliases), 6))):
                        use_best = False
                    display_records.append('b')
                else:
                    search_results = self.rl_model.search(
                        map(lambda x: (x.state, x.is_exploration), search_plans),
                        bushy=bushy,
                        beam_size=beam_size,
                        exploration=True,
                        exploration_bias=prob_bias,
                        detail=True,
                    )
                    new_exploration_probs.append(search_results['exploration_prob'])
                    exploration_branches = search_results['exploration_branches']
                    if exploration_branches > 0:
                        if exploration_branches >= 10:
                            display_records.append(f"({exploration_branches})")
                        else:
                            display_records.append(str(exploration_branches))
                    else:
                        display_records.append('p')

                    search_plans = []
                    for branch_index, (parent_index, parent_state, action) in enumerate(search_results['result']):
                        is_exploration = branch_index >= beam_size - exploration_branches
                        state = parent_state.clone()
                        self.rl_model.step(state, action)
                        search_plans.append(
                            self._plan_search_res(state, action, parent_state, parent_index, None, is_exploration))
            search_paths.append(search_plans)
        return {
            'paths': search_paths,
            'use_best': use_best,
            'prev_value': best_prev_value,
            'display_records': ''.join(display_records),
            'exploration_prob': sum(new_exploration_probs) / len(new_exploration_probs) if new_exploration_probs else self.rl_model.explorer.prob,
        }

    def _train_assign_values(
        self,
        sql,
        search_paths,
        best_prev_value=None,
    ):
        origin_value = plan_latency(sql, self.cache_manager)
        display_state_values = []
        res = {
            'history': [],
            'memory': [],
            'pair_memory': [],
            'priority_memory': [],
        }
        for this_index, search_res in enumerate(search_paths[-1]):
            if best_prev_value is None:
                timeout_tolerance = origin_value * 2.5
            else:
                timeout_tolerance = best_prev_value * 2.5
            timeout = database.timeout
            database.timeout = max(round(timeout_tolerance), 10000)
            state_value = plan_latency(search_res.state, self.cache_manager)
            database.timeout = timeout

            res['history'].append((search_res.state, state_value))
            display_state_values.append(state_value)

            search_res.value = state_value
            res['pair_memory'].append((search_res.state, state_value))

            actions = [search_res.action]
            plan_set = [search_res.parent_state]
            parent_index = search_res.parent_index
            i = 0
            while parent_index is not None:
                parent_list = search_paths[-2 - i]
                parent = parent_list[parent_index]
                parent.value = min(parent.value, state_value) if parent.value is not None else state_value
                parent_index = parent.parent_index
                if parent_index is not None:
                    actions.append(parent.action)
                    plan_set.append(parent.parent_state)
                i += 1
            res['priority_memory'].append((plan_set, state_value))

            if state_value < best_prev_value:
                self.plan_manager.update(sql, reversed(actions), state_value)

        for search_plans in search_paths:
            for search_res in search_plans:
                if search_res.value is None or search_res.action is None:
                    continue
                res['memory'].append((search_res.state, search_res.value))

        res['rc'] = sum(display_state_values) / origin_value / len(display_state_values)
        return res

    def _train_one_epoch(self):
        self.rl_model.train_mode()

        train_dataset = self._sample_dataset(self.train_dataset, self.sample_weights, self.prob_bias)

        postfix = dict(loss='/')
        gen = tqdm(train_dataset, desc=f'Epoch {self.epoch:03d}', postfix=postfix)

        bushy = database.config.bushy and self.epoch >= database.config.bushy_start_epoch
        time_dicts = []
        for sql, sql_prob_bias in gen:
            with timer:
                use_best = self.baseline_explorer.explore()
                if self.epoch < 10:
                    beam_size = 1 + self.randomizers['exploration'].randrange(database.config.beam_width)
                else:
                    beam_size = database.config.beam_width
                search_plans_dict = self._train_search_plan(sql, beam_size, use_best, bushy, sql_prob_bias)
                search_paths = search_plans_dict['paths']
                best_prev_value = search_plans_dict['prev_value']
                display_records = search_plans_dict['display_records']
                new_exploration_prob = search_plans_dict['exploration_prob']
                if search_plans_dict['use_best']:
                    self.baseline_explorer.step()
            time_plan_search = timer.time

            with timer:
                res = self._train_assign_values(sql, search_paths, best_prev_value)
                for state, value in res['history']:
                    self.epoch_history.append((
                        self.history_type.add_memory,
                        (state, value),
                    ))
                for state, value in res['pair_memory']:
                    self.rl_model.add_pair_memory(state, value)
                for plan_set, value in res['priority_memory']:
                    self.rl_model.add_priority_memory(plan_set, value)
                for state, value in res['memory']:
                    self.rl_model.add_memory(state, value)
                rc = res['rc']
            time_database_execution = timer.time

            with timer:
                _backward_times = 3
                if self.epoch < 8:
                    _backward_times = 1
                elif self.epoch < 16:
                    _backward_times = 2

                loss_detail = self.rl_model.train(_backward_times, epoch=self.epoch)
                self.epoch_history.append((
                    self.history_type.train,
                    (_backward_times, )
                ))
            time_train = timer.time

            postfix.update(
                #t_s=time_plan_search,
                #t_e=time_database_execution,
                #t_t=time_train,
                #t_b=loss_detail['time']['batch_update'],
                #t_sq=loss_detail['time']['to_sequence'],
                #t_p=loss_detail['time']['predict'],
                path=display_records,
                lr=self.rl_model.optim.param_groups[0]['lr'],
                pm=new_exploration_prob,
                pb=self.baseline_explorer.prob,
                rc=rc,
                loss=loss_detail['loss']['main'],
                l_rank=loss_detail['loss']['ranking'],
                l_fix=loss_detail['loss']['complementary'],
            )
            gen.set_postfix(postfix)

            time_dicts.append({
                'plan_search': time_plan_search,
                'database_execution': time_database_execution,
                'train': time_train,
                **loss_detail['time'],
            })
        time_dict = {
            'epoch': self.epoch,
            **sum_batch(time_dicts),
        }
        self.time_records.append(time_dict)
        time_record_df = dict_list_to_dataframe(self.time_records)
        path_results = Path(database.config.path_results) / self.experiment_id
        os.makedirs(path_results, exist_ok=True)
        time_record_df.to_csv(path_results / f'time.csv', index=False)

    def _train_after_validation(self):
        if (database.config.epoch_parameter_reset_interval is not None
            and database.config.epoch_parameter_reset_interval > 0):
            if (
                (database.config.epoch_parameter_reset_start_epoch is None
                    or self.epoch >= database.config.epoch_parameter_reset_start_epoch)
                and (self.epoch + 1) < database.config.epochs
                and (self.epoch + 1) % database.config.epoch_parameter_reset_interval == 0
            ):
                self.rl_model.reset_parameters()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', nargs=2, type=str, default=['dataset/train_tpcds', 'dataset/test_tpcds'],
                        help='Training and testing dataset.')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='Total epochs.')
    parser.add_argument('-F', '--id', type=str, default=None,
                        help='File ID.')
    parser.add_argument('--warm-up', type=int, default=None,
                        help='To warm up the database with specific iterations.')
    parser.add_argument('-S', '--seed', type=int, default=3407,
                        help='Random seed.')
    parser.add_argument('-D', '--database', type=str, default='database',
                        help='PostgreSQL database.')
    parser.add_argument('-U', '--user', type=str, default='postgres',
                        help='PostgreSQL user.')
    parser.add_argument('-P', '--password', type=str, default=None,
                        help='PostgreSQL user password.')
    parser.add_argument('--port', type=int, default=None,
                        help='PostgreSQL port.')
    parser.add_argument('--host', type=str, default=None,
                        help='PostgreSQL host.')
    parser.add_argument('--reset', action='store_true',
                        help='To ignore existing checkpoints.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    database_args = {'dbname': args.database}
    if args.user is not None:
        database_args['user'] = args.user
    if args.password is not None:
        database_args['password'] = args.password
    if args.port is not None:
        database_args['port'] = args.port
    if args.host is not None:
        database_args['host'] = args.host
    database.setup(**database_args)

    if args.id is not None:
        database.config.experiment_id = args.id
    if args.seed is not None:
        database.config.seed = args.seed
    if args.epochs is not None:
        database.config.epochs = args.epochs

    train_path, test_path = args.dataset

    print('Generating train set', file=sys.stderr)
    train_set = load_dataset(train_path, device=device, verbose=True)
    print('Generating test set', file=sys.stderr)
    test_set = load_dataset(test_path, device=device, verbose=True)

    if args.warm_up is not None:
        seed(database.config.seed)
        warm_up([*train_set, *test_set], iterations=args.warm_up)

    config_output_file = database.config.config_file_name.format(database.config.experiment_id)
    database.config.postgresql_shared_buffer_size = database.get_settings('shared_buffers')
    database.config.postgresql_server_version = database.get_settings('server_version')
    database.config.postgresql_max_parallel_workers = database.get_settings('max_parallel_workers')
    database.config.postgresql_max_parallel_workers_per_gather = database.get_settings('max_parallel_workers_per_gather')

    os.makedirs(os.path.dirname(config_output_file), exist_ok=True)
    with open(config_output_file, 'w') as f:
        json.dump({
            'id': database.config.experiment_id,
            'config': database.config.to_dict(),
            'args': args.__dict__,
        }, f, ensure_ascii=False, indent=4)

    try:
        trainer = Trainer(train_set, test_set, device, load_checkpoint=not args.reset)
        trainer.train()
    except Exception as e:
        if database.config.email is not None and database.config.email_password is not None:
            with smtp.login(database.config.email, database.config.email_password, ssl=True) as m:
                if m is not None:
                    from_ = f'{database.config.email_product_name} | {database.config.experiment_id}'
                    title = get_exception_str(e)
                    html_args = {
                        'product_name': database.config.email_product_name,
                        'message': get_traceback(e),
                    }
                    receiver = database.config.email_receiver if database.config.email_receiver else None
                    success = smtp.mail(m, smtp.mime_from_file(title, 'html/remind.html', replace=html_args, from_=from_), receiver=receiver)
                else:
                    success = False
            if not success:
                print(f'Failed to send mail as {database.config.email}', file=sys.stderr)
        raise

if __name__ == '__main__':
    main()
