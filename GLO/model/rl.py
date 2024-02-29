import math
import random
import typing
import logging
import asyncio

import torch
import torch.nn.functional as F
import dgl

from sql import Operators
from model.rl import PriorityMemory, HashPairMemory, BestCache
from model.rl import LinearExplorer
from model.nn import functional as mF
from lib.tools.log import Logger
from lib.tools.timer import timer
from lib.tools.interfaces import StateDictInterface
from lib.torch.torch_summary import sum_batch, dict_map

from ..core.sql_featurizer import database, Sql
from ..core.plan_featurizer import Plan
from .extractor import Extractor
from .transformer import PlanTransformer
from .tree_lstm import TreeLSTM
from .train import CacheManager
from lib.tools.randomizer import Randomizer


class ValueBasedRL(StateDictInterface):
    def __init__(
        self,
        device=torch.device('cpu'),
        num_table_layers=1,
        initial_exploration_prob=0.8,
    ):
        self.__device = device
        self.__train_mode = True
        self.__validating = False

        self._operator_types = [
            Operators.default,
        ]

        self.cache : CacheManager = ...
        self.register(
            'cache',
            CacheManager(),
        )

        self.randomizer : random.Random = ...
        self.register(
            'randomizer',
            Randomizer(random.getrandbits(64)),
        )

        self.memory : PriorityMemory = ...
        self.register(
            'memory',
            PriorityMemory(
                size=database.config.memory_size,
                priority_size=4,
                seed=self.randomizer.getrandbits(64),
            )
        )

        self.pair_memory : HashPairMemory = ...
        self.register(
            'pair_memory',
            HashPairMemory(
                seed=self.randomizer.getrandbits(64),
            ),
        )

        self.best_values : BestCache = ...
        self.register(
            'best_values',
            BestCache(),
        )

        self.explorer : LinearExplorer = ...
        self.register(
            'explorer',
            LinearExplorer(
                start=initial_exploration_prob,
                end=0.2,
                steps=320,
            ),
        )

        self.epoch : int = ...
        self.register('epoch', 0)

        self._num_table_layers = num_table_layers
        self.model_init()
        self.optimizer_init()
        if database.config.use_lstm:
            self.initial_parameters: typing.Dict[str, typing.List[torch.nn.Parameter]] = ...
            self.register(
                'initial_parameters',
                {
                    'model_extractor': self.model_extractor.state_dict(),
                    'model_lstm': self.model_lstm.state_dict(),
                }
            )
        else:
            self.initial_parameters : typing.Dict[str, typing.List[torch.nn.Parameter]] = ...
            self.register(
                'initial_parameters',
                {
                    'model_extractor': self.model_extractor.state_dict(),
                    'model_transformer': self.model_transformer.state_dict(),
                }
            )

        self.to(self.__device)
        self.train_mode(self.__train_mode)

    def model_init(self):
        self.model_extractor : Extractor = ...
        self.register(
            'model_extractor',
            Extractor(
                num_table_layers=self._num_table_layers,
            )
        )

        if database.config.use_lstm:
            self.model_lstm : TreeLSTM = ...
            self.register(
                'model_lstm',
                TreeLSTM(
                    feature_size=database.config.feature_size,
                    input_size=Plan.LSTM_EXTRA_INPUT_SIZE,
                )
            )
        else:
            self.model_transformer : PlanTransformer = ...
            self.register(
                'model_transformer',
                PlanTransformer(
                    emb_size=database.config.feature_size,
                    hidden_size=database.config.transformer_hidden_size,
                    head_size=database.config.transformer_attention_heads,
                    dropout=database.config.transformer_fc_dropout_rate,
                    attention_dropout_rate=database.config.transformer_attention_dropout_rate,
                    n_layers=database.config.transformer_attention_layers,
                    out_dim=1,
                    use_node_attrs=database.config.plan_use_node_attrs,
                    node_attr_norm_type=database.config.transformer_node_attrs_preprocess_norm_type,
                )
            )

    def optimizer_init(self):
        if database.config.use_lstm:
            self.optim: torch.optim.Adam = ...
            self.register(
                'optim',
                torch.optim.Adam(
                    [
                        {'params': self.model_extractor.parameters(), 'lr': 3e-4},
                        {'params': self.model_lstm.parameters(), 'lr': 3e-4},
                    ],
                    weight_decay=database.config.adam_weight_decay,
                )
            )
        else:
            self.optim : torch.optim.Adam = ...
            self.register(
                'optim',
                torch.optim.Adam(
                    [
                        {'params': self.model_extractor.parameters(), 'lr': 3e-4},
                        {'params': self.model_transformer.parameters(), 'lr': 3e-4},
                    ],
                    weight_decay=database.config.adam_weight_decay,
                )
            )
        self.sched : torch.optim.lr_scheduler.LRScheduler = ...
        self.register(
            'sched',
            torch.optim.lr_scheduler.MultiStepLR(
                self.optim,
                [
                    *range(50, 150, 2),
                ],
                gamma=0.1 ** (1 / 50),
            )
        )

    def reset(self):
        self.explorer.reset()
        self.memory.clear()
        self.best_values.clear()
        self.reset_parameters()
        return self

    def reset_parameters(self):
        old_model_extractor = getattr(self, 'model_extractor', None)
        if isinstance(old_model_extractor, Extractor):
            old_model_extractor_zscore = old_model_extractor.table_transform.zscore_table_features.state_dict()
            del old_model_extractor, self.model_extractor
        else:
            old_model_extractor_zscore = None
        if database.config.use_lstm:
            old_model_transformer_zscore = None
            del self.model_lstm
        else:
            old_model_transformer = getattr(self, 'model_transformer', None)
            if isinstance(old_model_transformer, PlanTransformer):
                old_model_transformer_zscore = old_model_transformer.node_attr_zscore.state_dict()
                del old_model_transformer, self.model_transformer
            else:
                old_model_transformer_zscore = None
        old_optimizer = getattr(self, 'optim', None)
        if old_optimizer is not None:
            del old_optimizer, self.optim
        old_scheduler = getattr(self, 'sched', None)
        if old_scheduler is not None:
            del old_scheduler, self.sched

        self.model_init()
        self.optimizer_init()
        if old_model_extractor_zscore:
            self.model_extractor.table_transform.zscore_table_features.load_state_dict(old_model_extractor_zscore)
        if not database.config.use_lstm:
            if old_model_transformer_zscore:
                self.model_transformer.node_attr_zscore.load_state_dict(old_model_transformer_zscore)
        self.to(self.__device)

    def train_mode(self, mode = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.__train_mode = mode
        self.model_extractor.train(mode)
        if database.config.use_lstm:
            self.model_lstm.train(mode)
        else:
            self.model_transformer.train(mode)
        return self

    def eval_mode(self):
        return self.train_mode(False)

    def validate(self, mode = True):
        if not isinstance(mode, bool):
            raise ValueError("validating mode is expected to be boolean")
        self.__validating = mode
        return self

    @property
    def validating(self):
        return self.__validating

    @property
    def device(self):
        return self.__device

    def to(self, device):
        self.__device = device
        self.model_extractor.to(device)
        if database.config.use_lstm:
            self.model_lstm.to(device)
        else:
            self.model_transformer.to(device)
        return self

    def schedule(self, epoch=None):
        self.sched.step(epoch)
        if epoch is None:
            self.epoch += 1
        else:
            self.epoch = epoch

    def _map_explain_info_to_plan(self, plan : Plan):
        if not database.config.plan_use_node_attrs:
            return plan
        if database.config.plan_explain_use_partial_plan:
            plan_sql = plan.to_sql_node(plan.total_branch_nodes - 1)
        else:
            plan_sql = str(plan)
        if database.config.plan_explain_use_cache and not self.validating:
            explain_plan_dic = self.cache.get(plan_sql, None)
            if explain_plan_dic is None:
                explain_plan_dic = database.plan(plan_sql)[0][0][0]
                self.cache.put(plan_sql, explain_plan_dic, hash_key=plan_sql)
        else:
            explain_plan_dic = database.plan(plan_sql)[0][0][0]
        plan.set_node_attrs_from_parsed_plan(explain_plan_dic['Plan'])
        return plan

    def plan_init(self, state : typing.Union[Plan, Sql], grad=False):
        if isinstance(state, Sql):
            state = Plan(state, device=self.device)

        graph, graph_node_indices = state.sql.graph, state.sql.graph_node_indices

        grad_prev = torch.is_grad_enabled()
        torch.set_grad_enabled(grad)
        original_mode = self.__train_mode
        self.eval_mode()
        try:
            graph = self.model_extractor(graph)
        finally:
            self.train_mode(original_mode)
            torch.set_grad_enabled(grad_prev)

        state.set_leaf_embeddings(graph.nodes['table'].data['res'], graph_node_indices)
        return state

    async def _async_map_explain_info_to_plan(self, plan : Plan):
        if not database.config.plan_use_node_attrs:
            return plan

        async def _get_database_plan(sql):
            res = await database.async_plan(sql)
            return res[0][0][0]

        if database.config.plan_explain_use_partial_plan:
            plan_sql = plan.to_sql_node(plan.total_branch_nodes - 1)
        else:
            plan_sql = str(plan)
        if database.config.plan_explain_use_cache and not self.validating:
            explain_plan_dic = self.cache.get(plan_sql, None)
            if explain_plan_dic is None:
                explain_plan_dic = await _get_database_plan(plan_sql)
                self.cache.put(plan_sql, explain_plan_dic, hash_key=plan_sql)
        else:
            explain_plan_dic = await _get_database_plan(plan_sql)
        plan.set_node_attrs_from_parsed_plan(explain_plan_dic['Plan'])
        return plan

    def _async_get_next_state_seqs(self, state, candidates):
        async def get_seq(state, left_alias, right_alias):
            new_state = state.clone()
            for operator_type in self._operator_types:
                new_state.join(left_alias, right_alias, operator_type)
            await self._async_map_explain_info_to_plan(new_state)
            seq = new_state.to_sequence()
            return seq

        async def get_all_seqs(state, candidates):
            sequences = await asyncio.gather(
                *(get_seq(state, left_alias, right_alias) for left_alias, right_alias in candidates)
            )
            return sequences

        return asyncio.run(get_all_seqs(state, candidates))

    def search(
        self,
        states : typing.Iterable[typing.Union[Plan, typing.Tuple[Plan, bool]]],
        bushy : bool = True,
        beam_size : int = 1,
        exploration : bool = False,
        exploration_bias : float = None,
        detail : bool = False,
    ):
        if exploration_bias is None:
            exploration_bias = 0.

        all_candidates = []
        all_res = []
        exploration_mask = []
        times = {
            'gpu': 0.,
            'explain': 0.,
            'to_sequence': 0.,
            'join': 0.,
        }

        for state_index, state in enumerate(states):
            state : Plan
            if isinstance(state, tuple):
                state, is_exploration_branch = state
            else:
                is_exploration_branch = False

            candidates = state.candidates(bushy=bushy, unordered=database.config.plan_table_unordered)
            # to make the results reproducible
            candidates = sorted(candidates, key=str)
            all_candidates.extend(map(lambda x: (state_index, state, x), candidates))

            with torch.no_grad():
                if not database.config.use_lstm and database.config.plan_explain_use_asyncio:
                    with timer:
                        sequences = self._async_get_next_state_seqs(state, candidates)
                    times['explain'] += timer.time
                else:
                    sequences = []
                    new_states = []
                    for left_alias, right_alias in candidates:
                        with timer:
                            new_state = state.clone()
                            new_states.append(new_state)
                            for operator_type in self._operator_types:
                                new_state.join(left_alias, right_alias, operator_type)
                        times['join'] += timer.time

                        with timer:
                            self._map_explain_info_to_plan(new_state)
                        times['explain'] += timer.time

                        if database.config.use_lstm:
                            with timer:
                                last_action = new_state.get_last_action_embeddings()
                            times['to_sequence'] += timer.time
                            sequences.append(last_action)
                        else:
                            with timer:
                                seq = new_state.to_sequence()
                            times['to_sequence'] += timer.time
                            sequences.append(seq)

                if database.config.use_lstm:
                    with timer:
                        new_embeddings = self.model_lstm(sequences)
                        root_embeddings = []
                        for index, new_state in enumerate(new_states):
                            new_state.set_layer_embeddings(
                                new_embeddings['node_index'],
                                new_embeddings,
                                source_indices=index,
                            )
                            root_embeddings.append(new_state.get_root_embeddings().mean(dim=0))
                        res = self.model_lstm.predict(root_embeddings)
                    times['gpu'] += timer.time
                else:
                    with timer:
                        res = self.model_transformer(sequences, dropout=False)
                        res = res.view(-1)
                    times['gpu'] += timer.time

            all_res.append(res)
            exploration_mask.extend((is_exploration_branch for i in range(res.shape[0])))

        with timer:
            all_res = torch.cat(all_res, dim=-1)
            exploration_mask = torch.tensor(exploration_mask, device=self.device)
        times['gpu'] += timer.time

        prob = self.explorer.prob
        prob_coef = database.config.topk_explore_prob_coef
        prob = prob + (1 - prob) * (1 - math.exp(-prob_coef * exploration_bias))

        exploration_branches = 0
        beam_size = min(beam_size, len(all_candidates))
        if exploration:
            if beam_size > 1:
                exploration_branches = 1
            for i in range(beam_size - 2):
                if self.explorer.explore(prob):
                    exploration_branches += 1

        with timer:
            selected_res = torch.topk(all_res, beam_size - exploration_branches, largest=False, sorted=True)
            selected_items, selected_res = selected_res.values.tolist(), selected_res.indices.tolist()
            exploration_mask[selected_res] = False
        times['gpu'] += timer.time

        if exploration_branches > 0:
            exploration_aug_ratio = database.config.topk_explore_lambda
            explore_res = self.randomizer.sample(range(len(all_candidates) * len(self._operator_types)), beam_size)
            explore_res = list(filter(lambda x: x not in selected_res, explore_res))
            explore_res = explore_res[:round(exploration_branches * exploration_aug_ratio)]

            with timer:
                exploration_mask[explore_res] = True
                explore_all_res = all_res - 256 * exploration_mask
                explore_res = torch.topk(explore_all_res, exploration_branches, largest=False, sorted=True).indices.tolist()
            times['gpu'] += timer.time
        else:
            explore_res = []

        for index in explore_res:
            selected_items.append(all_res[index].item())
        selected_res.extend(explore_res)

        selected = []
        for index in selected_res:
            selected_index = index // len(self._operator_types)
            selected_operator = index % len(self._operator_types)
            state_index, state, action = all_candidates[selected_index]
            selected.append((state_index, state, (*action, self._operator_types[selected_operator])))

        if detail:
            return {
                'result': selected,
                'exploration_branches': exploration_branches,
                'exploration_prob': prob,
                'selected_item_logits': selected_items,
                'stats': {
                    'min': all_res.min().item(),
                    'max': all_res.max().item(),
                    'mean': all_res.mean().item(),
                },
                'time': times,
            }

        return selected

    def step(self, state : Plan, action : tuple, join=None):
        if len(action) < 3:
            if join is None:
                join = self._operator_types[0]
            action = (*action, join)
        state.join(*action)
        if database.config.use_lstm:
            last_action = state.get_last_action_embeddings()
            new_embeddings = self.model_lstm([last_action])
            state.set_layer_embeddings(
                new_embeddings['node_index'],
                new_embeddings,
                source_indices=0,
            )
        self._map_explain_info_to_plan(state)
        return state

    def get_plan(self, sql : Sql, bushy=True, beam_size=None, detail=False):
        times = []
        _train_mode = self.__train_mode
        self.eval_mode()
        try:
            with timer:
                if beam_size is None:
                    beam_size = database.config.beam_width
                with timer:
                    plan = self.plan_init(sql)
                times.append({'init': timer.time})
                plans = [plan]
                while not plans[0].completed:
                    selected_result = self.search(plans, beam_size=beam_size, exploration=False, bushy=bushy, detail=True)
                    selected = selected_result['result']
                    times.append(selected_result['time'])
                    plans = []
                    for _, plan, action in selected:
                        plan = plan.clone()
                        self.step(plan, action)
                        plans.append(plan)
                plan = plans[0]
            times = sum_batch(times)
            times['total'] = timer.time
            if detail:
                return {
                    'result': plan,
                    'time': times,
                }
            return plan
        finally:
            self.train_mode(_train_mode)

    def _gt_convert(self, gt : typing.Union[int, float, typing.Iterable[float]], reward_weighting=0.1):
        if isinstance(gt, (int, float)):
            gt = (gt, gt)
        gt = tuple(map(self._value_preprocess, gt))
        gt = self._gt_process((gt,), reward_weighting=reward_weighting).item()
        return gt

    def predict_plan(
        self,
        state : Plan,
        detail=False,
        *,
        gt : typing.Union[None, int, float, typing.Iterable[float]] = None,
        reward_weighting=0.1,
    ):
        _train_mode = self.__train_mode
        self.eval_mode()
        try:
            state, *_ = self._batch_embedding_update((state, ))
            if database.config.use_lstm:
                root_embeddings = state.get_root_embeddings().mean(dim=0)
                pred = self.model_lstm.predict(root_embeddings)
            else:
                state_seq = state.to_sequence()
                pred = self.model_transformer([state_seq], dropout=False).view(-1)
            res = pred.item()
            if detail:
                res = {
                    'value': res,
                }
                if gt is not None:
                    gt = self._gt_convert(gt, reward_weighting)
                    loss = (res['value'] - gt) ** 2
                    res['loss'] = loss
            return res
        finally:
            self.train_mode(_train_mode)

    def compare_plan(
        self,
        state1 : Plan,
        state2 : Plan,
        detail=False,
        *,
        gt1 : typing.Union[None, int, float, typing.Iterable[float]] = None,
        gt2 : typing.Union[None, int, float, typing.Iterable[float]] = None,
        reward_weighting=0.1,
    ):
        _train_mode = self.__train_mode
        self.eval_mode()
        try:
            state1, state2 = self._batch_embedding_update((state1, state2))
            if database.config.use_lstm:
                preds = self.model_lstm.predict([
                    state1.get_root_embeddings().mean(dim=0),
                    state2.get_root_embeddings().mean(dim=0),
                ]).view(-1)
            else:
                state1_seq, state2_seq = state1.to_sequence(), state2.to_sequence()
                preds = self.model_transformer([state1_seq, state2_seq], dropout=False).view(-1)
            if detail:
                value1, value2 = preds[0].item(), preds[1].item()
                res = {
                    'value1': value1,
                    'value2': value2,
                    'comparison': value1 - value2,
                }
                if gt1 is not None:
                    gt1 = self._gt_convert(gt1, reward_weighting)
                    loss = (value1 - gt1) ** 2
                    res['loss1'] = loss
                if gt2 is not None:
                    gt2 = self._gt_convert(gt2, reward_weighting)
                    loss = (value2 - gt2) ** 2
                    res['loss2'] = loss
                if gt1 and gt2:
                    res['correct'] = (res['comparison'] < 0) ^ (gt1 >= gt2)
                return res
            return (preds[0] - preds[1]).item()
        finally:
            self.train_mode(_train_mode)

    def _update_best_value(self, state : Plan, value : float):
        state_hash = f'{state.sql.filename} {state.hints(operators=False)}'
        state_hash_op = f'{state.sql.filename} {state.hints(operators=True)}'
        if database.config.log_debug:
            previous_value = self.best_values.get(state_hash, None)
            if previous_value is not None and value < previous_value:
                logger = Logger(database.config.log_name)
                logger.log(logging.DEBUG, f'best update: {previous_value:.03f} -> {value:0.3f} | {state_hash}')
        self.best_values[state_hash] = value
        self.best_values[state_hash_op] = value

    def _get_best_value(self, state : Plan):
        state_hash = f'{state.sql.filename} {state.hints(operators=False)}'
        state_hash_op = f'{state.sql.filename} {state.hints(operators=True)}'
        return self.best_values.get(state_hash, None), self.best_values.get(state_hash_op, None)

    def _value_preprocess(self, value : float):
        return value / 1000

    def add_memory(self, state : Plan, value : float, update_previous_state: bool = False):
        value = self._value_preprocess(value)

        state = state.clone()
        state.clear_embeddings()
        self.memory.push(state)
        self._update_best_value(state, value)
        if update_previous_state:
            new_state = state.clone()
            new_state.revert()
            self._update_best_value(new_state, value)

    def add_pair_memory(self, state : Plan, value : float):
        value = self._value_preprocess(value)

        state = state.clone()
        state.clear_embeddings()
        state_hash_op = f'{state.sql.filename} {state.hints(operators=True)}'
        self.pair_memory.push(state, key=state.sql.filename, hash=state_hash_op)
        self._update_best_value(state, value)
        self.best_values[state.sql.filename] = value

    def add_priority_memory(self, states : typing.Iterable[Plan], value : float, index=None):
        value = self._value_preprocess(value)

        states = tuple(states)
        if len(states) == 0:
            raise RuntimeError('empty states')
        if index is None:
            index = states[0].sql.filename
        states = tuple(map(lambda x: x.clone(), states))
        for state in states:
            state.clear_embeddings()
        self.memory.push_priority_queue(states, value, index)
        for state in states:
            self._update_best_value(state, value)

    def _batch_embedding_update(self, states : typing.Iterable[Plan]):
        states = tuple(states)

        gs = []
        node_indices_list = []
        for state in states:
            gs.append(state.sql.graph)
            node_indices_list.append(state.sql.graph_node_indices)

        gs = self.model_extractor(dgl.batch(gs).to(self.device))
        gs = dgl.unbatch(gs)
        for g, node_indices, state in zip(gs, node_indices_list, states):
            state.clear_embeddings()
            state.set_leaf_embeddings(g.nodes['table'].data['res'], node_indices)

        if database.config.use_lstm:
            layer_actions = []
            for state_index, state in enumerate(states):
                state_layer_actions = state.layer_actions()
                for layer_index, layer in enumerate(state_layer_actions):
                    while layer_index >= len(layer_actions):
                        layer_actions.append([() for i in range(state_index)])
                    current_layer_actions = layer_actions[layer_index]
                    current_layer_actions.append(layer)
                for i in range(len(state_layer_actions), len(layer_actions)):
                    current_layer_actions = layer_actions[i]
                    current_layer_actions.append(())
            for depth, layer in enumerate(layer_actions):
                layer_indices = [0]
                layer_seqs = []
                for state, layer_nodes in zip(states, layer):
                    if not layer_nodes:
                        layer_indices.append(layer_indices[-1])
                    else:
                        layer_embedding_seq = state.get_layer_embeddings(layer_nodes)
                        layer_seqs.append(layer_embedding_seq)
                        layer_indices.append(layer_indices[-1] + layer_embedding_seq.sequence_length)
                new_embeddings = self.model_lstm(layer_seqs)
                for index, state in enumerate(states):
                    embedding_slice = slice(layer_indices[index], layer_indices[index + 1])
                    state.set_layer_embeddings(
                        new_embeddings['node_index'],
                        new_embeddings,
                        source_indices=embedding_slice,
                    )

        return states

    def _gt_process(self, gts, reward_weighting=0.1):
        gt, gt_op = zip(*gts)
        gt, gt_op = torch.tensor(gt, device=self.device), torch.tensor(gt_op, device=self.device)
        gt = gt * (1 - reward_weighting) + gt_op * reward_weighting
        gt = gt.clamp(1e-8).log10()
        gt = mF.log_transform_with_cap(gt, -1, 1)
        return gt

    def _train(
        self,
        batch_size=64,
        preserve=4,
        reward_weighting=0.1,
        use_ranking_loss=False,
        use_complementary_loss=False,
        equal_suppress=8,
        use_other_states=False,
    ):
        with timer:
            batch = self.memory.sample(batch_size, preserve=preserve)
            real_batch_size = len(batch)
            if real_batch_size == 0:
                raise RuntimeError('empty memory')

            pair_batch_size = 0
            pair_batch = None
            if use_ranking_loss:
                pair_batch = self.pair_memory.sample(batch_size, preserve=0)
                pair_batch_size = len(pair_batch)

            if pair_batch_size:
                pair_1, pair_2 = zip(*pair_batch)
                batch = [*batch, *pair_1, *pair_2]
                all_states = self._batch_embedding_update(batch)
                batch, pair_batch = all_states[:real_batch_size], all_states[real_batch_size:]
            else:
                use_ranking_loss = False
                all_states = self._batch_embedding_update(batch)
                pair_batch = None
                batch = all_states[:]

        _time_batch_update = timer.time

        with timer:
            gts = []
            seqs = []
            history_gts = []
            history_seqs = []
            pair_gts = []
            pair_seqs = []

            for index, state in enumerate(batch):
                best_value = self.best_values[state.sql.filename]
                gt, gt_op = self._get_best_value(state)

                if use_complementary_loss and state.total_branch_nodes > 0:
                    prev_state = state.clone()
                    prev_state.revert()
                    gt_prev, gt_prev_op = self._get_best_value(prev_state)
                    if gt_prev is None or gt_prev_op is None:
                        prev_state = None
                    elif equal_suppress > 0:
                        if gt <= gt_prev and gt_op <= gt_prev_op and index % equal_suppress != 0:
                            prev_state = None
                else:
                    gt_prev, gt_prev_op = None, None
                    prev_state = None

                if database.config.use_lstm:
                    seq = state.get_root_embeddings().mean(dim=0)
                else:
                    seq = state.to_sequence()
                if prev_state is None:
                    seqs.append(seq)
                    gts.append((gt, gt_op, best_value))
                else:
                    if database.config.use_lstm:
                        prev_seq = prev_state.get_root_embeddings().mean(dim=0)
                    else:
                        prev_seq = prev_state.to_sequence()
                    history_seqs.append((seq, prev_seq))
                    history_gts.append(((gt, gt_op, best_value), (gt_prev, gt_prev_op, best_value)))

            if pair_batch:
                for state in pair_batch:
                    best_value = self.best_values[state.sql.filename]
                    gt, gt_op = self._get_best_value(state)
                    if database.config.use_lstm:
                        seq = state.get_root_embeddings().mean(dim=0)
                    else:
                        seq = state.to_sequence()
                    pair_gts.append((gt, gt_op, best_value))
                    pair_seqs.append(seq)
        _time_to_sequence = timer.time

        with timer:
            wo_comp_size = len(seqs)
            comp_size = len(history_seqs)

            if comp_size:
                history_seqs_ori, history_seqs_prev = zip(*history_seqs)
                history_gts_ori, history_gts_prev = zip(*history_gts)
            else:
                history_seqs_ori, history_seqs_prev = (), ()
                history_gts_ori, history_gts_prev = (), ()

            all_seqs = [*seqs, *history_seqs_ori, *history_seqs_prev, *pair_seqs]
            all_gts = [*gts, *history_gts_ori, *history_gts_prev, *pair_gts]
            gt, gt_op, best_values = zip(*all_gts)

            all_gts = self._gt_process(zip(gt, gt_op), reward_weighting=reward_weighting)
            if database.config.use_lstm:
                all_pred = self.model_lstm.predict(all_seqs)
            else:
                all_pred = self.model_transformer(all_seqs).view(-1)

            _best_values = torch.tensor(best_values, device=self.device)
            _gt_op = torch.tensor(gt_op, device=self.device)
            mse_loss_weights = (_best_values / _gt_op) ** database.config.rl_sample_weighting_exponent \
                if database.config.rl_sample_weighting_exponent != 0 else None

            if use_other_states:
                main_res = all_pred
                main_gts = all_gts
                main_weights = mse_loss_weights if mse_loss_weights is not None else None
            else:
                main_res = all_pred[:wo_comp_size + comp_size]
                main_gts = all_gts[:wo_comp_size + comp_size]
                main_weights = mse_loss_weights[:wo_comp_size + comp_size] if mse_loss_weights is not None else None
            if main_weights is not None:
                main_weights = main_weights * main_weights.shape[0] / main_weights.sum()
                main_loss_ori = F.mse_loss(main_res, main_gts, reduction='none') * main_weights
            else:
                main_loss_ori = F.mse_loss(main_res, main_gts, reduction='none')
            main_loss = main_loss_ori.mean()

            if use_ranking_loss and pair_batch_size > 0:
                pair_res = all_pred[wo_comp_size + comp_size * 2:]
                pair_gts = all_gts[wo_comp_size + comp_size * 2:]

                assert pair_res.shape[0] == 2 * pair_batch_size
                pair_res_1, pair_res_2 = pair_res[:pair_batch_size], pair_res[pair_batch_size:]
                pair_gts_1, pair_gts_2 = pair_gts[:pair_batch_size], pair_gts[pair_batch_size:]

                pair_cmp_gts = (pair_gts_1 < pair_gts_2).to(torch.float) - (pair_gts_1 > pair_gts_2).to(torch.float)
                ranking_loss_ori = ((pair_res_1 - pair_res_2) * pair_cmp_gts).clamp(0)
                ranking_loss = ranking_loss_ori.mean()
            else:
                ranking_loss = 0.

            if use_complementary_loss and comp_size > 0:
                comp_res_ori, comp_res_prev = all_pred[wo_comp_size : wo_comp_size + comp_size], all_pred[wo_comp_size + comp_size : wo_comp_size + comp_size * 2]
                complementary_loss_ori = (comp_res_prev - comp_res_ori).clamp(0)
                complementary_loss = complementary_loss_ori.mean()
            else:
                complementary_loss = 0.
        _time_predict = timer.time

        for state in all_states:
            state.clear_embeddings()

        return {
            'loss': {
                'main': main_loss,
                'ranking': ranking_loss,
                'complementary': complementary_loss,
            },
            'time': {
                'batch_update': _time_batch_update,
                'to_sequence': _time_to_sequence,
                'predict': _time_predict,
            }
        }

    def train(self, iterations=1, epoch=None, separate_backward=True, sample_preserve=None):
        if iterations < 1:
            iterations = 1

        batch_size = database.config.batch_size
        if sample_preserve is None:
            if epoch is not None and epoch >= database.config.priority_memory_start_epoch:
                sample_preserve = round(batch_size * database.config.priority_memory_ratio)
            else:
                sample_preserve = 0
        loss_weights = {
            'main': 1.,
            'ranking': database.config.rl_ranking_loss_weight,
            'complementary': database.config.rl_complementary_loss_weight,
        }
        clip_ratio = database.config.rl_grad_clip_ratio

        _time_backward = 0.
        results = []
        for i in range(iterations):
            with timer:
                training_res = self._train(
                    batch_size,
                    sample_preserve,
                    reward_weighting=database.config.rl_reward_weighting,
                    use_ranking_loss=database.config.rl_use_ranking_loss and epoch >= database.config.rl_ranking_loss_start_epoch,
                    use_complementary_loss=database.config.rl_use_complementary_loss,
                    use_other_states=database.config.rl_use_other_states,
                    equal_suppress=database.config.rl_complementary_loss_equal_suppress,
                )
                results.append(training_res)

                if separate_backward:
                    self.optim.zero_grad()
                    total_loss = 0.
                    for k, v in training_res['loss'].items():
                        weight = loss_weights.get(k, 0.)
                        total_loss = total_loss + v * weight
                    total_loss.backward()
                    for param_group in self.optim.param_groups:
                        lr = param_group['lr']
                        params = param_group['params']
                        torch.nn.utils.clip_grad_norm_(params, max_norm=lr * clip_ratio, norm_type=float('inf'))
                    self.optim.step()
            _time_backward += timer.time

        results = sum_batch(results)
        if not separate_backward:
            self.optim.zero_grad()
            total_loss = 0.
            for k, v in results['loss'].items():
                weight = loss_weights.get(k, 0.)
                total_loss = total_loss + v * weight
            total_loss.backward()
            for param_group in self.optim.param_groups:
                lr = param_group['lr']
                params = param_group['params']
                torch.nn.utils.clip_grad_norm_(params, max_norm=lr * clip_ratio, norm_type=float('inf'))
            self.optim.step()

        self.explorer.step()

        return dict_map(lambda x: x.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else x, results)
