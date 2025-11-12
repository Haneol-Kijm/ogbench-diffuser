import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if "valids" in self._dict:
            (self.valid_idxs,) = np.nonzero(self["valids"] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if "valids" in self._dict:
            return self.valid_idxs[
                np.random.randint(len(self.valid_idxs), size=num_idxs)
            ]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if "next_observations" not in result:
            result["next_observations"] = self._dict["observations"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config["value_p_curgoal"]
            + self.config["value_p_trajgoal"]
            + self.config["value_p_randomgoal"],
            1.0,
        )
        assert np.isclose(
            self.config["actor_p_curgoal"]
            + self.config["actor_p_trajgoal"]
            + self.config["actor_p_randomgoal"],
            1.0,
        )

        if self.config["frame_stack"] is not None:
            # Only support compact (observation-only) datasets.
            assert "next_observations" not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(
                    np.arange(self.size)
                )
                self.dataset = Dataset(
                    self.dataset.copy(dict(observations=stacked_observations))
                )

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
        )

        batch["value_goals"] = self.get_observations(value_goal_idxs)
        batch["actor_goals"] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                self.augment(
                    batch,
                    ["observations", "next_observations", "value_goals", "actor_goals"],
                )

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(
                p=1 - self.config["discount"], size=batch_size
            )  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal),
                traj_goal_idxs,
                random_goal_idxs,
            )

            # Goals at the current state.
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs
            )

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config["frame_stack"] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(
                lambda arr: arr[idxs], self.dataset["observations"]
            )
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[
            np.searchsorted(self.initial_locs, idxs, side="right") - 1
        ]
        rets = []
        for i in reversed(range(self.config["frame_stack"])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(
                jax.tree_util.tree_map(
                    lambda arr: arr[cur_idxs], self.dataset["observations"]
                )
            )
        return jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=-1), *rets
        )


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        batch["value_goals"] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], final_state_idxs
        )
        batch["low_actor_goals"] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config["actor_geom_sample"]:
            # Geometric sampling.
            offsets = np.random.geometric(
                p=1 - self.config["discount"], size=batch_size
            )  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        high_traj_target_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], high_traj_goal_idxs
        )

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(
            idxs + self.config["subgoal_steps"], final_state_idxs
        )

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config["actor_p_randomgoal"]
        high_goal_idxs = np.where(
            pick_random, high_random_goal_idxs, high_traj_goal_idxs
        )
        high_target_idxs = np.where(
            pick_random, high_random_target_idxs, high_traj_target_idxs
        )

        batch["high_actor_goals"] = self.get_observations(high_goal_idxs)
        batch["high_actor_targets"] = self.get_observations(high_target_idxs)

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                self.augment(
                    batch,
                    [
                        "observations",
                        "next_observations",
                        "value_goals",
                        "low_actor_goals",
                        "high_actor_goals",
                        "high_actor_targets",
                    ],
                )

        return batch


from utils.normalization import DatasetNormalizer


@dataclasses.dataclass
class DiffuserSequenceDataset:
    """
    [Task 2] jannerm/diffuser의 SequenceDataset 로직을 포팅한 부모 클래스.
    ogbench의 sample(batch_size) 인터페이스를 따름.
    """

    dataset: Dataset  # ogbench/main.py가 생성한 Dataset 객체
    config: Any

    def __post_init__(self):
        # 1. diffuser/datasets/sequence.py의 __init__ 로직
        self.fields = self.dataset._dict  # 원본 numpy 배열 딕셔너리 (평평한 상태)
        self.horizon = self.config["horizon"]
        self.max_path_length = self.config["max_path_length"]
        self.use_padding = self.config.get("use_padding", True)

        # 2. ogbench('terminals') -> diffuser('path_lengths') 변환
        if "path_lengths" not in self.fields:
            path_lengths = self._compute_path_lengths_from_terminals(
                self.fields["terminals"]
            )
            # path_lengths를 self.fields에 저장 (Normalizer가 사용)
            self.fields["path_lengths"] = path_lengths
        else:
            path_lengths = self.fields["path_lengths"]

        self.n_episodes = len(path_lengths)
        obs_dim = self.fields["observations"].shape[-1]
        act_dim = self.fields["actions"].shape[-1]

        # --- [수정 시작] 원본 diffuser의 ReplayBuffer 로직을 여기에 구현 ---

        # 3. (N_episodes, max_path_length, dim) 모양의 빈 텐서 생성
        # (jannerm/diffuser/datasets/buffer.py ReplayBuffer.__init__)
        self.fields_reshaped = {}  # 재구성된 데이터를 저장할 새 딕셔너리
        self.fields_reshaped["observations"] = np.zeros(
            (self.n_episodes, self.max_path_length, obs_dim), dtype=np.float32
        )
        self.fields_reshaped["actions"] = np.zeros(
            (self.n_episodes, self.max_path_length, act_dim), dtype=np.float32
        )
        self.fields_reshaped["rewards"] = np.zeros(
            (self.n_episodes, self.max_path_length, 1), dtype=np.float32
        )

        # 4. 평평한(flat) 데이터를 2D 텐서로 복사
        # (jannerm/diffuser/datasets/buffer.py ReplayBuffer.add_path)
        current_idx = 0
        for i, length in enumerate(path_lengths):
            # 원본 diffuser의 'assert' 로직 (jannerm/diffuser/datasets/buffer.py L103)
            assert (
                int(length) <= self.max_path_length
            ), f"Episode {i} length {int(length)} exceeds max_path_length {self.max_path_length}"

            # (jannerm/diffuser/datasets/buffer.py L109)
            int_length = int(length)
            self.fields_reshaped["observations"][i, :int_length] = self.fields[
                "observations"
            ][current_idx : current_idx + int_length]
            self.fields_reshaped["actions"][i, :int_length] = self.fields["actions"][
                current_idx : current_idx + int_length
            ]
            self.fields_reshaped["rewards"][i, :int_length] = self.fields["rewards"][
                current_idx : current_idx + int_length
            ][
                :, None
            ]  # (L, 1) 형태로

            current_idx += int_length

        # 5. Normalizer 초기화 (재구성된 데이터로!)
        # (jannerm/diffuser/datasets/sequence.py L34)
        self.normalizer = DatasetNormalizer(
            self.fields_reshaped,  # <--- 'self.fields' (평평한) 대신 'self.fields_reshaped' (2D) 사용
            self.config["normalizer"],
            path_lengths=path_lengths,
        )

        # 6. 데이터 정규화 (jannerm/diffuser/datasets/sequence.py L41)
        # Normalizer가 2D 텐서를 받아 정규화된 2D 텐서를 반환
        self.normed_observations = self.normalizer(
            self.fields_reshaped["observations"], "observations"
        )
        self.normed_actions = self.normalizer(
            self.fields_reshaped["actions"], "actions"
        )

        # (ValueDataset에서 사용할 정규화되지 *않은* 2D 보상)
        self.fields["rewards_reshaped"] = self.fields_reshaped["rewards"]

        # --- [수정 완료] ---

        # 7. 샘플링 인덱스 생성 (jannerm/diffuser/datasets/sequence.py L35)
        self.indices = self.make_indices(path_lengths, self.horizon)

        # 8. Dims 저장 (jannerm/diffuser/datasets/sequence.py L37-38)
        self.observation_dim = obs_dim
        self.action_dim = act_dim

    def _compute_path_lengths_from_terminals(self, terminals):
        """
        ogbench의 'terminals' (bool 배열)로부터
        diffuser의 'path_lengths' (int 배열)를 생성합니다.
        """
        path_lengths = []
        start_idx = 0
        terminal_indices = np.where(terminals)[0]

        for end_idx in terminal_indices:
            length = end_idx - start_idx + 1
            path_lengths.append(length)
            start_idx = end_idx + 1

        return np.array(path_lengths)

    def make_indices(self, path_lengths, horizon):
        """
        diffuser/datasets/sequence.py (lines 48-59)
        """
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def _get_batch_from_indices(self, batch_indices):
        """
        [핵심 헬퍼 함수]
        상속을 위해 `sample` 로직 중 실제 데이터 슬라이싱 부분을 분리합니다.
        diffuser의 SequenceDataset.__getitem__ 로직을 배치 처리합니다.

        """
        trajectories_list = []
        conditions_list_0 = []  # t=0 시점의 observation만 저장

        for path_ind, start, end in batch_indices:
            # 2a. 정규화된 궤적 슬라이싱
            observations = self.normed_observations[path_ind, start:end]
            actions = self.normed_actions[path_ind, start:end]

            # 2b. 궤적 생성 (action, observation 순서 중요)
            trajectories = np.concatenate([actions, observations], axis=-1)

            # 2c. 컨디션 생성 (diffuser 모델은 t=0 observation만 사용)
            conditions_0 = observations[0]

            trajectories_list.append(trajectories)
            conditions_list_0.append(conditions_0)

        # 4. 리스트를 numpy 배치로 스택
        batch_trajectories = np.stack(trajectories_list)
        batch_conditions = {0: np.stack(conditions_list_0)}

        return {
            "trajectories": batch_trajectories,  # (B, H, A+O)
            "conditions": batch_conditions,  # {0: (B, O)}
        }

    def sample(self, batch_size: int):
        """
        ogbench/main.py가 호출할 DiffuserSequenceDataset의 샘플링 함수.
        """
        # 1. 배치 크기만큼 랜덤 인덱스 샘플링
        rand_indices = np.random.randint(len(self.indices), size=batch_size)
        batch_indices = self.indices[rand_indices]  # (path_ind, start, end)의 배치

        # 2. 헬퍼 함수를 호출하여 딕셔너리 반환
        return self._get_batch_from_indices(batch_indices)


@dataclasses.dataclass
class DiffuserValueDataset(DiffuserSequenceDataset):
    """
    [Task 2] jannerm/diffuser의 ValueDataset 로직을 포팅한 자식 클래스.
    부모의 초기화/샘플링 로직을 재사용하고 'values' 계산만 추가함.
    """

    def __post_init__(self):
        # 1. 부모의 __post_init__을 먼저 호출 (정규화, 인덱싱 등 수행)
        super().__post_init__()

        # 2. ValueDataset 고유의 로직만 추가
        #
        self.discount = self.config["discount"]
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]

    def sample(self, batch_size: int):
        """
        ogbench/main.py가 호출할 DiffuserValueDataset의 샘플링 함수.
        diffuser의 ValueDataset.__getitem__ 로직을 재현.

        """
        # 1. 배치 크기만큼 랜덤 인덱스 샘플링
        rand_indices = np.random.randint(len(self.indices), size=batch_size)
        batch_indices = self.indices[rand_indices]  # (path_ind, start, end)의 배치

        # 2. 부모의 헬퍼 함수를 호출해 (trajectories, conditions) 딕셔너리를 가져옴
        batch_dict = self._get_batch_from_indices(batch_indices)

        # 3. ValueDataset 고유의 'values' 계산 로직 추가
        values_list = []

        rewards_reshaped = self.fields["rewards_reshaped"]

        for path_ind, start, end in batch_indices:
            # 3a. 원본(unnormalized) 보상 슬라이싱
            rewards = rewards_reshaped[path_ind, start:]  # (L, 1)
            rewards = rewards.squeeze()  # (L, )

            # 3b. 할인 계수 슬라이싱
            discounts = self.discounts[: len(rewards)]

            # 3c. 가치(미래 보상 합) 계산
            value = (discounts * rewards).sum().astype(np.float32)
            values_list.append(value)

        # 4. 딕셔너리에 'values' 키를 추가하여 반환
        batch_dict["values"] = np.stack(values_list)  # (B,)

        # --- 🔻 [수정] main.py 호환성을 위해 키 추가 🔻 ---
        # Diffuser 에이전트의 create는 ex_observations로 (B, H, A+O) 궤적을 사용
        batch_dict["observations"] = batch_dict["trajectories"]

        # ex_actions로 (B, H, A) 액션 부분을 사용
        # (self.action_dim은 부모 클래스의 __post_init__에서 설정됨)
        batch_dict["actions"] = batch_dict["trajectories"][:, :, : self.action_dim]
        # --- 🔺 [수정 완료] 🔺 ---

        return batch_dict
