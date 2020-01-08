import util

import numpy as np

import torch
from torch import nn

K = 5


class ContrastiveSWMMASK(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """

    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, hinge=1., sigma=0.5, encoder='large',
                 ignore_action=False, copy_action=False):
        super(ContrastiveSWMMASK, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        self.pos_loss = 0
        self.neg_loss = 0

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        self.transition_model = TransitionGNNMASK(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action)

        self.width = width_height[0]
        self.height = width_height[1]
        self.seg_encoder = EncoderCNNSeg(
            input_dim=num_channels,
            hidden_dim=hidden_dim // 16,
            num_objects=num_objects
        )

    @staticmethod
    def discriminator(energy):
        """Energy-based discriminator."""
        return torch.exp(-energy)

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma ** 2)

        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs, action, next_obs):

        objs = self.seg_encoder(obs)
        # objs = self.obj_extractor(obs)
        next_objs = self.seg_encoder(next_obs)

        # next_objs = self.obj_extractor(next_obs)

        # state = self.obj_encoder(objs)
        # next_state = self.obj_encoder(next_objs)

        # Sample negative state across episodes at random
        batch_size = objs.size(0)
        perm = np.random.permutation(batch_size)
        neg_next_objs = next_objs[perm]

        self.pos_loss = self.energy(objs, action, next_objs, no_trans=True)
        zeros = torch.zeros_like(self.pos_loss)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                objs, action, neg_next_objs, no_trans=True)).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class TransitionGNNMASK(torch.nn.Module):
    """GNN-based transition function."""

    def __init__(self, input_dim, hidden_dim, action_dim, num_objects,
                 ignore_action=False, copy_action=False, act_fn='relu'):
        super(TransitionGNNMASK, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            util.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            util.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            util.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            util.get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = util.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states, action):

        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)

            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col], edge_attr)

        if not self.ignore_action:

            if self.copy_action:
                action_vec = util.to_one_hot(
                    action, self.action_dim).repeat(1, self.num_objects)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                action_vec = util.to_one_hot(
                    action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(batch_size, num_nodes, -1)


class EncoderCNNSeg(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNSeg, self).__init__()

        self.cnn1 = nn.Conv2d(6, 32, (8, 8), stride=4)
        self.act1 = util.get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(32)

        self.cnn2 = nn.Conv2d(32, 64, (4, 4), stride=2)
        self.act2 = util.get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(64, 64, (3, 3), stride=1)
        self.act3 = util.get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Flatten(1, -1)
        self.act4 = util.get_act_fn(act_fn_hid)

        self.fc2 = nn.Linear(256, 1323)
        self.act5 = util.get_act_fn(act_fn_hid)
        self.act6 = util.get_act_fn(act_fn)

        self.cnn4 = nn.Conv2d(3, K, (1, 1))

        self.upsample1 = nn.UpsamplingBilinear2d(32)
        self.upsample2 = nn.UpsamplingBilinear2d(50)

        self.ln4=nn.BatchNorm2d(3)
        self.ln5=nn.BatchNorm2d(3)



    def forward(self, obs):
        bs = obs.size(0)
        h = self.ln1(self.act1(self.cnn1(obs)))
        h = self.ln2(self.act2(self.cnn2(h)))
        h = self.ln3(self.act3(self.cnn3(h)))
        h = self.fc2(self.fc1(h)).view(bs, 3, 21, 21)

        h = self.ln4(self.act4(self.upsample1(h)))
        h = self.ln5(self.act5(self.upsample2(h)))

        h = self.cnn4(h)
        h = self.act6(h)

        return h
