#!/usr/bin/env python3
"""
LBEBM3D baseline inference script (standalone).

Goal:
  After training, load a saved checkpoint (.pt) and run *direct inference*:
  given a past 3D trajectory (T_obs, 3), predict future trajectory (T_pred, 3).

Why standalone?
  In this repo, LBEBM3D/MLP/SC_LSTM are defined inside lbebm3D.py's main(),
  so importing them is inconvenient. This script re-implements the *inference*
  parts and reconstructs model hyperparameters from the checkpoint state_dict.

Input formats:
  - .npy: array shape (T, 3)
  - .csv: columns x,y,z (header optional)

Output:
  - prints predicted future to stdout
  - optionally saves as .npy/.csv and a quick plot
  
python infer_lbebm3d_baseline.py ^
  --model_path "..\saved_models\lbebm3D_scene1.pt" ^
  --input_path "..\dataset\swarm\test\saved_data.pickle" ^
  --sample_idx 0 ^
  --output_path "..\pred_from_pickle_sample0.csv" ^
  --device cuda:0 ^
  --plot ^
  --e_l_steps 20 ^
  --e_l_step_size 0.4 ^
  --e_init_sig 2.0 ^
  --e_prior_sig 2.0 ^
  --e_l_with_noise ^
  --smooth --dt 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: List[int],
        activation: str = "relu",
        discrim: bool = False,
        dropout: float = -1,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_size) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout / 3) if i == 1 else self.dropout)(x)
            elif self.sigmoid is not None:
                x = self.sigmoid(x)
        return x


class SC_LSTM(nn.Module):
    # Copied (minimally) from lbebm3D.py to match checkpoint parameter names.
    def __init__(self, input_sz: int, hidden_sz: int, reduce_sz: int, peephole: bool = False, num_state: int = 1):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.peephole = peephole
        self.num_iter = num_state

        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias_c = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.W_x = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U_x = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias_x = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.W_y = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U_y = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias_y = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.W_z = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U_z = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias_z = nn.Parameter(torch.Tensor(hidden_sz * 4))

        # state correlation
        self.W_rc1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz // reduce_sz))
        self.W_rx1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz // reduce_sz))
        self.W_ry1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz // reduce_sz))
        self.W_rz1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz // reduce_sz))

        self.W_cx = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz * 2, hidden_sz))
        self.W_cy = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz * 2, hidden_sz))
        self.W_cz = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz * 2, hidden_sz))
        self.bias_cx = nn.Parameter(torch.Tensor(hidden_sz))
        self.bias_cy = nn.Parameter(torch.Tensor(hidden_sz))
        self.bias_cz = nn.Parameter(torch.Tensor(hidden_sz))
        self.W_cxyz = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        # state aggregation
        self.W_rc2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz // reduce_sz))
        self.W_rx2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz // reduce_sz))
        self.W_ry2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz // reduce_sz))
        self.W_rz2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz // reduce_sz))

        self.W_tcc = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz * 4, hidden_sz // reduce_sz))
        self.W_tcx = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz * 4, hidden_sz // reduce_sz))
        self.W_tcy = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz * 4, hidden_sz // reduce_sz))
        self.W_tcz = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz * 4, hidden_sz // reduce_sz))

        self.W_xc = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz, hidden_sz))
        self.W_yc = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz, hidden_sz))
        self.W_zc = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz, hidden_sz))
        self.bias_xc = nn.Parameter(torch.Tensor(hidden_sz))
        self.bias_yc = nn.Parameter(torch.Tensor(hidden_sz))
        self.bias_zc = nn.Parameter(torch.Tensor(hidden_sz))
        self.W_xyzc = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_q = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz, hidden_sz // reduce_sz))
        self.W_k = nn.Parameter(torch.Tensor(hidden_sz // reduce_sz, hidden_sz // reduce_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def state_extraction(self, h_tc, h_tx, h_ty, h_tz, c_tx, c_ty, c_tz):
        r_tc = h_tc @ self.W_rc1
        r_tx = h_tx @ self.W_rx1
        r_ty = h_ty @ self.W_ry1
        r_tz = h_tz @ self.W_rz1

        gate_tcx = torch.sigmoid(torch.cat((r_tc, r_tx), dim=-1) @ self.W_cx + self.bias_cx)
        gate_tcy = torch.sigmoid(torch.cat((r_tc, r_ty), dim=-1) @ self.W_cy + self.bias_cy)
        gate_tcz = torch.sigmoid(torch.cat((r_tc, r_tz), dim=-1) @ self.W_cz + self.bias_cz)

        cc_tx = c_tx + (h_tc * gate_tcx) @ self.W_cxyz
        cc_ty = c_ty + (h_tc * gate_tcy) @ self.W_cxyz
        cc_tz = c_tz + (h_tc * gate_tcz) @ self.W_cxyz

        return cc_tx, cc_ty, cc_tz

    def state_aggregation(self, h_tc, h_tx, h_ty, h_tz, c_tc):
        r_tc = h_tc @ self.W_rc2
        r_tx = h_tx @ self.W_rx2
        r_ty = h_ty @ self.W_ry2
        r_tz = h_tz @ self.W_rz2

        state_tc = torch.cat((r_tc, r_tx, r_ty, r_tz), dim=-1) @ self.W_tcc
        state_tx = torch.cat((r_tc, r_tx, r_ty, r_tz), dim=-1) @ self.W_tcx
        state_ty = torch.cat((r_tc, r_tx, r_ty, r_tz), dim=-1) @ self.W_tcy
        state_tz = torch.cat((r_tc, r_tx, r_ty, r_tz), dim=-1) @ self.W_tcz

        gate_txc = state_tx @ self.W_xc + self.bias_xc
        gate_tyc = state_ty @ self.W_yc + self.bias_yc
        gate_tzc = state_tz @ self.W_zc + self.bias_cz

        state_xyz = torch.cat((state_tx.unsqueeze(1), state_ty.unsqueeze(1), state_tz.unsqueeze(1)), dim=1)  # Bx3xN
        query_tc = state_tc @ self.W_q  # BxN
        key_txyz = state_xyz @ self.W_k  # Bx3xN

        att = torch.bmm(key_txyz, query_tc.unsqueeze(2)).squeeze()  # Bx3
        att = torch.softmax(att, dim=-1)
        if len(att.shape) == 1:
            att = att.unsqueeze(0)

        cc_tc = att[:, 0:1] * (h_tx * gate_txc) + att[:, 1:2] * (h_ty * gate_tyc) + att[:, 2:3] * (h_tz * gate_tzc)
        cc_tc = cc_tc @ self.W_xyzc
        return cc_tc

    def forward(self, x, y, z, c, init_states_c=None, init_states_x=None, init_states_y=None, init_states_z=None):
        bs, seq_sz, _ = x.size()
        hidden_seq_c, hidden_seq_x, hidden_seq_y, hidden_seq_z = [], [], [], []

        def _init(init_states):
            if init_states is None:
                h = torch.zeros(bs, self.hidden_size, dtype=torch.double, device=x.device)
                c_ = torch.zeros(bs, self.hidden_size, dtype=torch.double, device=x.device)
                return h, c_
            return init_states

        h_tc, c_tc = _init(init_states_c)
        h_tx, c_tx = _init(init_states_x)
        h_ty, c_ty = _init(init_states_y)
        h_tz, c_tz = _init(init_states_z)

        HS = self.hidden_size
        for t in range(seq_sz):
            c_t = c[:, t, :]
            x_t = x[:, t, :]
            y_t = y[:, t, :]
            z_t = z[:, t, :]

            gates_c = c_t @ self.W_c + c_tc @ self.U_c + self.bias_c
            gates_x = x_t @ self.W_x + c_tx @ self.U_x + self.bias_x
            gates_y = y_t @ self.W_y + c_ty @ self.U_y + self.bias_y
            gates_z = z_t @ self.W_z + c_tz @ self.U_z + self.bias_z

            g_tc = torch.tanh(gates_c[:, HS * 2 : HS * 3])
            g_tx = torch.tanh(gates_x[:, HS * 2 : HS * 3])
            g_ty = torch.tanh(gates_y[:, HS * 2 : HS * 3])
            g_tz = torch.tanh(gates_z[:, HS * 2 : HS * 3])

            i_tc, f_tc, o_tc = torch.sigmoid(gates_c[:, :HS]), torch.sigmoid(gates_c[:, HS : HS * 2]), torch.sigmoid(gates_c[:, HS * 3 :])
            i_tx, f_tx, o_tx = torch.sigmoid(gates_x[:, :HS]), torch.sigmoid(gates_x[:, HS : HS * 2]), torch.sigmoid(gates_x[:, HS * 3 :])
            i_ty, f_ty, o_ty = torch.sigmoid(gates_y[:, :HS]), torch.sigmoid(gates_y[:, HS : HS * 2]), torch.sigmoid(gates_y[:, HS * 3 :])
            i_tz, f_tz, o_tz = torch.sigmoid(gates_z[:, :HS]), torch.sigmoid(gates_z[:, HS : HS * 2]), torch.sigmoid(gates_z[:, HS * 3 :])

            c_tc = f_tc * c_tc + i_tc * g_tc
            h_tc = o_tc * torch.tanh(c_tc)
            c_tx = f_tx * c_tx + i_tx * g_tx
            h_tx = o_tx * torch.tanh(c_tx)
            c_ty = f_ty * c_ty + i_ty * g_ty
            h_ty = o_ty * torch.tanh(c_ty)
            c_tz = f_tz * c_tz + i_tz * g_tz
            h_tz = o_tz * torch.tanh(c_tz)

            for _ in range(self.num_iter):
                cc_tx, cc_ty, cc_tz = self.state_extraction(h_tc, h_tx, h_ty, h_tz, c_tx, c_ty, c_tz)
                cc_tc = self.state_aggregation(h_tc, h_tx, h_ty, h_tz, c_tc)
                c_tc, c_tx, c_ty, c_tz = cc_tc, cc_tx, cc_ty, cc_tz
                h_tc = o_tc * torch.tanh(c_tc)
                h_tx = o_tx * torch.tanh(c_tx)
                h_ty = o_ty * torch.tanh(c_ty)
                h_tz = o_tz * torch.tanh(c_tz)

            hidden_seq_c.append(h_tc.unsqueeze(0))
            hidden_seq_x.append(h_tx.unsqueeze(0))
            hidden_seq_y.append(h_ty.unsqueeze(0))
            hidden_seq_z.append(h_tz.unsqueeze(0))

        hidden_seq_c = torch.cat(hidden_seq_c, dim=0).transpose(0, 1).contiguous()
        hidden_seq_x = torch.cat(hidden_seq_x, dim=0).transpose(0, 1).contiguous()
        hidden_seq_y = torch.cat(hidden_seq_y, dim=0).transpose(0, 1).contiguous()
        hidden_seq_z = torch.cat(hidden_seq_z, dim=0).transpose(0, 1).contiguous()
        return hidden_seq_c, (h_tc, c_tc), hidden_seq_x, (h_tx, c_tx), hidden_seq_y, (h_ty, c_ty), hidden_seq_z, (h_tz, c_tz)


class LBEBM3DInfer(nn.Module):
    """
    Inference-only LBEBM3D (plan generation + predict).
    Parameter names match the training checkpoint so load_state_dict works.
    """

    def __init__(
        self,
        enc_past_size: List[int],
        enc_dest_size: List[int],
        enc_latent_size: List[int],
        dec_size: List[int],
        predictor_size: List[int],
        fdim: int,
        zdim: int,
        ny: int,
        past_length: int,
        future_length: int,
        sub_goal_indexes: List[int],
        non_local_dim: int = 128,
        non_local_theta_size: List[int] = None,
        non_local_phi_size: List[int] = None,
        non_local_g_size: List[int] = None,
        lstm_layers: int = 1,
        state_layers: int = 3,
    ):
        super().__init__()
        self.zdim = zdim
        self.ny = ny
        self.past_length = past_length
        self.future_length = future_length
        self.sub_goal_indexes = sub_goal_indexes
        self.lstm_layers = lstm_layers
        self.state_layers = state_layers

        non_local_theta_size = non_local_theta_size or [256, 128, 64]
        non_local_phi_size = non_local_phi_size or [256, 128, 64]
        non_local_g_size = non_local_g_size or [256, 128, 64]

        self.encoder_past = MLP(input_dim=past_length * 3, output_dim=fdim, hidden_size=enc_past_size)
        self.encoder_dest = MLP(input_dim=len(sub_goal_indexes) * 3, output_dim=fdim, hidden_size=enc_dest_size)
        self.encoder_latent = MLP(input_dim=2 * fdim, output_dim=2 * zdim, hidden_size=enc_latent_size)

        self.decoder_z = MLP(input_dim=fdim + zdim, output_dim=len(sub_goal_indexes), hidden_size=dec_size)
        self.predictor_z = MLP(input_dim=2 * fdim, output_dim=1 * (future_length), hidden_size=predictor_size)
        self.decoder_x = MLP(input_dim=fdim + zdim, output_dim=len(sub_goal_indexes), hidden_size=dec_size)
        self.predictor_x = MLP(input_dim=2 * fdim, output_dim=1 * (future_length), hidden_size=predictor_size)
        self.decoder_y = MLP(input_dim=fdim + zdim, output_dim=len(sub_goal_indexes), hidden_size=dec_size)
        self.predictor_y = MLP(input_dim=2 * fdim, output_dim=1 * (future_length), hidden_size=predictor_size)

        self.non_local_theta = MLP(input_dim=fdim, output_dim=non_local_dim, hidden_size=non_local_theta_size)
        self.non_local_phi = MLP(input_dim=fdim, output_dim=non_local_dim, hidden_size=non_local_phi_size)
        self.non_local_g = MLP(input_dim=fdim, output_dim=fdim, hidden_size=non_local_g_size)

        self.EBM = nn.Sequential(
            nn.Linear(zdim + fdim, 200),
            nn.GELU(),
            nn.Linear(200, 200),
            nn.GELU(),
            nn.Linear(200, ny),
        )

        # refine
        self.encoder_futurex = MLP(input_dim=3, output_dim=fdim, hidden_size=[128, 64])
        self.encoder_futurey = MLP(input_dim=3, output_dim=fdim, hidden_size=[128, 64])
        self.encoder_futurez = MLP(input_dim=3, output_dim=fdim, hidden_size=[128, 64])
        self.encoder_futures = MLP(input_dim=2 * fdim, output_dim=fdim, hidden_size=[128, 64])
        self.sc_lstm = SC_LSTM(fdim, fdim * 4, 4, num_state=self.state_layers)
        self.decoder_offsetx = nn.Linear(fdim * 4, 1)
        self.decoder_offsety = nn.Linear(fdim * 4, 1)
        self.decoder_offsetz = nn.Linear(fdim * 4, 1)

        self.scale_weight_x = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale_weight_y = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale_weight_z = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale_weight_x.data.fill_(15)
        self.scale_weight_y.data.fill_(15)
        self.scale_weight_z.data.fill_(15)

    def ebm(self, z: torch.Tensor, condition: torch.Tensor, cls_output: bool = False) -> torch.Tensor:
        condition_encoding = condition.detach().clone()
        z_c = torch.cat((z, condition_encoding), dim=1)
        conditional_neg_energy = self.EBM(z_c)
        if cls_output:
            return -conditional_neg_energy
        return -conditional_neg_energy.logsumexp(dim=1)

    def sample_langevin_prior_z(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        e_l_steps: int,
        e_l_step_size: float,
        e_prior_sig: float,
        e_l_with_noise: bool,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = z.clone().detach()
        z.requires_grad_(True)
        for _ in range(e_l_steps):
            if y is None:
                en = self.ebm(z, condition)
            else:
                en = self.ebm(z, condition, cls_output=True)[range(z.size(0)), y]
            z_grad = torch.autograd.grad(en.sum(), z)[0]
            z.data = z.data - 0.5 * e_l_step_size * e_l_step_size * (z_grad + 1.0 / (e_prior_sig * e_prior_sig) * z.data)
            if e_l_with_noise:
                z.data += e_l_step_size * torch.randn_like(z).data
        return z.detach()

    def sample_plan(self, past_flat: torch.Tensor, langevin_cfg: Dict) -> torch.Tensor:
        # past_flat: (B, past_length*3), dtype double
        ftraj = self.encoder_past(past_flat)
        n = ftraj.size(0)
        e_init_sig = float(langevin_cfg["e_init_sig"])
        z0 = e_init_sig * torch.randn(n, self.zdim, dtype=torch.double, device=past_flat.device)
        z = self.sample_langevin_prior_z(
            z0,
            ftraj,
            e_l_steps=int(langevin_cfg["e_l_steps"]),
            e_l_step_size=float(langevin_cfg["e_l_step_size"]),
            e_prior_sig=float(langevin_cfg["e_prior_sig"]),
            e_l_with_noise=bool(langevin_cfg["e_l_with_noise"]),
        )
        decoder_input = torch.cat((ftraj, z), dim=1)
        dest_x = self.decoder_x(decoder_input)
        dest_y = self.decoder_y(decoder_input)
        dest_z = self.decoder_z(decoder_input)
        dest_xyz = torch.cat((dest_x.unsqueeze(2), dest_y.unsqueeze(2), dest_z.unsqueeze(2)), dim=-1)
        return dest_xyz.view(-1, dest_z.shape[1] * 3)

    @torch.no_grad()
    def predict(self, past_flat: torch.Tensor, plan_flat: torch.Tensor) -> torch.Tensor:
        ftraj = self.encoder_past(past_flat)
        plan_feat = self.encoder_dest(plan_flat)
        prediction_features = torch.cat((ftraj, plan_feat), dim=1)

        fut_z = self.predictor_z(prediction_features)
        fut_x = self.predictor_x(prediction_features)
        fut_y = self.predictor_y(prediction_features)

        # refinement via SC_LSTM offsets
        interpolated_future_x = fut_x
        interpolated_future_y = fut_y
        interpolated_future_z = fut_z

        for n in range(self.lstm_layers):
            if n != 0:
                interpolated_future_x = interpolated_future_x + offset_x
                interpolated_future_y = interpolated_future_y + offset_y
                interpolated_future_z = interpolated_future_z + offset_z

            trans_x = past_flat.reshape(-1, past_flat.shape[-1] // 3, 3)
            tem_x = torch.cat((trans_x[:, :, 0], interpolated_future_x), dim=-1)
            tem_y = torch.cat((trans_x[:, :, 1], interpolated_future_y), dim=-1)
            tem_z = torch.cat((trans_x[:, :, 2], interpolated_future_z), dim=-1)
            tem_vx = tem_x[:, 1:] - tem_x[:, :-1]
            tem_vy = tem_y[:, 1:] - tem_y[:, :-1]
            tem_vz = tem_z[:, 1:] - tem_z[:, :-1]

            future_c_x = tem_x[:, 2:]
            future_c_y = tem_y[:, 2:]
            future_c_z = tem_z[:, 2:]
            future_v_x = tem_vx[:, 1:]
            future_v_y = tem_vy[:, 1:]
            future_v_z = tem_vz[:, 1:]
            future_a_x = tem_vx[:, 1:] - tem_vx[:, :-1]
            future_a_y = tem_vy[:, 1:] - tem_vy[:, :-1]
            future_a_z = tem_vz[:, 1:] - tem_vz[:, :-1]

            cva_x = torch.cat((future_c_x.unsqueeze(2), future_v_x.unsqueeze(2), future_a_x.unsqueeze(2)), dim=-1)
            cva_y = torch.cat((future_c_y.unsqueeze(2), future_v_y.unsqueeze(2), future_a_y.unsqueeze(2)), dim=-1)
            cva_z = torch.cat((future_c_z.unsqueeze(2), future_v_z.unsqueeze(2), future_a_z.unsqueeze(2)), dim=-1)

            cva_featurex = self.encoder_futurex(cva_x)
            cva_featurey = self.encoder_futurey(cva_y)
            cva_featurez = self.encoder_futurez(cva_z)

            cva_feat = torch.cat((cva_featurex.unsqueeze(3), cva_featurey.unsqueeze(3), cva_featurez.unsqueeze(3)), dim=-1)
            cva_maxfeat, _ = torch.max(cva_feat, dim=-1)
            cva_meanfeat = torch.mean(cva_feat, dim=-1)
            cva_features = torch.cat((cva_maxfeat, cva_meanfeat), dim=-1)
            cva_features = self.encoder_futures(cva_features)

            _, _, h_x, _, h_y, _, h_z, _ = self.sc_lstm(cva_featurex, cva_featurey, cva_featurez, cva_features)
            length = interpolated_future_x.shape[1]
            offset_x = self.scale_weight_x * torch.sigmoid(self.decoder_offsetx(h_x)[:, -length:, 0])
            offset_y = self.scale_weight_y * torch.sigmoid(self.decoder_offsety(h_y)[:, -length:, 0])
            offset_z = self.scale_weight_z * torch.sigmoid(self.decoder_offsetz(h_z)[:, -length:, 0])

        interpolated_future_x = interpolated_future_x + offset_x
        interpolated_future_y = interpolated_future_y + offset_y
        interpolated_future_z = interpolated_future_z + offset_z

        fut = torch.cat((interpolated_future_x.unsqueeze(2), interpolated_future_y.unsqueeze(2), interpolated_future_z.unsqueeze(2)), dim=-1)
        return fut  # (B, T, 3)


def _extract_hidden_sizes(state_dict: Dict[str, torch.Tensor], module_name: str) -> List[int]:
    hidden: List[int] = []
    i = 0
    while f"{module_name}.layers.{i}.weight" in state_dict:
        w = state_dict[f"{module_name}.layers.{i}.weight"]
        hidden.append(int(w.shape[0]))
        i += 1
    if len(hidden) > 1:
        return hidden[:-1]
    return hidden


def infer_model_params_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict:
    enc_past = _extract_hidden_sizes(state_dict, "encoder_past") or [512, 256]
    enc_dest = _extract_hidden_sizes(state_dict, "encoder_dest") or [256, 128]
    enc_latent = _extract_hidden_sizes(state_dict, "encoder_latent") or [256, 512]
    dec = _extract_hidden_sizes(state_dict, "decoder_z") or [1024, 512, 1024]
    pred = _extract_hidden_sizes(state_dict, "predictor_x") or [1024, 512, 256]

    # infer fdim, zdim, lengths
    last_enc_past_idx = len(enc_past)
    fdim = int(state_dict[f"encoder_past.layers.{last_enc_past_idx}.weight"].shape[0])

    last_enc_latent_idx = len(enc_latent)
    zdim = int(state_dict[f"encoder_latent.layers.{last_enc_latent_idx}.weight"].shape[0] // 2)

    # lengths from layer input/output shapes
    past_input_dim = int(state_dict["encoder_past.layers.0.weight"].shape[1])
    past_length = past_input_dim // 3

    future_length = int(state_dict[f"predictor_x.layers.{len(pred)}.weight"].shape[0])

    # number of subgoals from decoder output dim
    num_subgoals = int(state_dict[f"decoder_x.layers.{len(dec)}.weight"].shape[0])

    # infer ny from EBM last linear
    # EBM is Sequential: 0 Linear,1 GELU,2 Linear,3 GELU,4 Linear
    ny = int(state_dict["EBM.4.weight"].shape[0]) if "EBM.4.weight" in state_dict else 1

    return {
        "enc_past_size": enc_past,
        "enc_dest_size": enc_dest,
        "enc_latent_size": enc_latent,
        "dec_size": dec,
        "predictor_size": pred,
        "fdim": fdim,
        "zdim": zdim,
        "ny": ny,
        "past_length": past_length,
        "future_length": future_length,
        "num_subgoals": num_subgoals,
    }


def load_past_trajectory(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        arr = np.load(path)
    elif path.lower().endswith(".csv"):
        # try header/no-header
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
            if arr.ndim == 1:
                arr = np.loadtxt(path, delimiter=",", skiprows=0)
        except Exception:
            arr = np.loadtxt(path, delimiter=",", skiprows=0)
    else:
        raise ValueError("Unsupported input file type. Use .npy or .csv")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected shape (T, >=3). Got {arr.shape}")
    return arr[:, :3]


def save_csv(path: str, arr: np.ndarray):
    header = "x,y,z"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def load_from_pickle(path: str, sample_idx: int, past_length: int, future_length: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load past (and optional future) from saved_data.pickle."""
    data = pickle.load(open(path, "rb"))
    if not isinstance(data, dict) or "src" not in data:
        raise ValueError("pickle must be a dict with key 'src'")
    src = np.asarray(data["src"])
    if sample_idx < 0 or sample_idx >= src.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} out of range (0..{src.shape[0]-1})")
    past = src[sample_idx, :past_length, :3]
    if past.shape[0] != past_length:
        raise ValueError(f"src has only {past.shape[0]} past steps, expected {past_length}")

    future = None
    if future_length is not None and "trg" in data:
        trg = np.asarray(data["trg"])
        if trg.shape[0] > sample_idx:
            future = trg[sample_idx, :future_length, :3]
    return past, future


def load_from_npz(
    input_path: str,
    sample_idx: int,
    past_length: int,
    output_path: Optional[str] = None,
    future_length: Optional[int] = None,
    agent_idx: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load past trajectory (and optional ground truth) from npz files (GNN format)
    
    Args:
        input_path: Path to input npz (seq_in, samples, agents, 3)
        sample_idx: Sample index
        past_length: Expected past length
        output_path: Optional output npz path for GT
        future_length: Expected future length (if loading GT)
        agent_idx: Agent index to extract (default 0)
    
    Returns:
        past: (past_length, 3) past trajectory
        gt_future: (future_length, 3) or None
    """
    data_in = np.load(input_path)
    if "data" not in data_in:
        raise ValueError("Input npz must contain 'data' key")
    
    X_raw = data_in["data"]  # (seq_in, samples, agents, 3)
    X = np.transpose(X_raw, (1, 0, 2, 3))  # (samples, seq_in, agents, 3)
    
    if sample_idx < 0 or sample_idx >= X.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} out of range (0..{X.shape[0]-1})")
    
    sample = X[sample_idx]  # (seq_in, agents, 3)
    if sample.shape[0] != past_length:
        raise ValueError(f"Input seq {sample.shape[0]} != expected {past_length}")
    
    if agent_idx >= sample.shape[1]:
        raise IndexError(f"agent_idx {agent_idx} out of range (0..{sample.shape[1]-1})")
    
    past = sample[:, agent_idx, :]  # (seq_in, 3)
    
    gt_future = None
    if output_path and future_length is not None:
        data_out = np.load(output_path)
        if "data" not in data_out:
            raise ValueError("Output npz must contain 'data' key")
        
        Y_raw = data_out["data"]  # (seq_out, samples, agents, 3)
        Y = np.transpose(Y_raw, (1, 0, 2, 3))  # (samples, seq_out, agents, 3)
        
        gt_sample = Y[sample_idx]  # (seq_out, agents, 3)
        gt_future = gt_sample[:, agent_idx, :]  # (seq_out, 3)
    
    return past, gt_future


def smooth_with_physical_constraints(
    history_abs: np.ndarray,
    pred_abs: np.ndarray,
    dt: float = 1.0,
    smoothing_weight: float = 0.3,
) -> np.ndarray:
    """
    简单的物理约束平滑：
    - 使用历史末端速度/加速度作为参考，限制预测加速度幅值
    - 保持首点与历史末尾连续，减少可视化断点
    """
    history = np.asarray(history_abs, dtype=np.float64)
    pred = np.asarray(pred_abs, dtype=np.float64)
    if history.shape[0] < 2 or pred.shape[0] == 0:
        return pred

    hist_vel = np.diff(history, axis=0) / dt
    if hist_vel.shape[0] == 0:
        hist_vel = np.zeros((1, 3), dtype=np.float64)
    last_vel = hist_vel[-min(3, len(hist_vel)) :].mean(axis=0)

    hist_acc = np.diff(hist_vel, axis=0) / dt if hist_vel.shape[0] > 1 else np.zeros((1, 3), dtype=np.float64)
    avg_acc = hist_acc.mean(axis=0) if hist_acc.size else np.zeros_like(last_vel)

    max_acc = np.linalg.norm(hist_acc, axis=1).max() if hist_acc.size else 0.0
    max_acc = max(max_acc, 1e-3)

    current_pos = history[-1].copy()
    current_vel = last_vel.copy()
    smoothed = []

    for t in range(pred.shape[0]):
        desired_pos = pred[t]
        desired_vel = (desired_pos - current_pos) / dt

        raw_acc = (desired_vel - current_vel) / dt
        constrained_acc = (1 - smoothing_weight) * raw_acc + smoothing_weight * avg_acc

        acc_norm = np.linalg.norm(constrained_acc) + 1e-8
        if acc_norm > 2 * max_acc:
            constrained_acc = constrained_acc * (2 * max_acc / acc_norm)

        new_vel = current_vel + constrained_acc * dt
        current_pos = current_pos + new_vel * dt
        current_vel = new_vel
        smoothed.append(current_pos.copy())

    if smoothed:
        smoothed[0] = history[-1].copy()
    return np.asarray(smoothed)


def plot_prediction(
    past: np.ndarray,
    pred: np.ndarray,
    future: Optional[np.ndarray],
    out_path: str,
):
    """Comprehensive visualization: 3D + XY + XZ + YZ + per-step axis error + step error bar."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def _error_stats(gt: np.ndarray, pd: np.ndarray):
        # gt/pd shape: (B, T, 3); we want per-step stats => mean over batch (axis=0)
        err_axis = np.abs(pd - gt).mean(axis=0)          # (T, 3)
        err_l2 = np.linalg.norm(pd - gt, axis=2)         # (B, T)
        err_l2_mean = err_l2.mean(axis=0)                # (T,)
        return err_axis, err_l2_mean

    fig = plt.figure(figsize=(16, 10))

    # 1) 3D
    ax1 = fig.add_subplot(231, projection="3d")
    ax1.plot(past[:, 0], past[:, 1], past[:, 2], "b-o", label="past")
    if future is not None:
        ax1.plot(future[:, 0], future[:, 1], future[:, 2], "g-s", label="gt")
    ax1.plot(pred[:, 0], pred[:, 1], pred[:, 2], "r-^", label="pred")
    ax1.set_title("3D")
    ax1.legend()

    # 2) XY
    ax2 = fig.add_subplot(232)
    ax2.plot(past[:, 0], past[:, 1], "b-o", label="past")
    if future is not None:
        ax2.plot(future[:, 0], future[:, 1], "g-s", label="gt")
    ax2.plot(pred[:, 0], pred[:, 1], "r-^", label="pred")
    ax2.set_title("XY")
    ax2.axis("equal")
    ax2.legend()

    # 3) XZ
    ax3 = fig.add_subplot(233)
    ax3.plot(past[:, 0], past[:, 2], "b-o", label="past")
    if future is not None:
        ax3.plot(future[:, 0], future[:, 2], "g-s", label="gt")
    ax3.plot(pred[:, 0], pred[:, 2], "r-^", label="pred")
    ax3.set_title("XZ")
    ax3.axis("equal")
    ax3.legend()

    # 4) YZ
    ax4 = fig.add_subplot(234)
    ax4.plot(past[:, 1], past[:, 2], "b-o", label="past")
    if future is not None:
        ax4.plot(future[:, 1], future[:, 2], "g-s", label="gt")
    ax4.plot(pred[:, 1], pred[:, 2], "r-^", label="pred")
    ax4.set_title("YZ")
    ax4.axis("equal")
    ax4.legend()

    # 5) per-step axis MAE
    ax5 = fig.add_subplot(235)
    if future is not None:
        err_axis, err_l2_mean = _error_stats(future[None, ...], pred[None, ...])
        steps = np.arange(pred.shape[0])
        ax5.plot(steps, err_axis[:, 0], "r-^", label="|x|")
        ax5.plot(steps, err_axis[:, 1], "g-s", label="|y|")
        ax5.plot(steps, err_axis[:, 2], "b-o", label="|z|")
        ax5.set_ylabel("MAE (m)")
        ax5.set_title("Per-step Axis MAE")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No GT provided", ha="center", va="center")
        ax5.axis("off")

    # 6) per-step L2
    ax6 = fig.add_subplot(236)
    if future is not None:
        steps = np.arange(pred.shape[0])
        _, err_l2_mean = _error_stats(future[None, ...], pred[None, ...])
        bars = ax6.bar(steps, err_l2_mean, color="tab:red", alpha=0.7, edgecolor="darkred")
        for i, err in enumerate(err_l2_mean):
            ax6.text(i, err + 1e-3, f"{err:.3f}", ha="center", va="bottom", fontsize=8)
        ax6.set_ylabel("L2 error (m)")
        ax6.set_title("Per-step Position Error")
        ax6.grid(True, axis="y", alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "No GT provided", ha="center", va="center")
        ax6.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="LBEBM3D baseline inference (standalone)")
    ap.add_argument("--model_path", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Past trajectory file (.npy/.csv) or saved_data.pickle (use --sample_idx)",
    )
    ap.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="When input_path is a saved_data.pickle, choose this sample index",
    )
    ap.add_argument("--output_path", type=str, default="", help="Optional output (.npy or .csv)")
    ap.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
    ap.add_argument("--data_scale", type=float, default=1.0, help="same as training args.data_scale")
    ap.add_argument("--sub_goal_indexes", type=str, default="", help="comma-separated, e.g. 2,5,7,9 (optional)")
    ap.add_argument("--e_init_sig", type=float, default=2.0)
    ap.add_argument("--e_prior_sig", type=float, default=2.0)
    ap.add_argument("--e_l_steps", type=int, default=20)
    ap.add_argument("--e_l_step_size", type=float, default=0.4)
    ap.add_argument("--e_l_with_noise", action="store_true", help="enable Langevin noise (default: off)")
    ap.add_argument("--plot", action="store_true", help="save a quick plot next to output (requires matplotlib)")
    ap.add_argument("--smooth", action="store_true", help="apply velocity/acc smoothing for continuity")
    ap.add_argument("--dt", type=float, default=1.0, help="time step between frames for smoothing")
    ap.add_argument("--plot_path", type=str, default="", help="optional png path for visualization")
    ap.add_argument("--output_dir", type=str, default="", help="folder to save outputs (csv/npy/png)")
    ap.add_argument("--input_npz", type=str, default="", help="Path to input npz (GNN format, overrides --input_path)")
    ap.add_argument("--output_npz", type=str, default="", help="Path to output npz (GT for visualization)")
    ap.add_argument("--agent_idx", type=int, default=0, help="Agent index when using npz (default: 0)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    params = infer_model_params_from_state_dict(state_dict)
    if args.sub_goal_indexes.strip():
        sub_goal_indexes = [int(x) for x in args.sub_goal_indexes.split(",") if x.strip() != ""]
    else:
        # default: spread across horizon, must match num_subgoals
        if params["num_subgoals"] == 4 and params["future_length"] >= 10:
            sub_goal_indexes = [2, 5, 7, 9]
        else:
            # fallback: pick last K steps evenly
            sub_goal_indexes = list(np.linspace(0, params["future_length"] - 1, params["num_subgoals"], dtype=int))

    model = LBEBM3DInfer(
        enc_past_size=params["enc_past_size"],
        enc_dest_size=params["enc_dest_size"],
        enc_latent_size=params["enc_latent_size"],
        dec_size=params["dec_size"],
        predictor_size=params["predictor_size"],
        fdim=params["fdim"],
        zdim=params["zdim"],
        ny=params["ny"],
        past_length=params["past_length"],
        future_length=params["future_length"],
        sub_goal_indexes=sub_goal_indexes,
    ).double()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    if args.input_npz:
        # Load from npz (GNN format)
        output_npz_arg = args.output_npz if args.output_npz else None
        past, gt_future = load_from_npz(
            args.input_npz,
            args.sample_idx,
            params["past_length"],
            output_npz_arg,
            params["future_length"],
            args.agent_idx
        )
    elif args.input_path.lower().endswith(".pickle"):
        past, gt_future = load_from_pickle(args.input_path, args.sample_idx, params["past_length"], params["future_length"])
    else:
        past = load_past_trajectory(args.input_path)  # (T,3)
        gt_future = None
    if past.shape[0] != params["past_length"]:
        raise ValueError(f"Input past length {past.shape[0]} != model past_length {params['past_length']}")

    # match training preprocessing: shift by last obs
    last = past[-1:].copy()
    past_rel = (past - last) * args.data_scale

    past_t = torch.from_numpy(past_rel.reshape(1, -1)).to(device=device, dtype=torch.double)
    langevin_cfg = {
        "e_init_sig": args.e_init_sig,
        "e_prior_sig": args.e_prior_sig,
        "e_l_steps": args.e_l_steps,
        "e_l_step_size": args.e_l_step_size,
        "e_l_with_noise": args.e_l_with_noise,
    }
    plan = model.sample_plan(past_t, langevin_cfg)
    fut = model.predict(past_t, plan).detach().cpu().numpy()[0]  # (T_pred,3) in relative coords * data_scale

    fut = fut / args.data_scale + last  # back to absolute

    # 平滑与对齐（可选）
    if args.smooth:
        fut = smooth_with_physical_constraints(past, fut, dt=args.dt, smoothing_weight=0.3)

    # 首点强制与历史末尾对齐，避免可视化跳变
    if fut.shape[0] > 0:
        fut[0] = past[-1]

    # print
    np.set_printoptions(suppress=True, precision=6)
    print("Predicted future trajectory (T_pred,3):")
    print(fut)

    # resolve output directory
    out_dir = args.output_dir or (os.path.dirname(args.output_path) if args.output_path else ".")
    os.makedirs(out_dir or ".", exist_ok=True)

    # save prediction
    if args.output_path:
        out = os.path.join(out_dir, os.path.basename(args.output_path))
        if out.lower().endswith(".npy"):
            np.save(out, fut)
        elif out.lower().endswith(".csv"):
            save_csv(out, fut)
        else:
            raise ValueError("output_path must end with .npy or .csv")
        print(f"Saved prediction to: {out}")

    # plot: if gt available, draw comparison
    if args.plot:
        try:
            base_name = os.path.splitext(os.path.basename(args.output_path) or f"sample_{args.sample_idx}")[0]
            png = args.plot_path or os.path.join(out_dir, base_name + "_plot.png")
            plot_prediction(past, fut, gt_future, png)
            print(f"Saved plot to: {png}")
        except Exception as e:
            print(f"[WARN] plot failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

