# -*- coding: utf-8 -*-

"""
Define the boundary-aware, physically structured GNN4CF architecture.

The model uses type-specific node and edge encoders, explicit boundary-interior
coupling, interior message passing, boundary-preservation skip connections,
and a state-aware decoder with residual water-depth prediction. Boundary ghost
forcing is injected through boundary conditioning; computational-node rainfall
is reinforced through dynamic-only FiLM conditioning before interior propagation.
Physical structure is expressed through node types, graph connections, and
separate coupling/interior processors; training objectives are defined separately.
"""

import warnings
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Dict, List, Optional
import torch.nn.functional as F


# =============================================================================
# Helper Functions
# =============================================================================

def _resolve_component_hyperparams(
    shared_steps: int,
    shared_layers: int,
    shared_hidden: int,
    component_steps: Optional[int] = None,
    component_layers: Optional[int] = None,
    component_hidden: Optional[int] = None
) -> tuple[int, int, int]:
    """
    Resolves component-specific hyperparameters with fallback to shared values.

    This helper function centralizes the fallback logic for component-specific
    hyperparameters. If a component-specific value is provided (not None), it is used.
    Otherwise, the shared value is used as the default.

    Args:
        shared_steps: Shared message passing steps (default)
        shared_layers: Shared MLP layers (default)
        shared_hidden: Shared MLP hidden dimension (default)
        component_steps: Optional component-specific message passing steps
        component_layers: Optional component-specific MLP layers
        component_hidden: Optional component-specific MLP hidden dimension

    Returns:
        tuple[int, int, int]: (resolved_steps, resolved_layers, resolved_hidden)
    """
    return (
        component_steps if component_steps is not None else shared_steps,
        component_layers if component_layers is not None else shared_layers,
        component_hidden if component_hidden is not None else shared_hidden
    )


def init_weights(layer):
    """Initializes weights using Xavier normal initialization."""
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.normal_(layer.bias)


def build_mlp(input_size: int, hidden_layer_sizes: List[int], output_size: int = None,
              output_activation: nn.Module = nn.Identity, activation: nn.Module = nn.ReLU) -> nn.Module:
    """Builds an MLP with the specified structure."""
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)
    nlayers = len(layer_sizes) - 1
    act = [activation for _ in range(nlayers)]
    act[-1] = output_activation
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())
    return mlp


class BoundaryConditioning(nn.Module):
    """
    Node-wise boundary conditioning for boundary ghost nodes.

    Boundary conditioning combines the following operations:
    1. Encode current boundary state, optional boundary history, and optional geometry
    2. Modulate state latents with geometry via element-wise multiplication
    3. Combine the enabled terms with either:
       - concat -> update MLP -> residual reinjection
       - FiLM gamma/beta modulation

    Notes about feature extraction:
    - Geometry uses the first two static node features, assumed to be (x, y)
    - Boundary sea-level and trend are taken from the last two dynamic variables
      at each history step, matching the current config ordering:
      [state_vars..., rainfall_rate, acc_rainfall, sea_level, sea_level_trend]
    """

    def __init__(
        self,
        latent_dim: int,
        model_history_steps: int,
        n_dynamic_node_vars: int,
        n_static_node_bghost: int,
        enabled: bool = False,
        mode: str = "concat",
        inject_every_processor_step: bool = True,
        use_current_state: bool = True,
        use_history_state: bool = True,
        use_geometry: bool = True,
        geometry_interaction: str = "multiply",
        current_encoder_hidden_dim: int = 64,
        history_encoder_hidden_dim: int = 64,
        geometry_encoder_hidden_dim: int = 64,
        update_mlp_hidden_dim: int = 128,
        film_hidden_dim: int = 128,
    ):
        super().__init__()
        self.enabled = enabled
        self.mode = mode
        self.inject_every_processor_step = inject_every_processor_step
        self.use_current_state = use_current_state
        self.use_history_state = use_history_state
        self.use_geometry = use_geometry
        self.geometry_interaction = geometry_interaction
        self.latent_dim = latent_dim
        self.model_history_steps = model_history_steps
        self.n_dynamic_node_vars = n_dynamic_node_vars
        self.n_static_node_bghost = n_static_node_bghost

        if not self.enabled:
            return

        if self.n_dynamic_node_vars < 2:
            raise ValueError(
                "Boundary conditioning requires the last two dynamic node variables "
                "to represent sea_level and sea_level_trend."
            )

        if self.use_current_state:
            self.current_encoder = nn.Sequential(
                build_mlp(2, [current_encoder_hidden_dim], latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.current_encoder = None

        if self.use_history_state:
            history_in_dim = 2 * model_history_steps
            self.history_encoder = nn.Sequential(
                build_mlp(history_in_dim, [history_encoder_hidden_dim], latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.history_encoder = None

        if self.use_geometry:
            if self.n_static_node_bghost < 2:
                raise ValueError(
                    "Boundary conditioning with geometry requires at least two static "
                    "boundary-node features for node coordinates."
                )
            self.geometry_encoder = nn.Sequential(
                build_mlp(2, [geometry_encoder_hidden_dim], latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.geometry_encoder = None

        num_terms = int(self.use_current_state) + int(self.use_history_state)
        fused_dim = max(1, num_terms) * latent_dim

        if self.mode == "concat":
            self.update_mlp = nn.Sequential(
                build_mlp(fused_dim, [update_mlp_hidden_dim], latent_dim),
                nn.LayerNorm(latent_dim),
            )
            self.film_generator = None
        elif self.mode == "film":
            self.film_generator = nn.Sequential(
                build_mlp(fused_dim, [film_hidden_dim], 2 * latent_dim),
                nn.LayerNorm(2 * latent_dim),
            )
            self.update_mlp = None
        else:
            raise ValueError(f"Unsupported boundary conditioning mode: {self.mode}")

    def build_context(self, x: torch.Tensor, node_type: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if not self.enabled:
            return None

        boundary_mask = node_type == 1
        if not boundary_mask.any():
            return None

        x_b = x[boundary_mask]
        dynamic_flat = x_b[:, self.n_static_node_bghost:]
        dynamic_hist = dynamic_flat.reshape(-1, self.model_history_steps, self.n_dynamic_node_vars)

        sea_level_hist = dynamic_hist[:, :, -2]
        sea_level_trend_hist = dynamic_hist[:, :, -1]

        context: Dict[str, torch.Tensor] = {"boundary_mask": boundary_mask}

        if self.use_current_state and self.current_encoder is not None:
            current_input = torch.cat(
                [sea_level_hist[:, -1:].contiguous(), sea_level_trend_hist[:, -1:].contiguous()],
                dim=-1,
            )
            context["z_cur"] = self.current_encoder(current_input)

        if self.use_history_state and self.history_encoder is not None:
            history_input = torch.cat([sea_level_hist, sea_level_trend_hist], dim=-1)
            context["z_hist"] = self.history_encoder(history_input)

        if self.use_geometry and self.geometry_encoder is not None:
            geometry_input = x_b[:, :2]
            context["z_geo"] = self.geometry_encoder(geometry_input)

        return context

    def _modulate_with_geometry(
        self,
        state_latent: torch.Tensor,
        geometry_latent: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if geometry_latent is None or self.geometry_interaction == "none":
            return state_latent
        if self.geometry_interaction != "multiply":
            raise ValueError(f"Unsupported geometry interaction: {self.geometry_interaction}")
        return state_latent * geometry_latent

    def apply_bc_conditioning(self, node_latent: torch.Tensor, context: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if not self.enabled or context is None:
            return node_latent

        boundary_mask = context["boundary_mask"]
        if not boundary_mask.any():
            return node_latent

        terms = []
        z_geo = context.get("z_geo")

        if "z_cur" in context:
            terms.append(self._modulate_with_geometry(context["z_cur"], z_geo))
        if "z_hist" in context:
            terms.append(self._modulate_with_geometry(context["z_hist"], z_geo))

        if not terms:
            return node_latent

        cond = terms[0] if len(terms) == 1 else torch.cat(terms, dim=-1)
        h_b = node_latent[boundary_mask]

        if self.mode == "concat":
            update = self.update_mlp(cond)
            h_b = h_b + update
        else:
            gamma_beta = self.film_generator(cond)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            h_b = (1.0 + torch.tanh(gamma)) * h_b + beta

        node_latent = node_latent.clone()
        node_latent[boundary_mask] = h_b
        return node_latent


class RainfallConditioning(nn.Module):
    """
    Dynamic-only rainfall conditioning for computational nodes.

    Boundary conditioning handles coastal/sea-level forcing on boundary ghost
    nodes. This module addresses a different failure mode: rainfall-driven
    interior flooding can become weak after encoding and repeated message
    passing. Rainfall FiLM therefore reinjects only the current dynamic rainfall
    forcing into computational-node latents before/during the interior processor.

    Static terrain/runoff features are deliberately not reinjected here: they
    are already present in the raw node features and encoded by the node encoder
    at the start of the model.

    Feature extraction is name-based rather than position-based. When enabled,
    the model-construction code must pass:
    - dynamic_input_feature_names: state variables + driver variables in the
      per-history-step order used in x

    Missing requested rainfall dynamic features are errors because they define
    the forcing signal for this module.
    """

    def __init__(
        self,
        latent_dim: int,
        model_history_steps: int,
        n_dynamic_node_vars: int,
        n_static_node_comp: int,
        enabled: bool = False,
        mode: str = "film",
        inject_before_interior: bool = True,
        inject_every_interior_step: bool = True,
        rainfall_features: Optional[List[str]] = None,
        current_encoder_hidden_dim: int = 64,
        film_hidden_dim: int = 64,
        dynamic_input_feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.mode = mode
        self.inject_before_interior = inject_before_interior
        self.inject_every_interior_step = inject_every_interior_step
        self.latent_dim = latent_dim
        self.model_history_steps = model_history_steps
        self.n_dynamic_node_vars = n_dynamic_node_vars
        self.n_static_node_comp = n_static_node_comp
        self.rainfall_features = rainfall_features or ["rainfall_rate", "acc_rainfall"]
        self.dynamic_input_feature_names = dynamic_input_feature_names
        self.rainfall_feature_indices: List[int] = []
        self.current_encoder = None
        self.film_generator = None

        if not self.enabled:
            return

        if self.mode != "film":
            raise ValueError(
                "RainfallConditioning currently supports mode='film' only. "
                "Set rainfall_conditioning.enabled=false to disable rainfall conditioning."
            )

        if dynamic_input_feature_names is None:
            raise ValueError(
                "rainfall_conditioning_enabled=True requires dynamic_input_feature_names. "
                "Pass dynamic_input_state_variables + dynamic_input_drivers in the same "
                "per-history-step order used to build node features."
            )

        if len(dynamic_input_feature_names) != n_dynamic_node_vars:
            warnings.warn(
                "dynamic_input_feature_names length does not match n_dynamic_node_vars. "
                "Rainfall conditioning will still resolve by name, but verify the "
                "model-construction feature order.",
                RuntimeWarning,
            )

        self.rainfall_feature_indices = self._resolve_required_indices(
            available_names=dynamic_input_feature_names,
            requested_names=self.rainfall_features,
            feature_group="dynamic rainfall",
        )

        self.current_encoder = nn.Sequential(
            build_mlp(len(self.rainfall_feature_indices), [current_encoder_hidden_dim], latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.film_generator = nn.Sequential(
            build_mlp(latent_dim, [film_hidden_dim], 2 * latent_dim),
            nn.LayerNorm(2 * latent_dim),
        )

    @staticmethod
    def _canonical_name(name: str) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _resolve_required_indices(
        cls,
        available_names: List[str],
        requested_names: List[str],
        feature_group: str,
    ) -> List[int]:
        lookup = {cls._canonical_name(name): idx for idx, name in enumerate(available_names)}
        indices: List[int] = []
        missing: List[str] = []
        for name in requested_names:
            key = cls._canonical_name(name)
            if key in lookup:
                indices.append(lookup[key])
            else:
                missing.append(name)
        if missing:
            raise ValueError(
                f"Missing required {feature_group} feature(s): {missing}. "
                f"Available features: {available_names}"
            )
        return indices

    def build_context(self, x: torch.Tensor, node_type: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if not self.enabled:
            return None

        comp_mask = node_type == 0
        if not comp_mask.any():
            return None

        x_comp = x[comp_mask]
        expected_dynamic_width = self.model_history_steps * self.n_dynamic_node_vars
        dynamic_flat = x_comp[:, self.n_static_node_comp:self.n_static_node_comp + expected_dynamic_width]
        if dynamic_flat.size(-1) != expected_dynamic_width:
            raise ValueError(
                "Rainfall conditioning could not slice the expected dynamic history "
                f"width ({expected_dynamic_width}); got {dynamic_flat.size(-1)}. "
                "Check n_static_node_comp, n_dynamic_node_vars, and model_history_steps."
            )
        dynamic_hist = dynamic_flat.reshape(-1, self.model_history_steps, self.n_dynamic_node_vars)
        rain_hist = dynamic_hist[:, :, self.rainfall_feature_indices]
        current_rain = rain_hist[:, -1, :].contiguous()

        context: Dict[str, torch.Tensor] = {"comp_mask": comp_mask}
        # Current rainfall is the last history step for each computational node.
        context["z_cur"] = self.current_encoder(current_rain)

        # Lightweight debug metadata for callers/tests. It avoids per-forward
        # printing while preserving the key diagnostics: number of conditioned
        # nodes and resolved dynamic rainfall feature indices.
        context["rainfall_feature_indices"] = torch.tensor(
            self.rainfall_feature_indices,
            device=x.device,
            dtype=torch.long,
        )
        return context

    def apply_rainfall_conditioning(
        self,
        node_latent: torch.Tensor,
        context: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if not self.enabled or context is None:
            return node_latent

        comp_mask = context["comp_mask"]
        if not comp_mask.any():
            return node_latent

        cond = context.get("z_cur")
        if cond is None:
            return node_latent

        h_comp = node_latent[comp_mask]

        gamma_beta = self.film_generator(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        h_comp = (1.0 + torch.tanh(gamma)) * h_comp + beta

        node_latent = node_latent.clone()
        node_latent[comp_mask] = h_comp
        return node_latent


# =============================================================================
# 1. Specialized Encoder
# =============================================================================

class Encoder(nn.Module):
    """
    Specialized Encoder for HEC-RAS 2D Mesh.

    This encoder has four separate MLPs to encode the distinct feature sets of:
    1.  Computational nodes ('comp', type 0)
    2.  Boundary ghost nodes ('bghost', type 1)
    3.  Internal edges ('internal', type 0)
    4.  Boundary edges ('boundary', type 1)

    All components are projected to a common `latent_dim`.
    """

    def __init__(self,
                 n_node_comp_in: int,
                 n_node_bghost_in: int,
                 n_edge_internal_in: int,
                 n_edge_boundary_in: int,
                 latent_dim: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # --- Node Encoders ---
        self.node_encoder_comp = nn.Sequential(
            build_mlp(n_node_comp_in, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.node_encoder_bghost = nn.Sequential(
            build_mlp(n_node_bghost_in, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # --- Edge Encoders ---
        self.edge_encoder_internal = nn.Sequential(
            build_mlp(n_edge_internal_in, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.edge_encoder_boundary = nn.Sequential(
            build_mlp(n_edge_boundary_in, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor,
                node_type: torch.Tensor, edge_type: torch.Tensor):
        """
        Applies the correct encoder to each component of the graph.

        Args:
            x (torch.Tensor): Node feature tensor [num_nodes, n_node_features_raw]
            edge_attr (torch.Tensor): Edge feature tensor [num_edges, n_edge_features_raw]
            node_type (torch.Tensor): Long tensor [num_nodes] (0=comp, 1=bghost, 2=non_bc_ghost)
            edge_type (torch.Tensor): Long tensor [num_edges] (0=internal, 1=boundary)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - node_latent: [num_nodes, latent_dim]
                - edge_latent: [num_edges, latent_dim]
        """
        # Initialize latent representations with zeros
        node_latent = torch.zeros(x.size(0), self.latent_dim, device=x.device, dtype=x.dtype)
        edge_latent = torch.zeros(edge_attr.size(0), self.latent_dim, device=edge_attr.device, dtype=edge_attr.dtype)

        # --- Node Encoding ---
        mask_node_comp = (node_type == 0)
        mask_node_bghost = (node_type == 1)

        if mask_node_comp.any():
            node_latent[mask_node_comp] = self.node_encoder_comp(x[mask_node_comp])
        if mask_node_bghost.any():
            node_latent[mask_node_bghost] = self.node_encoder_bghost(x[mask_node_bghost])

        # --- Edge Encoding ---
        mask_edge_internal = (edge_type == 0)
        mask_edge_boundary = (edge_type == 1)

        if mask_edge_internal.any():
            edge_latent[mask_edge_internal] = self.edge_encoder_internal(edge_attr[mask_edge_internal])
        if mask_edge_boundary.any():
            edge_latent[mask_edge_boundary] = self.edge_encoder_boundary(edge_attr[mask_edge_boundary])

        return node_latent, edge_latent


# =============================================================================
# 2. INTERIOR ONLY: Message Passing for Internal Edges
# =============================================================================

class InteractionNetwork(MessagePassing):
    """
    Core GNN layer for message passing.
    Updates both node and edge features using message passing and residuals.
    """

    def __init__(self, latent_dim: int, nmlp_layers: int, mlp_hidden_dim: int):
        super(InteractionNetwork, self).__init__(aggr='add')

        # MLP for computing messages from [x_i, x_j, edge_features]
        self.edge_fn = nn.Sequential(
            build_mlp(latent_dim * 2 + latent_dim, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        # MLP to update node features from [aggr_messages, x_i]
        self.node_fn = nn.Sequential(
            build_mlp(latent_dim + latent_dim, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        # MLP to update edge features from [x_i_updated, x_j_updated, edge_features]
        self.edge_update_fn = nn.Sequential(
            build_mlp(latent_dim * 2 + latent_dim, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor):
        """
        Processes internal edges (comp↔comp).

        Note: For internal edges, it's acceptable to update all nodes since internal edges
        can connect to any computational nodes. However, we still identify the subgraph
        for consistency and to avoid updating nodes not in the subgraph.
        """
        edge_residual = edge_features.clone()

        # --- Identify nodes in internal subgraph (INTERIOR ONLY) ---
        # For internal edges (comp↔comp), both source and target are computational nodes
        # Both can be updated since internal edges are bidirectional
        node_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        if edge_index.size(1) > 0:
            src, tgt = edge_index[0], edge_index[1]
            # Direct boolean indexing (efficient) - mark both src and tgt
            node_mask[src] = True
            node_mask[tgt] = True

        # --- Node Message Passing ---
        if edge_index.size(1) > 0:
            aggr_messages = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)
        else:
            aggr_messages = torch.zeros_like(x)

        # Update only nodes in internal subgraph
        x_updated = x.clone()  # Start with copy (unchanged for nodes not in mask)
        if node_mask.any():
            x_updated[node_mask] = self.node_fn(torch.cat([x[node_mask], aggr_messages[node_mask]], dim=-1))

        # --- Edge Update ---
        if edge_index.size(1) > 0:
            src, tgt = edge_index[0], edge_index[1]
            edge_input = torch.cat([x_updated[src], x_updated[tgt], edge_features], dim=-1)
            edge_updated = self.edge_update_fn(edge_input)
        else:
            edge_updated = edge_features.clone()

        # Apply residual connections ONLY to masked nodes
        # Apply residual updates only to receiver nodes; other nodes remain unchanged.
        # For nodes not in mask: x_out = x (unchanged)
        # For nodes in mask: x_out = x_updated + x (residual connection)
        x_out = x.clone()  # Start with original (unchanged for nodes not in mask)
        if node_mask.any():
            x_out[node_mask] = x_updated[node_mask] + x[node_mask]  # Residual only on masked nodes
        edge_out = edge_updated + edge_residual

        return x_out, edge_out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        m = torch.cat([x_i, x_j, edge_features], dim=-1)
        return self.edge_fn(m)


class InteriorProcessor(nn.Module):
    """
    INTERIOR ONLY: Processes message passing ONLY on internal edges (comp↔comp).

    This component handles spatial interactions between computational nodes,
    excluding boundary-interior coupling which is handled by CouplingBlock.
    """

    def __init__(self, latent_dim: int, nmessage_passing_steps: int, nmlp_layers: int, mlp_hidden_dim: int):
        super(InteriorProcessor, self).__init__()
        self.message_passing_steps = nmessage_passing_steps
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                latent_dim=latent_dim,
                nmlp_layers=nmlp_layers,
                mlp_hidden_dim=mlp_hidden_dim
            ) for _ in range(nmessage_passing_steps)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        rainfall_conditioner: Optional[RainfallConditioning] = None,
        rainfall_context: Optional[Dict[str, torch.Tensor]] = None,
        rainfall_inject_every_step: bool = False,
    ):
        """
        Processes ONLY internal edges (comp↔comp).

        Args:
            x: Node latents [num_nodes, latent_dim]
            edge_index: Edge connectivity [2, num_internal_edges] - ONLY internal edges
            edge_features: Edge latents [num_internal_edges, latent_dim] - ONLY internal edges
            rainfall_conditioner: Optional computational-node rainfall FiLM module.
            rainfall_context: Rainfall context built once from raw node features.
            rainfall_inject_every_step: If true, reinject rainfall before every
                interior InteractionNetwork layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Updated node latents [num_nodes, latent_dim]
                - Updated edge latents [num_internal_edges, latent_dim]
        """
        for gnn in self.gnn_stacks:
            # Rainfall reinjection differs from boundary conditioning: it is
            # applied only to computational nodes and only inside the interior
            # processor, where rainfall-driven inland flooding needs a stronger
            # forcing signal.
            if (
                rainfall_inject_every_step
                and rainfall_conditioner is not None
                and rainfall_context is not None
            ):
                x = rainfall_conditioner.apply_rainfall_conditioning(x, rainfall_context)
            x, edge_features = gnn(x, edge_index, edge_features)
        return x, edge_features


# =============================================================================
# 3. COUPLING ONLY: Explicit Boundary-Interior Coupling Block
# =============================================================================

class CouplingInteraction(MessagePassing):
    """
    COUPLING ONLY: Message passing layer for boundary edges with learnable gate.

    This layer processes boundary edges (comp↔bghost) with a gate mechanism
    that can be set to learned, always-on (ones), or always-off (zeros) for ablation.
    """

    def __init__(self, latent_dim: int, nmlp_layers: int, mlp_hidden_dim: int,
                 gate_mode: str = "learned"):
        super(CouplingInteraction, self).__init__(aggr='add')
        self.gate_mode = gate_mode

        # MLP for computing messages from [x_i, x_j, edge_features]
        self.edge_fn = nn.Sequential(
            build_mlp(latent_dim * 2 + latent_dim, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # MLP to update node features from [aggr_messages, x_i]
        self.node_fn = nn.Sequential(
            build_mlp(latent_dim + latent_dim, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # MLP to update edge features from [x_i_updated, x_j_updated, edge_features]
        self.edge_update_fn = nn.Sequential(
            build_mlp(latent_dim * 2 + latent_dim, [mlp_hidden_dim] * nmlp_layers, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Gate MLP: computes gate value per boundary edge
        # Input: [x_src, x_tgt, edge_features] -> Output: scalar gate value
        if gate_mode == "learned":
            self.gate_mlp = nn.Sequential(
                build_mlp(latent_dim * 2 + latent_dim, [mlp_hidden_dim] * nmlp_layers, 1),
                nn.Sigmoid()  # Gate value in [0, 1]
            )
        else:
            self.gate_mlp = None  # Not used for "ones" or "zeros"

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_type: torch.Tensor,
        boundary_conditioner: Optional[BoundaryConditioning] = None,
        boundary_context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Processes boundary edges with gated messages.

        Only updates nodes connected to boundary edges.
        Nodes not in the boundary subgraph remain unchanged to maintain "COUPLING ONLY" semantics.

        Args:
            x: Node latents [num_nodes, latent_dim]
            edge_index: Edge connectivity [2, num_boundary_edges] - ONLY boundary edges
            edge_features: Edge latents [num_boundary_edges, latent_dim] - ONLY boundary edges
            node_type: Node type tensor [num_nodes] (0=comp, 1=bghost, 2=non_bc_ghost)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Updated node latents [num_nodes, latent_dim]
                - Updated edge latents [num_boundary_edges, latent_dim]
        """
        edge_residual = edge_features.clone()

        # --- Identify TARGET nodes only (COUPLING ONLY - receiver nodes) ---
        # Only receiver nodes are updated; sender nodes remain unchanged.
        # This ensures clean directional semantics: B→I updates only comp nodes, I→B updates only boundary ghosts
        # This also makes gate_mode="zeros" a true interaction-disabled ablation
        node_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        if edge_index.size(1) > 0:
            tgt = edge_index[1]  # Only target nodes (receivers)
            node_mask[tgt] = True  # Direct boolean indexing (efficient)

        # --- Compute gate values ---
        # ABLATION SWITCH: Gate mode determines coupling strength
        if self.gate_mode == "learned":
            # Compute gate from node and edge features
            if edge_index.size(1) > 0:
                src, tgt = edge_index[0], edge_index[1]
                gate_input = torch.cat([x[src], x[tgt], edge_features], dim=-1)
                gates = self.gate_mlp(gate_input).squeeze(-1)  # [num_boundary_edges]
            else:
                gates = torch.zeros(0, device=x.device, dtype=x.dtype)
        elif self.gate_mode == "ones":
            # Coupling always on (no ablation)
            gates = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype) if edge_index.size(1) > 0 else torch.zeros(0, device=x.device, dtype=x.dtype)
        elif self.gate_mode == "zeros":
            # ABLATION: Coupling disabled (interaction-disabled ablation)
            gates = torch.zeros(edge_index.size(1), device=x.device, dtype=x.dtype) if edge_index.size(1) > 0 else torch.zeros(0, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        # --- Node Message Passing with Gated Messages ---
        if edge_index.size(1) > 0:
            aggr_messages = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features, gates=gates)
        else:
            # No boundary edges - zero messages for all nodes
            aggr_messages = torch.zeros_like(x)

        # Only receiver nodes in the boundary subgraph are updated.
        # Nodes not connected to boundary edges remain unchanged
        x_updated = x.clone()  # Start with copy (unchanged for nodes not in mask)
        if node_mask.any():
            # Only apply node_fn to receiver nodes (targets)
            x_updated[node_mask] = self.node_fn(torch.cat([x[node_mask], aggr_messages[node_mask]], dim=-1))

        # --- Edge Update (only for boundary edges) ---
        if edge_index.size(1) > 0:
            src, tgt = edge_index[0], edge_index[1]
            edge_input = torch.cat([x_updated[src], x_updated[tgt], edge_features], dim=-1)
            edge_updated = self.edge_update_fn(edge_input)
        else:
            edge_updated = edge_features.clone()

        # Apply residual connections ONLY to masked nodes
        # Apply residual updates only to receiver nodes; other nodes remain unchanged.
        # For nodes not in mask: x_out = x (unchanged)
        # For nodes in mask: x_out = x_updated + x (residual connection)
        x_out = x.clone()  # Start with original (unchanged for nodes not in mask)
        if node_mask.any():
            x_out[node_mask] = x_updated[node_mask] + x[node_mask]  # Residual only on masked nodes
        edge_out = edge_updated + edge_residual

        return x_out, edge_out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_features: torch.Tensor,
                gates: torch.Tensor) -> torch.Tensor:
        """
        Computes gated messages for boundary edges.

        Args:
            x_i: Source node latents [num_boundary_edges, latent_dim]
            x_j: Target node latents [num_boundary_edges, latent_dim]
            edge_features: Edge latents [num_boundary_edges, latent_dim]
            gates: Gate values [num_boundary_edges] in [0, 1]

        Returns:
            Gated messages [num_boundary_edges, latent_dim]
        """
        m = torch.cat([x_i, x_j, edge_features], dim=-1)
        message = self.edge_fn(m)
        # Apply gate: if gate=0, message is zero (ablation)
        gated_message = gates.unsqueeze(-1) * message
        return gated_message


class CouplingBlock(nn.Module):
    """
    COUPLING ONLY: Explicit boundary-interior coupling with bidirectional passes.

    This block processes boundary edges (comp↔bghost) in two EXPLICIT directional passes:
    1. B→I: Messages from boundary ghost nodes to computational nodes only
    2. I→B: Messages from computational nodes to boundary ghost nodes only

    The gate mechanism allows ablation by setting gate=0 (interaction-disabled).
    """

    def __init__(self, latent_dim: int, nmessage_passing_steps: int, nmlp_layers: int,
                 mlp_hidden_dim: int, gate_mode: str = "learned"):
        super(CouplingBlock, self).__init__()
        self.message_passing_steps = nmessage_passing_steps
        self.gate_mode = gate_mode

        # Stack of coupling interaction layers
        self.coupling_layers = nn.ModuleList([
            CouplingInteraction(
                latent_dim=latent_dim,
                nmlp_layers=nmlp_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                gate_mode=gate_mode
            ) for _ in range(nmessage_passing_steps)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_type: torch.Tensor,
        boundary_conditioner: Optional[BoundaryConditioning] = None,
        boundary_context: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Processes boundary edges with EXPLICIT bidirectional coupling passes.

        This implementation processes boundary edges in two explicit passes:
        1. B→I: Messages from boundary ghost nodes to computational nodes
        2. I→B: Messages from computational nodes to boundary ghost nodes

        Args:
            x: Node latents [num_nodes, latent_dim]
            edge_index: Edge connectivity [2, num_boundary_edges] - ONLY boundary edges
            edge_features: Edge latents [num_boundary_edges, latent_dim] - ONLY boundary edges
            node_type: Node type tensor [num_nodes] (0=comp, 1=bghost, 2=non_bc_ghost)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Updated node latents [num_nodes, latent_dim]
                - Updated edge latents [num_boundary_edges, latent_dim]
        """
        # Process through multiple coupling layers
        for coupling_layer in self.coupling_layers:
            # Reinject boundary context before each coupling iteration when enabled.
            if boundary_conditioner is not None and boundary_context is not None:
                x = boundary_conditioner.apply_bc_conditioning(x, boundary_context)

            # Create directional masks for explicit B→I and I→B passes
            src, tgt = edge_index[0], edge_index[1]

            # B→I: boundary ghost (src) -> computational (tgt)
            mask_B2I = (node_type[src] == 1) & (node_type[tgt] == 0)

            # I→B: computational (src) -> boundary ghost (tgt)
            mask_I2B = (node_type[src] == 0) & (node_type[tgt] == 1)

            # Initialize updated edge features
            edge_features_updated = edge_features.clone()

            # Process B→I direction (boundary -> interior)
            if mask_B2I.any():
                edge_index_B2I = edge_index[:, mask_B2I]
                edge_features_B2I = edge_features[mask_B2I]
                x, edge_features_B2I_new = coupling_layer(x, edge_index_B2I, edge_features_B2I, node_type)
                # Store updated edge features
                edge_features_updated[mask_B2I] = edge_features_B2I_new

            # Process I→B direction (interior -> boundary)
            # Note: x has been updated by B→I pass, allowing bidirectional influence
            if mask_I2B.any():
                edge_index_I2B = edge_index[:, mask_I2B]
                edge_features_I2B = edge_features_updated[mask_I2B]  # Use updated features
                x, edge_features_I2B_new = coupling_layer(x, edge_index_I2B, edge_features_I2B, node_type)
                # Store updated edge features
                edge_features_updated[mask_I2B] = edge_features_I2B_new

            # Update edge_features for next iteration
            edge_features = edge_features_updated

        return x, edge_features


# =============================================================================
# 4. Helper: Scatter Edge Latents Back
# =============================================================================

def scatter_edge_latents_back(edge_latent_full: torch.Tensor,
                              edge_latent_internal: torch.Tensor,
                              edge_latent_boundary: torch.Tensor,
                              mask_internal: torch.Tensor,
                              mask_boundary: torch.Tensor) -> torch.Tensor:
    """
    Scatters updated internal and boundary edge latents back into full edge_latent tensor.

    Args:
        edge_latent_full: Full edge latent tensor [num_edges, latent_dim]
        edge_latent_internal: Updated internal edge latents [num_internal_edges, latent_dim]
        edge_latent_boundary: Updated boundary edge latents [num_boundary_edges, latent_dim]
        mask_internal: Boolean mask for internal edges [num_edges]
        mask_boundary: Boolean mask for boundary edges [num_edges]

    Returns:
        Updated full edge latent tensor [num_edges, latent_dim]
    """
    edge_latent_full = edge_latent_full.clone()  # Avoid in-place ops
    if mask_internal.any():
        edge_latent_full[mask_internal] = edge_latent_internal
    if mask_boundary.any():
        edge_latent_full[mask_boundary] = edge_latent_boundary
    return edge_latent_full


# =============================================================================
# 5. Decoder
# =============================================================================

class MainDecoder(nn.Module):
    """
    State-aware decoder.

    The decoder receives both the graph latent and the
    explicit current state extracted from the input history.
    """

    def __init__(self, decoder_in_dim: int, n_node_out: int, nmlp_layers: int, mlp_hidden_dim: int):
        super(MainDecoder, self).__init__()
        self.node_fn = build_mlp(
            decoder_in_dim,
            [mlp_hidden_dim for _ in range(nmlp_layers)],
            n_node_out,
        )

    def forward(self, x: torch.Tensor):
        return self.node_fn(x)


# =============================================================================
# 6. Main GNN Model with Explicit Coupling, Input-to-Output Residuals, and Boundary Skip Connections
# =============================================================================

class GNNModel(nn.Module):
    """
    Encode-process-decode GNN4CF model.

    Architecture:
    1. Type-specific encoders, coupling, interior processing, and boundary skip
    2. State-aware decoder input [node_latent, node_residual]
    3. Optional boundary conditioning before each coupling iteration
    4. Optional rainfall FiLM conditioning on computational nodes before/during
       the interior processor
    """

    def __init__(self,
                 # --- Flexible I/O Definition (from config) ---
                 predictor_step: int,
                 n_static_node_comp: int,
                 n_static_node_bghost: int,
                 n_static_edge_internal: int,
                 n_static_edge_boundary: int,
                 n_dynamic_node_vars: int,
                 n_dynamic_edge_vars: int,
                 n_label_vars: int,
                 # --------------------------------------------

                # Model Hyperparameters
                latent_dim: int,
                nmessage_passing_steps: int,  # Default/fallback value for processors
                nmlp_layers: int,             # Default for Encoder, Decoder, and processors
                mlp_hidden_dim: int,          # Default for Encoder, Decoder, and processors

                # InteriorProcessor-specific overrides (optional)
                nmessage_passing_steps_interior: Optional[int] = None,
                nmlp_layers_interior: Optional[int] = None,
                mlp_hidden_dim_interior: Optional[int] = None,

                # CouplingBlock-specific overrides (optional)
                nmessage_passing_steps_coupling: Optional[int] = None,
                nmlp_layers_coupling: Optional[int] = None,
                mlp_hidden_dim_coupling: Optional[int] = None,

                # Coupling control settings
                enable_coupling: bool = True,                            # Can disable entire coupling block
                coupling_gate_mode: str = "learned",                    # "learned", "ones", or "zeros"

                # Residual prediction settings
                residual: bool = True,                                    # Enable input-to-output residual connections

                # Boundary skip connection settings
                boundary_skip: bool = True,                               # Enable boundary skip connections
                boundary_preserve_weight: Optional[float] = None,        # Fixed weight (0.0-1.0), None = learnable

                # Boundary conditioning settings
                boundary_conditioning_enabled: bool = False,
                boundary_conditioning_mode: str = "concat",
                inject_every_processor_step: bool = True,
                use_current_state: bool = True,
                use_history_state: bool = True,
                use_geometry: bool = True,
                geometry_interaction: str = "multiply",
                current_encoder_hidden_dim: int = 64,
                history_encoder_hidden_dim: int = 64,
                geometry_encoder_hidden_dim: int = 64,
                update_mlp_hidden_dim: int = 128,
                film_hidden_dim: int = 128,

                # Rainfall conditioning settings
                rainfall_conditioning_enabled: bool = False,
                rainfall_conditioning_mode: str = "film",
                rainfall_inject_before_interior: bool = True,
                rainfall_inject_every_interior_step: bool = True,
                rainfall_features: Optional[List[str]] = None,
                rainfall_current_encoder_hidden_dim: int = 64,
                rainfall_film_hidden_dim: int = 64,
                dynamic_input_feature_names: Optional[List[str]] = None,
                ):
        super(GNNModel, self).__init__()

        # --- ABLATION SWITCH: Enable/disable coupling ---
        self.enable_coupling = enable_coupling

        # Enable or disable input-to-output residual prediction.
        self.residual = residual

        # Enable or disable boundary skip connections.
        self.boundary_skip = boundary_skip

        # --- 1. Calculate input and output dimensions ---
        model_history_steps = max(2, predictor_step)
        self.model_history_steps = model_history_steps
        self.predictor_step = predictor_step
        self.n_label_vars = n_label_vars
        self.latent_dim = latent_dim
        self.n_static_node_comp = n_static_node_comp
        self.n_static_node_bghost = n_static_node_bghost
        self.n_dynamic_node_vars = n_dynamic_node_vars

        # Input features = static + (dynamic_vars * model_history_steps)
        n_node_comp_in = n_static_node_comp + (n_dynamic_node_vars * model_history_steps)
        n_node_bghost_in = n_static_node_bghost + (n_dynamic_node_vars * model_history_steps)

        n_edge_internal_in = n_static_edge_internal + (n_dynamic_edge_vars * model_history_steps)
        n_edge_boundary_in = n_static_edge_boundary + (n_dynamic_edge_vars * model_history_steps)

        # Output features = label_vars * predictor_steps
        n_node_out = n_label_vars * predictor_step

        # --- 2. Resolve component-specific hyperparameters ---
        # InteriorProcessor: use component-specific or fallback to shared
        n_interior_steps, n_interior_layers, n_interior_hidden = _resolve_component_hyperparams(
            nmessage_passing_steps, nmlp_layers, mlp_hidden_dim,
            nmessage_passing_steps_interior, nmlp_layers_interior, mlp_hidden_dim_interior
        )

        # CouplingBlock: use component-specific or fallback to shared
        n_coupling_steps, n_coupling_layers, n_coupling_hidden = _resolve_component_hyperparams(
            nmessage_passing_steps, nmlp_layers, mlp_hidden_dim,
            nmessage_passing_steps_coupling, nmlp_layers_coupling, mlp_hidden_dim_coupling
        )

        # --- 3. Instantiate Encoder (uses shared parameters) ---
        self.encoder = Encoder(
            n_node_comp_in=n_node_comp_in,
            n_node_bghost_in=n_node_bghost_in,
            n_edge_internal_in=n_edge_internal_in,
            n_edge_boundary_in=n_edge_boundary_in,
            latent_dim=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim
        )

        # --- 4. Instantiate InteriorProcessor (INTERIOR ONLY) ---
        # Uses component-specific parameters (resolved above)
        self.interior_processor = InteriorProcessor(
            latent_dim=latent_dim,  # Shared
            nmessage_passing_steps=n_interior_steps,
            nmlp_layers=n_interior_layers,
            mlp_hidden_dim=n_interior_hidden
        )

        # --- 5. Instantiate CouplingBlock (COUPLING ONLY) ---
        # Uses component-specific parameters (resolved above)
        if self.enable_coupling:
            self.coupling_block = CouplingBlock(
                latent_dim=latent_dim,  # Shared
                nmessage_passing_steps=n_coupling_steps,
                nmlp_layers=n_coupling_layers,
                mlp_hidden_dim=n_coupling_hidden,
                gate_mode=coupling_gate_mode
            )
        else:
            self.coupling_block = None

        # --- 6. Instantiate the state-aware decoder ---
        decoder_in_dim = latent_dim + n_label_vars
        self.decoder = MainDecoder(
            decoder_in_dim=decoder_in_dim,
            n_node_out=n_node_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim
        )

        # --- 7. Boundary skip connection with learned or fixed weight ---
        if self.boundary_skip:
            if boundary_preserve_weight is not None:
                # Fixed weight (not learnable)
                self.boundary_preserve_weight = boundary_preserve_weight
                self.learnable_boundary_weight = False
            else:
                # Learnable weight (initialized to 0.3)
                self.boundary_preserve_weight_param = nn.Parameter(torch.tensor(0.3))
                self.learnable_boundary_weight = True
        else:
            self.boundary_preserve_weight = None
            self.learnable_boundary_weight = False

        # --- 8. Boundary conditioning module ---
        self.boundary_conditioner = BoundaryConditioning(
            latent_dim=latent_dim,
            model_history_steps=model_history_steps,
            n_dynamic_node_vars=n_dynamic_node_vars,
            n_static_node_bghost=n_static_node_bghost,
            enabled=boundary_conditioning_enabled,
            mode=boundary_conditioning_mode,
            inject_every_processor_step=inject_every_processor_step,
            use_current_state=use_current_state,
            use_history_state=use_history_state,
            use_geometry=use_geometry,
            geometry_interaction=geometry_interaction,
            current_encoder_hidden_dim=current_encoder_hidden_dim,
            history_encoder_hidden_dim=history_encoder_hidden_dim,
            geometry_encoder_hidden_dim=geometry_encoder_hidden_dim,
            update_mlp_hidden_dim=update_mlp_hidden_dim,
            film_hidden_dim=film_hidden_dim,
        )

        # --- 9. Rainfall conditioning module ---
        # Rainfall conditioning is active only when enabled at construction.
        # When enabled, dynamic feature names are required so rainfall features
        # are resolved robustly instead of by brittle positional assumptions.
        # Static features are intentionally not reinjected here; the node
        # encoder already sees them in the initial node features.
        self.rainfall_conditioner = RainfallConditioning(
            latent_dim=latent_dim,
            model_history_steps=model_history_steps,
            n_dynamic_node_vars=n_dynamic_node_vars,
            n_static_node_comp=n_static_node_comp,
            enabled=rainfall_conditioning_enabled,
            mode=rainfall_conditioning_mode,
            inject_before_interior=rainfall_inject_before_interior,
            inject_every_interior_step=rainfall_inject_every_interior_step,
            rainfall_features=rainfall_features,
            current_encoder_hidden_dim=rainfall_current_encoder_hidden_dim,
            film_hidden_dim=rainfall_film_hidden_dim,
            dynamic_input_feature_names=dynamic_input_feature_names,
        )

        # --- 10. Initialize weights ---
        self.apply(init_weights)

    def _compute_node_residuals(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """
        Computes residual values from input features for all node types.

        Extracts the last known values of label variables from the input history.
        The input features are structured as STEP-MAJOR INTERLEAVED:
        [static_features, var1_t, var2_t, ..., varN_t, var1_t+1, var2_t+1, ..., varN_t+1, ...]

        For each label variable i (where i=0 to n_label_vars-1), the last value is at:
        n_static_node_comp + (model_history_steps - 1) * n_dynamic_node_vars + i

        This assumes label_vars are the first n_label_vars variables in input_vars
        (which is true when label_vars = dynamic_input_state_variables and
        input_vars = state_vars + driver_vars).

        Args:
            x: Raw node features [num_nodes, n_node_features]
            node_type: Node type tensor [num_nodes] (0=comp, 1=bghost, 2=non_bc_ghost)

        Returns:
            Residual tensor [num_nodes, n_label_vars]
        """
        num_nodes = x.size(0)
        node_residual = torch.zeros(num_nodes, self.n_label_vars, device=x.device, dtype=x.dtype)

        # Extract residuals for computational nodes (type 0)
        mask_comp = (node_type == 0)
        if mask_comp.any():
            x_comp = x[mask_comp]  # [num_comp_nodes, n_node_features]
            for label_idx in range(self.n_label_vars):
                # Step-major interleaved layout:
                # Last timestep offset: (model_history_steps - 1) * n_dynamic_node_vars
                # Variable position within timestep: label_idx (label_vars are first in input_vars)
                residual_idx = self.n_static_node_comp + (self.model_history_steps - 1) * self.n_dynamic_node_vars + label_idx
                node_residual[mask_comp, label_idx] = x_comp[:, residual_idx]

        # Note: Boundary ghost nodes (type 1) and non_bc_ghost nodes (type 2) remain zeros.
        # Bghost nodes don't have label variables (wd, vx, vy) in their input features,
        # so residuals cannot be extracted. They are used only for message passing.

        return node_residual

    def _build_decoder_input(
        self,
        node_latent: torch.Tensor,
        node_residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build the decoder input for the state-aware decoder.
        """
        if node_residual is None:
            node_residual = torch.zeros(
                node_latent.size(0),
                self.n_label_vars,
                device=node_latent.device,
                dtype=node_latent.dtype,
            )
        return torch.cat([node_latent, node_residual], dim=-1)

    def _identify_boundary_affected_nodes(self, edge_index_boundary: torch.Tensor,
                                         node_type: torch.Tensor) -> torch.Tensor:
        """
        Identifies computational nodes that receive boundary information (targets of B→I edges).

        Args:
            edge_index_boundary: Boundary edge connectivity [2, num_boundary_edges]
            node_type: Node type tensor [num_nodes] (0=comp, 1=bghost, 2=non_bc_ghost)

        Returns:
            Boolean mask [num_nodes] - True for nodes that received boundary updates
        """
        boundary_affected_mask = torch.zeros(node_type.size(0), dtype=torch.bool, device=node_type.device)

        if edge_index_boundary.size(1) > 0:
            src, tgt = edge_index_boundary[0], edge_index_boundary[1]
            # B→I edges: boundary ghost (src) -> computational (tgt)
            mask_B2I = (node_type[src] == 1) & (node_type[tgt] == 0)
            # Mark target nodes (computational nodes that received boundary info)
            boundary_affected_mask[tgt[mask_B2I]] = True

        return boundary_affected_mask

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                node_type: torch.Tensor,
                edge_type: torch.Tensor):
        """
        Run encoding, coupling, interior propagation, and decoding.

        Args:
            x: Raw node features (sized by model_history_steps).
            edge_index: Graph connectivity [2, num_edges].
            edge_attr: Raw edge features (sized by model_history_steps).
            node_type: Tensor identifying node types [num_nodes] (0=comp, 1=bghost, 2=non_bc_ghost).
            edge_type: Tensor identifying edge types [num_edges] (0=internal, 1=boundary).

        Returns:
            torch.Tensor: The predicted node labels (y_pred),
                          flattened for `predictor_step` steps.
                          Shape: [num_nodes, n_label_vars * predictor_step]
        """

        # === Extract residual state for stability ===
        if self.residual:
            # Extract last known values of label variables from input
            # Shape: [num_nodes, n_label_vars]
            node_residual = self._compute_node_residuals(x, node_type)
        else:
            node_residual = None

        # 1. Encode all graph components into a common latent space
        node_latent, edge_latent = self.encoder(x, edge_attr, node_type, edge_type)

        # 1b. Precompute optional boundary-conditioning context from raw boundary-node features.
        boundary_context = self.boundary_conditioner.build_context(x, node_type)

        # 1c. Precompute optional rainfall-conditioning context from raw
        # computational-node features. This context is built once and reused
        # for either one-time or per-layer FiLM reinjection during interior
        # processing.
        rainfall_context = self.rainfall_conditioner.build_context(x, node_type)

        # 2. Split edges into internal and boundary
        mask_internal = (edge_type == 0)  # INTERIOR ONLY
        mask_boundary = (edge_type == 1)   # COUPLING ONLY

        edge_index_internal = edge_index[:, mask_internal]
        edge_latent_internal = edge_latent[mask_internal]

        edge_index_boundary = edge_index[:, mask_boundary]
        edge_latent_boundary = edge_latent[mask_boundary]

        # 3. Process boundary edges with explicit coupling FIRST (COUPLING ONLY)
        # Coupling precedes interior propagation so boundary information can move inland.
        # Boundary information is injected early, then interior processing propagates it through
        # multiple hops to inland computational nodes via interior edges.

        # === Identify boundary-affected nodes before coupling ===
        boundary_affected_mask = None
        if self.boundary_skip and self.enable_coupling and mask_boundary.any():
            boundary_affected_mask = self._identify_boundary_affected_nodes(edge_index_boundary, node_type)

        if self.enable_coupling and mask_boundary.any():
            if (
                self.boundary_conditioner.enabled
                and not self.boundary_conditioner.inject_every_processor_step
                and boundary_context is not None
            ):
                node_latent = self.boundary_conditioner.apply_bc_conditioning(node_latent, boundary_context)

            node_latent, edge_latent_boundary = self.coupling_block(
                node_latent,
                edge_index_boundary,
                edge_latent_boundary,
                node_type,
                boundary_conditioner=(
                    self.boundary_conditioner
                    if self.boundary_conditioner.enabled
                    and self.boundary_conditioner.inject_every_processor_step
                    else None
                ),
                boundary_context=boundary_context,
            )

            # === Store boundary-affected node latents after coupling ===
            if self.boundary_skip and boundary_affected_mask is not None:
                boundary_affected_latents = node_latent.clone()  # Store latents after coupling
            else:
                boundary_affected_latents = None
        else:
            boundary_affected_latents = None

        # 4. Process internal edges (INTERIOR ONLY)
        # After coupling, interior processing propagates boundary information inland through
        # multi-hop message passing on interior edges (comp↔comp).
        if (
            self.rainfall_conditioner.enabled
            and self.rainfall_conditioner.inject_before_interior
            and not self.rainfall_conditioner.inject_every_interior_step
            and rainfall_context is not None
        ):
            node_latent = self.rainfall_conditioner.apply_rainfall_conditioning(
                node_latent,
                rainfall_context,
            )

        node_latent, edge_latent_internal = self.interior_processor(
            node_latent,
            edge_index_internal,
            edge_latent_internal,
            rainfall_conditioner=(
                self.rainfall_conditioner
                if self.rainfall_conditioner.enabled
                and self.rainfall_conditioner.inject_before_interior
                and self.rainfall_conditioner.inject_every_interior_step
                else None
            ),
            rainfall_context=rainfall_context,
            rainfall_inject_every_step=(
                self.rainfall_conditioner.enabled
                and self.rainfall_conditioner.inject_before_interior
                and self.rainfall_conditioner.inject_every_interior_step
            ),
        )

        # === Apply boundary skip connection ===
        # Mix preserved boundary latents back into interior-processed latents
        if self.boundary_skip and boundary_affected_latents is not None and boundary_affected_mask is not None:
            # Get mixing weight (learnable or fixed)
            if self.learnable_boundary_weight:
                # Clamp learnable weight to [0, 1] using sigmoid
                alpha = torch.sigmoid(self.boundary_preserve_weight_param)
            else:
                alpha = self.boundary_preserve_weight

            # Mix: alpha * boundary_latents + (1 - alpha) * interior_latents
            # Only apply to boundary-affected nodes
            node_latent[boundary_affected_mask] = (
                alpha * boundary_affected_latents[boundary_affected_mask] +
                (1 - alpha) * node_latent[boundary_affected_mask]
            )

        # 5. Scatter updated edge latents back into full tensor
        edge_latent = scatter_edge_latents_back(
            edge_latent, edge_latent_internal, edge_latent_boundary,
            mask_internal, mask_boundary
        )

        # 6. Decode final node predictions from [latent, explicit current state]
        decoder_input = self._build_decoder_input(node_latent, node_residual)
        y_pred = self.decoder(decoder_input)

        # === Apply residual prediction when enabled ===
        if self.residual and node_residual is not None:
            # Cascading residuals for multi-step prediction (matches paper's autoregressive update):
            # The decoder predicts incremental changes (deltas) relative to previous time step.
            # For multi-step predictions, residuals cascade:
            #   Step 1: ŷ(t+1) = delta_1 + y(t)           (delta_1 from decoder, y(t) from input)
            #   Step 2: ŷ(t+2) = delta_2 + ŷ(t+1)          (delta_2 from decoder, ŷ(t+1) from step 1)
            #   Step 3: ŷ(t+3) = delta_3 + ŷ(t+2)          (delta_3 from decoder, ŷ(t+2) from step 2)
            #   etc.
            #
            # This ensures each prediction builds on the previous one, matching the paper's approach:
            #   ŷ(t+1) = D(P(E(...))) + y(t)
            #
            # node_residual: [num_nodes, n_label_vars] = y(t) (last known value from input)
            # y_pred: [num_nodes, n_label_vars * predictor_step] (decoder deltas in step-major order)
            # Output format: [var1_t1, var2_t1, var3_t1, var1_t2, var2_t2, var3_t2, ...] (step-major)

            N = node_residual.shape[0]

            # Reshape decoder output to [N, predictor_step, n_label_vars] for step-wise processing
            # Decoder output is step-major: [var1_t1, var2_t1, var3_t1, var1_t2, var2_t2, var3_t2, ...]
            y_pred_reshaped = y_pred.reshape(N, self.predictor_step, self.n_label_vars)

            # Start with y(t) as the base for the first step
            current_value = node_residual  # [N, n_label_vars] = y(t)

            # Accumulate predictions step by step (cascading residuals)
            for step_idx in range(self.predictor_step):
                # Add residual (current_value) to decoder delta for this step
                # y_pred_reshaped[:, step_idx, :] contains the delta for step step_idx
                y_pred_reshaped[:, step_idx, :] = y_pred_reshaped[:, step_idx, :] + current_value

                # Update current_value for next step (use the prediction we just made)
                # This becomes the baseline for the next step
                if step_idx < self.predictor_step - 1:
                    current_value = y_pred_reshaped[:, step_idx, :]  # [N, n_label_vars]

            # Reshape back to [N, predictor_step * n_label_vars] (step-major flattened)
            y_pred = y_pred_reshaped.reshape(N, self.predictor_step * self.n_label_vars)

        return y_pred


# =============================================================================
# Model construction
# =============================================================================
#
# With rainfall_conditioning_enabled=False, rainfall conditioning is inactive
# and the dynamic feature-name list is not required.
#
# To enable rainfall FiLM, pass the rainfall_conditioning config block into the
# constructor and provide the dynamic feature-name list:
#
#   dynamic_input_feature_names =
#       config["features"]["dynamic_input_state_variables"] +
#       config["features"]["dynamic_input_drivers"]
#
#   model = GNNModel(...,
#       rainfall_conditioning_enabled=True,
#       rainfall_conditioning_mode="film",
#       rainfall_inject_before_interior=True,
#       rainfall_inject_every_interior_step=True,
#       rainfall_features=["rainfall_rate", "acc_rainfall"],
#       dynamic_input_feature_names=dynamic_input_feature_names,
#   )
#
