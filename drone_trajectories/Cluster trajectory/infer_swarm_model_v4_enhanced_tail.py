#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Enhanced Tail Dynamics Inference Script
==========================================

Enhanced version of v4 inference with multi-scale tail dynamics analysis.

Improvements over standard v4:
- Multi-scale tail dynamics analysis (short/medium/long windows)
- Acceleration change detection (tangential/normal acceleration trends)
- Curvature change rate analysis (second-order curvature derivative)
- Angular acceleration detection (angular velocity change rate)
- Intelligent behavior classification (accelerating/turning/decelerating/constant/sharp_turn)
- Adaptive prediction enhancement (adjusts based on behavior type)
- Supports v4's 32D features and GNN

Tail Enhancement Principle:
    Trajectory tail dynamics changes indicate future behavior:
    - Acceleration increase -> about to accelerate
    - Curvature increase + normal acceleration increase -> about to turn
    - Rapid velocity direction change -> sharp turn
    - Angular velocity increase -> turning rate increasing
    - Acceleration decrease -> about to decelerate

Architecture:
    Input positions -> [Multi-scale tail analysis] -> [Behavior classification]
    -> [Adaptive enhancement] -> Absolute position prediction
                                        |
                    (32D features + GNN) -> BiGRU -> Multi-branch decoder

Usage:
    python infer_swarm_model_v4_enhanced_tail.py ^
        --model gru_models_v4_fixed_agents_3_v4_fixed_gnn/best_model_agents_3_v4_fixed_gnn.pt ^
        --agents 3 --output_dir infer_results_v4_enhanced ^
        --features_dir features_32d --tail_window 8 --use_multi_scale
"""

from pathlib import Path
import sys
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Configure matplotlib for Chinese fonts (if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from train_swarm_model_v2_dynamics_aware import (
        DynamicsAwareSwarmGRUModel,
    )
    from train_swarm_model_v3_with_gnn import (
        DynamicsAwareSwarmGRUModel_with_GNN,
        build_adjacency_from_positions,
    )
except ImportError as e:
    logger.error("Failed to import required modules: %s", e)
    raise


# =====================================================================
# Motion Behavior Enum
# =====================================================================

class MotionBehavior(Enum):
    """Motion behavior types"""
    ACCELERATING = "accelerating"
    TURNING = "turning"
    DECELERATING = "decelerating"
    CONSTANT_SPEED = "constant_speed"
    SHARP_TURN = "sharp_turn"
    UNKNOWN = "unknown"


@dataclass
class TailDynamicsInfo:
    """Tail dynamics information"""
    # Basic features
    velocity_magnitude: np.ndarray  # (agents,) velocity magnitude
    velocity_direction: np.ndarray  # (agents, 3) velocity direction
    acceleration_tangent: np.ndarray  # (agents,) tangential acceleration
    acceleration_normal: np.ndarray  # (agents,) normal acceleration
    curvature: np.ndarray  # (agents,) curvature
    angular_velocity: np.ndarray  # (agents,) angular velocity
    
    # Change trends (relative to window start)
    velocity_change_rate: np.ndarray  # (agents,) velocity change rate
    acceleration_change_rate: np.ndarray  # (agents,) acceleration change rate
    curvature_change_rate: np.ndarray  # (agents,) curvature change rate
    angular_acceleration: np.ndarray  # (agents,) angular acceleration
    
    # Behavior classification
    behavior: np.ndarray  # (agents,) MotionBehavior enum values
    behavior_confidence: np.ndarray  # (agents,) behavior confidence [0, 1]
    
    # Enhancement parameters
    enhancement_factor: np.ndarray  # (agents,) enhancement factor
    enhancement_direction: np.ndarray  # (agents, 3) enhancement direction


# =====================================================================
# Enhanced Tail Dynamics Analyzer
# =====================================================================

class EnhancedTailDynamicsAnalyzer:
    """
    Enhanced tail dynamics analyzer
    
    Features:
    - Multi-scale window analysis (short/medium/long windows)
    - Precise acceleration and curvature change detection
    - Intelligent behavior classification
    - Adaptive enhancement parameter computation
    """
    
    def __init__(self, 
                 short_window=3, 
                 medium_window=5, 
                 long_window=8,
                 dt=0.1):
        """
        Args:
            short_window: Short window size (for rapid change detection)
            medium_window: Medium window size (for trend detection)
            long_window: Long window size (for overall trend)
            dt: Time step
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.dt = dt
        
        # Behavior classification thresholds
        self.acceleration_threshold = 0.5  # m/s^2
        self.curvature_threshold = 0.1  # 1/m
        self.angular_velocity_threshold = 0.5  # rad/s
        self.velocity_change_threshold = 0.3  # m/s
        
    def compute_tail_dynamics(self, trajectory: np.ndarray) -> TailDynamicsInfo:
        """
        Compute tail dynamics information
        
        Args:
            trajectory: (seq_in, agents, 3) input trajectory
        
        Returns:
            TailDynamicsInfo: Tail dynamics information
        """
        seq_len, num_agents, _ = trajectory.shape
        
        # Extract tail trajectories with different windows
        short_start = max(0, seq_len - self.short_window)
        medium_start = max(0, seq_len - self.medium_window)
        long_start = max(0, seq_len - self.long_window)
        
        short_traj = trajectory[short_start:, :, :]  # (short_window, agents, 3)
        medium_traj = trajectory[medium_start:, :, :]  # (medium_window, agents, 3)
        long_traj = trajectory[long_start:, :, :]  # (long_window, agents, 3)
        
        # Compute velocities
        vel_short = np.diff(short_traj, axis=0) / self.dt  # (short_window-1, agents, 3)
        vel_medium = np.diff(medium_traj, axis=0) / self.dt  # (medium_window-1, agents, 3)
        vel_long = np.diff(long_traj, axis=0) / self.dt  # (long_window-1, agents, 3)
        
        # Compute accelerations
        acc_short = np.diff(vel_short, axis=0) / self.dt if len(vel_short) > 1 else np.zeros((1, num_agents, 3))
        acc_medium = np.diff(vel_medium, axis=0) / self.dt if len(vel_medium) > 1 else np.zeros((1, num_agents, 3))
        
        # Current velocity and acceleration (use last value from short window)
        if len(vel_short) > 0:
            current_vel = vel_short[-1]  # (agents, 3)
        else:
            current_vel = np.zeros((num_agents, 3))
            
        if len(acc_short) > 0:
            current_acc = acc_short[-1]  # (agents, 3)
        else:
            current_acc = np.zeros((num_agents, 3))
        
        # Velocity magnitude and direction
        vel_mag = np.linalg.norm(current_vel, axis=1)  # (agents,)
        vel_dir = current_vel / (vel_mag[:, np.newaxis] + 1e-8)  # (agents, 3)
        
        # Acceleration decomposition (tangential and normal)
        a_tangent, a_normal = self._decompose_acceleration(current_vel, current_acc)
        
        # Curvature computation
        curvature = self._compute_curvature(vel_medium, acc_medium)  # (agents,)
        
        # Angular velocity computation
        angular_velocity = self._compute_angular_velocity(vel_medium)  # (agents,)
        
        # Change trend computation
        velocity_change_rate = self._compute_velocity_change_rate(vel_long)  # (agents,)
        acceleration_change_rate = self._compute_acceleration_change_rate(acc_medium)  # (agents,)
        curvature_change_rate = self._compute_curvature_change_rate(medium_traj)  # (agents,)
        angular_acceleration = self._compute_angular_acceleration(vel_medium)  # (agents,)
        
        # Behavior classification
        behavior, confidence = self._classify_behavior(
            vel_mag, a_tangent, a_normal, curvature, angular_velocity,
            velocity_change_rate, acceleration_change_rate, curvature_change_rate, angular_acceleration
        )
        
        # Compute enhancement parameters
        enhancement_factor, enhancement_dir = self._compute_enhancement_params(
            behavior, confidence, vel_dir, a_tangent, a_normal, curvature, angular_velocity
        )
        
        return TailDynamicsInfo(
            velocity_magnitude=vel_mag,
            velocity_direction=vel_dir,
            acceleration_tangent=a_tangent,
            acceleration_normal=a_normal,
            curvature=curvature,
            angular_velocity=angular_velocity,
            velocity_change_rate=velocity_change_rate,
            acceleration_change_rate=acceleration_change_rate,
            curvature_change_rate=curvature_change_rate,
            angular_acceleration=angular_acceleration,
            behavior=behavior,
            behavior_confidence=confidence,
            enhancement_factor=enhancement_factor,
            enhancement_direction=enhancement_dir
        )
    
    def _decompose_acceleration(self, velocity: np.ndarray, acceleration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose acceleration into tangential and normal components"""
        num_agents = velocity.shape[0]
        a_tangent = np.zeros(num_agents)
        a_normal = np.zeros(num_agents)
        
        for i in range(num_agents):
            v = velocity[i]  # (3,)
            a = acceleration[i]  # (3,)
            
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-8:
                v_unit = v / v_norm
                a_tangent[i] = np.dot(a, v_unit)
                a_normal_vec = a - a_tangent[i] * v_unit
                a_normal[i] = np.linalg.norm(a_normal_vec)
            else:
                a_tangent[i] = np.linalg.norm(a)
                a_normal[i] = 0.0
        
        return a_tangent, a_normal
    
    def _compute_curvature(self, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
        """Compute curvature"""
        num_agents = velocity.shape[1] if velocity.ndim > 1 else 1
        curvatures = np.zeros(num_agents)
        
        if velocity.shape[0] < 2 or acceleration.shape[0] < 1:
            return curvatures
        
        # Use average curvature from last few time steps
        for i in range(num_agents):
            curv_list = []
            for t in range(min(len(velocity) - 1, len(acceleration))):
                v = velocity[t, i] if velocity.ndim > 1 else velocity[t]
                a = acceleration[t, i] if acceleration.ndim > 1 else acceleration[t]
                
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-8:
                    cross = np.cross(v, a)
                    curv = np.linalg.norm(cross) / (v_norm ** 3 + 1e-8)
                    curv_list.append(curv)
            
            if curv_list:
                curvatures[i] = np.mean(curv_list)
        
        return curvatures
    
    def _compute_angular_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Compute angular velocity"""
        num_agents = velocity.shape[1] if velocity.ndim > 1 else 1
        angular_velocities = np.zeros(num_agents)
        
        if velocity.shape[0] < 2:
            return angular_velocities
        
        for i in range(num_agents):
            vel_seq = velocity[:, i] if velocity.ndim > 1 else velocity
            vel_norms = np.linalg.norm(vel_seq, axis=1) if vel_seq.ndim > 1 else np.abs(vel_seq)
            
            # Compute velocity direction changes
            if vel_seq.ndim > 1 and len(vel_seq) >= 2:
                vel_dirs = vel_seq / (vel_norms[:, np.newaxis] + 1e-8)
                angles = []
                for t in range(len(vel_dirs) - 1):
                    dot = np.clip(np.dot(vel_dirs[t], vel_dirs[t+1]), -1.0, 1.0)
                    angle = np.arccos(dot)
                    angles.append(angle)
                
                if angles:
                    angular_velocities[i] = np.mean(angles) / self.dt
        
        return angular_velocities
    
    def _compute_velocity_change_rate(self, velocity: np.ndarray) -> np.ndarray:
        """Compute velocity change rate"""
        num_agents = velocity.shape[1] if velocity.ndim > 1 else 1
        change_rates = np.zeros(num_agents)
        
        if velocity.shape[0] < 2:
            return change_rates
        
        for i in range(num_agents):
            vel_seq = velocity[:, i] if velocity.ndim > 1 else velocity
            vel_mags = np.linalg.norm(vel_seq, axis=1) if vel_seq.ndim > 1 else np.abs(vel_seq)
            
            if len(vel_mags) >= 2:
                # Compute velocity magnitude change rate
                vel_start = vel_mags[0]
                vel_end = vel_mags[-1]
                change_rates[i] = (vel_end - vel_start) / (len(vel_mags) * self.dt + 1e-8)
        
        return change_rates
    
    def _compute_acceleration_change_rate(self, acceleration: np.ndarray) -> np.ndarray:
        """Compute acceleration change rate"""
        num_agents = acceleration.shape[1] if acceleration.ndim > 1 else 1
        change_rates = np.zeros(num_agents)
        
        if acceleration.shape[0] < 2:
            return change_rates
        
        for i in range(num_agents):
            acc_seq = acceleration[:, i] if acceleration.ndim > 1 else acceleration
            acc_mags = np.linalg.norm(acc_seq, axis=1) if acc_seq.ndim > 1 else np.abs(acc_seq)
            
            if len(acc_mags) >= 2:
                acc_start = acc_mags[0]
                acc_end = acc_mags[-1]
                change_rates[i] = (acc_end - acc_start) / (len(acc_mags) * self.dt + 1e-8)
        
        return change_rates
    
    def _compute_curvature_change_rate(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute curvature change rate"""
        num_agents = trajectory.shape[1]
        change_rates = np.zeros(num_agents)
        
        if trajectory.shape[0] < 3:
            return change_rates
        
        # Compute curvature at each time step
        curvatures = []
        vel = np.diff(trajectory, axis=0) / self.dt
        acc = np.diff(vel, axis=0) / self.dt
        
        for t in range(min(len(vel) - 1, len(acc))):
            curv_t = []
            for i in range(num_agents):
                v = vel[t, i]
                a = acc[t, i] if t < len(acc) else np.zeros(3)
                
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-8:
                    cross = np.cross(v, a)
                    curv = np.linalg.norm(cross) / (v_norm ** 3 + 1e-8)
                    curv_t.append(curv)
                else:
                    curv_t.append(0.0)
            curvatures.append(curv_t)
        
        if len(curvatures) >= 2:
            curvatures = np.array(curvatures)  # (time_steps, agents)
            for i in range(num_agents):
                curv_start = curvatures[0, i]
                curv_end = curvatures[-1, i]
                change_rates[i] = (curv_end - curv_start) / (len(curvatures) * self.dt + 1e-8)
        
        return change_rates
    
    def _compute_angular_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """Compute angular acceleration (rate of change of angular velocity)"""
        num_agents = velocity.shape[1] if velocity.ndim > 1 else 1
        angular_accels = np.zeros(num_agents)
        
        if velocity.shape[0] < 3:
            return angular_accels
        
        # Compute angular velocity at each time step
        angular_vels = []
        for t in range(len(velocity) - 1):
            omega_t = []
            for i in range(num_agents):
                v_t = velocity[t, i] if velocity.ndim > 1 else velocity[t]
                v_t1 = velocity[t+1, i] if velocity.ndim > 1 else velocity[t+1]
                
                v_t_norm = np.linalg.norm(v_t)
                v_t1_norm = np.linalg.norm(v_t1)
                
                if v_t_norm > 1e-8 and v_t1_norm > 1e-8:
                    v_t_unit = v_t / v_t_norm
                    v_t1_unit = v_t1 / v_t1_norm
                    dot = np.clip(np.dot(v_t_unit, v_t1_unit), -1.0, 1.0)
                    angle = np.arccos(dot)
                    omega_t.append(angle / self.dt)
                else:
                    omega_t.append(0.0)
            angular_vels.append(omega_t)
        
        if len(angular_vels) >= 2:
            angular_vels = np.array(angular_vels)  # (time_steps, agents)
            for i in range(num_agents):
                omega_start = angular_vels[0, i]
                omega_end = angular_vels[-1, i]
                angular_accels[i] = (omega_end - omega_start) / (len(angular_vels) * self.dt + 1e-8)
        
        return angular_accels
    
    def _classify_behavior(self, 
                          vel_mag: np.ndarray,
                          a_tangent: np.ndarray,
                          a_normal: np.ndarray,
                          curvature: np.ndarray,
                          angular_velocity: np.ndarray,
                          velocity_change_rate: np.ndarray,
                          acceleration_change_rate: np.ndarray,
                          curvature_change_rate: np.ndarray,
                          angular_acceleration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify motion behavior
        
        Returns:
            behavior: (agents,) MotionBehavior enum array
            confidence: (agents,) confidence array
        """
        num_agents = len(vel_mag)
        behaviors = np.full(num_agents, MotionBehavior.UNKNOWN, dtype=object)
        confidences = np.zeros(num_agents)
        
        for i in range(num_agents):
            # Sharp turn detection
            if (angular_velocity[i] > self.angular_velocity_threshold * 2 and 
                curvature[i] > self.curvature_threshold * 2):
                behaviors[i] = MotionBehavior.SHARP_TURN
                confidences[i] = min(1.0, angular_velocity[i] / (self.angular_velocity_threshold * 2))
            
            # Turning detection
            elif (curvature[i] > self.curvature_threshold and 
                  a_normal[i] > self.acceleration_threshold * 0.5):
                behaviors[i] = MotionBehavior.TURNING
                confidences[i] = min(1.0, curvature[i] / self.curvature_threshold)
            
            # Accelerating detection
            elif (a_tangent[i] > self.acceleration_threshold and 
                  velocity_change_rate[i] > 0 and
                  acceleration_change_rate[i] > 0):
                behaviors[i] = MotionBehavior.ACCELERATING
                confidences[i] = min(1.0, a_tangent[i] / (self.acceleration_threshold * 2))
            
            # Decelerating detection
            elif (a_tangent[i] < -self.acceleration_threshold and 
                  velocity_change_rate[i] < 0):
                behaviors[i] = MotionBehavior.DECELERATING
                confidences[i] = min(1.0, abs(a_tangent[i]) / (self.acceleration_threshold * 2))
            
            # Constant speed detection
            elif (abs(a_tangent[i]) < self.acceleration_threshold * 0.3 and
                  abs(curvature[i]) < self.curvature_threshold * 0.3):
                behaviors[i] = MotionBehavior.CONSTANT_SPEED
                confidences[i] = 0.5
        
        return behaviors, confidences
    
    def _compute_enhancement_params(self,
                                   behavior: np.ndarray,
                                   confidence: np.ndarray,
                                   vel_dir: np.ndarray,
                                   a_tangent: np.ndarray,
                                   a_normal: np.ndarray,
                                   curvature: np.ndarray,
                                   angular_velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute enhancement parameters
        
        Returns:
            enhancement_factor: (agents,) enhancement factor
            enhancement_direction: (agents, 3) enhancement direction
        """
        num_agents = len(behavior)
        enhancement_factors = np.ones(num_agents)
        enhancement_directions = vel_dir.copy()  # Default: along velocity direction
        
        for i in range(num_agents):
            conf = confidence[i]
            beh = behavior[i]
            
            if beh == MotionBehavior.ACCELERATING:
                # Accelerating: enhance velocity direction
                enhancement_factors[i] = 1.0 + conf * 0.3  # Up to 30% enhancement
                enhancement_directions[i] = vel_dir[i]
            
            elif beh == MotionBehavior.TURNING:
                # Turning: enhance normal acceleration direction
                if a_normal[i] > 1e-8:
                    # Compute normal direction
                    vel_unit = vel_dir[i]
                    # Compute unit vector perpendicular to velocity
                    if abs(vel_unit[0]) < 0.9:
                        perp_dir = np.array([1, 0, 0]) - vel_unit[0] * vel_unit
                    else:
                        perp_dir = np.array([0, 1, 0]) - vel_unit[1] * vel_unit
                    perp_dir = perp_dir / (np.linalg.norm(perp_dir) + 1e-8)
                    enhancement_directions[i] = perp_dir
                    enhancement_factors[i] = 1.0 + conf * 0.4  # Up to 40% enhancement
                else:
                    enhancement_factors[i] = 1.0
            
            elif beh == MotionBehavior.SHARP_TURN:
                # Sharp turn: significantly enhance normal direction
                vel_unit = vel_dir[i]
                if abs(vel_unit[0]) < 0.9:
                    perp_dir = np.array([1, 0, 0]) - vel_unit[0] * vel_unit
                else:
                    perp_dir = np.array([0, 1, 0]) - vel_unit[1] * vel_unit
                perp_dir = perp_dir / (np.linalg.norm(perp_dir) + 1e-8)
                enhancement_directions[i] = perp_dir
                enhancement_factors[i] = 1.0 + conf * 0.6  # Up to 60% enhancement
            
            elif beh == MotionBehavior.DECELERATING:
                # Decelerating: slight reverse enhancement
                enhancement_factors[i] = 1.0 - conf * 0.2  # Up to 20% reduction
                enhancement_directions[i] = -vel_dir[i]
            
            else:
                # Other cases: no enhancement
                enhancement_factors[i] = 1.0
        
        return enhancement_factors, enhancement_directions


# =====================================================================
# Prediction Enhancement Function
# =====================================================================

def enhance_prediction_with_tail_dynamics(
    pred_delta: np.ndarray,
    x_orig_batch: np.ndarray,
    tail_analyzer: EnhancedTailDynamicsAnalyzer,
    decay_factor: float = 0.15
) -> np.ndarray:
    """
    Enhance prediction displacement using tail dynamics information
    
    Args:
        pred_delta: (B, seq_out, agents, 3) Original predicted displacement
        x_orig_batch: (B, seq_in, agents, 3) Input positions
        tail_analyzer: EnhancedTailDynamicsAnalyzer instance
        decay_factor: Time decay factor (larger = faster decay)
    
    Returns:
        pred_delta_enhanced: (B, seq_out, agents, 3) Enhanced displacement
    """
    pred_delta_enhanced = pred_delta.copy()
    batch_size, seq_out, num_agents, _ = pred_delta.shape
    
    for b in range(batch_size):
        # Analyze tail dynamics for this sample
        tail_info = tail_analyzer.compute_tail_dynamics(x_orig_batch[b])
        
        for agent in range(num_agents):
            behavior = tail_info.behavior[agent]
            confidence = tail_info.behavior_confidence[agent]
            enhancement_factor = tail_info.enhancement_factor[agent]
            enhancement_dir = tail_info.enhancement_direction[agent]
            
            # Skip enhancement if confidence is too low
            if confidence < 0.3:
                continue
            
            # Compute enhancement amount
            base_enhancement = enhancement_factor - 1.0  # Relative enhancement
            
            for t in range(seq_out):
                # Get current step predicted displacement
                orig_delta = pred_delta_enhanced[b, t, agent, :]
                orig_mag = np.linalg.norm(orig_delta)
                
                if orig_mag < 1e-8:
                    continue
                
                # Time decay: farther prediction steps get less enhancement
                time_decay = np.exp(-decay_factor * t)
                
                # Compute enhancement amount
                enhancement_mag = orig_mag * base_enhancement * confidence * time_decay
                
                # Apply enhancement
                enhancement_vec = enhancement_dir * enhancement_mag
                pred_delta_enhanced[b, t, agent, :] += enhancement_vec
    
    return pred_delta_enhanced


# =====================================================================
# Physical Constraints Function
# =====================================================================

def apply_physical_constraints(history, pred_delta, dt=0.1, smoothing_weight=0.3):
    """
    Apply physical constraints to achieve smoother position reconstruction.
    
    Improved version that:
    1. Matches velocity scale from input sequence end
    2. Adjusts speed based on acceleration trends (acceleration/deceleration)
    3. Ensures smooth transitions and continuous trajectories
    
    Args:
        history: (B, seq_in, agents, 3) Input position history
        pred_delta: (B, seq_out, agents, 3) Predicted displacement (total delta from last position)
        dt: Time step
        smoothing_weight: Weight for acceleration smoothing (0-1)
    
    Returns:
        reconstructed: (B, seq_out, agents, 3) Smooth reconstructed absolute positions
    """
    history = np.array(history, dtype=np.float32)
    B, seq, agents, _ = history.shape
    if seq < 2:
        history_vel = np.zeros((B, 1, agents, 3), dtype=np.float32)
    else:
        history_vel = np.diff(history, axis=1) / dt

    if history_vel.shape[1] == 0:
        history_vel = np.zeros((B, 1, agents, 3), dtype=np.float32)

    # Compute last velocity from history (use last few steps for stability)
    # Use last 3 steps for more recent velocity estimate
    if history_vel.shape[1] >= 3:
        last_vel = history_vel[:, -3:, :, :].mean(axis=1)  # (B, agents, 3)
        # Also get the most recent velocity for direction
        last_vel_recent = history_vel[:, -1, :, :]  # (B, agents, 3)
    elif history_vel.shape[1] > 0:
        last_vel = history_vel[:, -1, :, :]  # (B, agents, 3)
        last_vel_recent = last_vel
    else:
        last_vel = np.zeros((B, agents, 3), dtype=np.float32)
        last_vel_recent = last_vel

    # Compute acceleration from history to detect acceleration/deceleration trends
    history_acc = (
        np.diff(history_vel, axis=1) / dt if history_vel.shape[1] > 1 else np.zeros((B, 1, agents, 3), dtype=np.float32)
    )
    
    # Average acceleration (overall trend)
    avg_acc = (
        history_acc.mean(axis=1, keepdims=True)
        if history_acc.shape[1] > 0
        else np.zeros((B, 1, agents, 3), dtype=np.float32)
    )
    
    # Recent acceleration trend (last few steps) to detect current acceleration/deceleration
    if history_acc.shape[1] >= 3:
        recent_acc = history_acc[:, -3:, :, :].mean(axis=1)  # (B, agents, 3)
    elif history_acc.shape[1] > 0:
        recent_acc = history_acc[:, -1, :, :]  # (B, agents, 3)
    else:
        recent_acc = np.zeros((B, agents, 3), dtype=np.float32)

    # Compute velocity and acceleration magnitudes for scaling
    last_vel_mag = np.linalg.norm(last_vel, axis=2, keepdims=True)  # (B, agents, 1)
    last_vel_mag = np.maximum(last_vel_mag, 1e-3)  # Avoid division by zero
    
    # Compute maximum velocity and acceleration from history (for constraints)
    max_vel = np.maximum(
        np.max(np.linalg.norm(history_vel, axis=3), axis=1, keepdims=True),
        1e-3,
    )
    max_acc = np.maximum(
        np.max(np.linalg.norm(history_acc, axis=3), axis=1, keepdims=True),
        1e-3,
    )
    max_vel = max_vel[:, 0, :, np.newaxis]
    max_acc = max_acc[:, 0, :, np.newaxis]

    # Start from last position with last velocity
    current_pos = history[:, -1, :, :].copy()
    current_vel = last_vel.copy()
    steps = pred_delta.shape[1]
    reconstructed = np.zeros((B, steps, agents, 3), dtype=np.float32)

    # Reconstruct positions step by step with improved velocity matching
    for step in range(steps):
        # Compute desired velocity direction from predicted displacement
        # pred_delta is cumulative displacement from start, so we need incremental
        if step == 0:
            # First step: use the first predicted delta
            step_delta = pred_delta[:, step, :, :]  # (B, agents, 3)
        else:
            # Subsequent steps: compute incremental delta
            step_delta = pred_delta[:, step, :, :] - pred_delta[:, step-1, :, :]  # (B, agents, 3)
        
        # Desired velocity direction from predicted displacement
        desired_vel_dir = step_delta / (np.linalg.norm(step_delta, axis=2, keepdims=True) + 1e-8)  # (B, agents, 3)
        
        # Match velocity magnitude to input sequence end, with acceleration adjustment
        # Base velocity magnitude from input sequence end
        base_vel_mag = last_vel_mag  # (B, agents, 1)
        
        # Adjust velocity magnitude based on acceleration trend
        # If accelerating, increase speed; if decelerating, decrease speed
        recent_acc_mag = np.linalg.norm(recent_acc, axis=2, keepdims=True)  # (B, agents, 1)
        recent_acc_dir = recent_acc / (recent_acc_mag + 1e-8)  # (B, agents, 3)
        
        # Project acceleration onto velocity direction to get tangential acceleration
        accel_tangent = np.sum(recent_acc * (last_vel / (last_vel_mag + 1e-8)), axis=2, keepdims=True)  # (B, agents, 1)
        
        # Adjust velocity magnitude based on acceleration trend
        # Positive tangential acceleration -> speed up, negative -> slow down
        # Use exponential decay to gradually apply acceleration effect
        accel_factor = 1.0 + np.tanh(accel_tangent * dt * 2.0) * 0.3  # Scale: 0.7 to 1.3
        target_vel_mag = base_vel_mag * accel_factor
        
        # Combine direction from prediction with magnitude from input sequence
        desired_vel = desired_vel_dir * target_vel_mag  # (B, agents, 3)
        
        # Compute required acceleration to reach desired velocity
        raw_accel = (desired_vel - current_vel) / dt
        
        # Smooth acceleration with historical average (weighted by recent trend)
        accel_weight = 0.4  # Weight for recent acceleration trend
        constrained_accel = (
            (1 - smoothing_weight) * raw_accel + 
            smoothing_weight * (1 - accel_weight) * avg_acc[:, 0, :, :] +
            smoothing_weight * accel_weight * recent_acc
        )

        # Constrain acceleration magnitude
        accel_norm = np.linalg.norm(constrained_accel, axis=2, keepdims=True)
        accel_scale = np.minimum(1.0, (max_acc / (accel_norm + 1e-8)))
        constrained_accel = constrained_accel * accel_scale

        # Update velocity
        new_vel = current_vel + constrained_accel * dt
        
        # Constrain velocity magnitude (but allow it to match input sequence scale)
        vel_norm = np.linalg.norm(new_vel, axis=2, keepdims=True)  # (B, agents, 1)
        # Use a more lenient constraint: allow up to 1.5x max velocity to match input scale
        vel_scale = np.minimum(1.5, (max_vel * 1.5 / (vel_norm + 1e-8)))
        # But don't scale down if velocity is close to input sequence scale (per agent)
        # If velocity is within 1.2x of max, keep it as is to match input sequence scale
        vel_scale = np.where(vel_norm <= max_vel * 1.2, 1.0, vel_scale)
        current_vel = new_vel * vel_scale

        # Update position
        current_pos = current_pos + current_vel * dt
        reconstructed[:, step, :, :] = current_pos

    return reconstructed


# =====================================================================
# Helper Functions
# =====================================================================

def load_all_32d_features(features_dir, num_agents, use_subset=False):
    """Load all 32D features into memory at once to avoid repeated disk reads"""
    features_dir = Path(features_dir)
    subset_suffix = '_subset' if use_subset else ''
    
    feature_candidates = [
        features_dir / f'features_agents_{num_agents}{subset_suffix}_32d.npz',
        features_dir / f'features_agents_{num_agents}_32d{subset_suffix}.npz',
        features_dir / f'features_agents_{num_agents}{subset_suffix}_features.npz',
    ]
    
    for feat_path in feature_candidates:
        if feat_path.exists():
            try:
                logger.info(f"Preloading feature file: {feat_path} ...")
                data = np.load(feat_path)
                features_all = np.asarray(data['features'])
                logger.info(f"Feature preloading complete: {features_all.shape}")
                return features_all
            except Exception as e:
                logger.warning(f"Failed to preload feature file {feat_path}: {e}")
    return None


def load_data_robust(data_dir, num_agents, use_subset=False):
    """Load input/output data pairs"""
    data_path = Path(data_dir)
    
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    subset_suffix = '_subset' if use_subset else ''
    input_file = data_path / f"input_agents_{num_agents}{subset_suffix}.npz"
    output_file = data_path / f"output_agents_{num_agents}{subset_suffix}.npz"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    if not output_file.exists():
        raise FileNotFoundError(f"Output file does not exist: {output_file}")
    
    logger.info("Loading input data: %s", input_file)
    input_data = np.load(input_file)
    X_raw = input_data["data"]
    logger.info("Input raw shape: %s", X_raw.shape)
    
    logger.info("Loading output data: %s", output_file)
    output_data = np.load(output_file)
    Y_raw = output_data["data"]
    logger.info("Output raw shape: %s", Y_raw.shape)
    
    # Transpose to (samples, seq, agents, 3)
    X = np.transpose(X_raw, (1, 0, 2, 3))
    Y = np.transpose(Y_raw, (1, 0, 2, 3))
    
    logger.info("After transpose - Input: %s, Output: %s", X.shape, Y.shape)
    
    assert X.shape[2] == num_agents, f"Input agents count {X.shape[2]} != expected {num_agents}"
    assert Y.shape[2] == num_agents, f"Output agents count {Y.shape[2]} != expected {num_agents}"
    
    return X, Y


def detect_model_version(checkpoint):
    """Detect model version (v2/v3/v4)"""
    config = checkpoint.get("config", {})
    
    # Priority: check model_version in config
    if 'model_version' in config:
        model_version = str(config['model_version']).lower()
        if 'v4' in model_version:
            return 'v4'
        elif 'v3' in model_version:
            return 'v3'
        elif 'v2' in model_version:
            return 'v2'
    
    # Check feature dimension
    if 'input_features' in config:
        input_dim = config['input_features']
        if input_dim == 32:
            return 'v4'
    
    # Check use_gnn flag
    if 'use_gnn' in config:
        return 'v3' if config['use_gnn'] else 'v2'
    
    # Check GNN-related parameters
    if any(key in config for key in ['gnn_hidden', 'gnn_heads', 'edge_threshold']):
        return 'v3'
    
    # Default v2
    return 'v2'


def compute_feature_statistics(features_all_cache, num_samples=1000):
    """
    Compute feature statistics from preloaded features (consistent with training script)
    
    Args:
        features_all_cache: (samples, seq_in, agents, 32) preloaded features
        num_samples: Number of samples to use for statistics
    
    Returns:
        feature_mean: (32,) feature mean
        feature_std: (32,) feature std
    """
    if features_all_cache is None or features_all_cache.size == 0:
        logger.warning("No preloaded features, using default statistics")
        return np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)
    
    # Use subset for statistics computation
    subset_size = min(num_samples, len(features_all_cache))
    subset_for_stats = features_all_cache[:subset_size].reshape(-1, 32)
    
    feature_mean = np.mean(subset_for_stats, axis=0)  # (32,)
    feature_std = np.std(subset_for_stats, axis=0)  # (32,)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)  # Handle zero variance
    
    logger.info(f"Computed feature statistics from {subset_size} samples")
    logger.info(f"  Feature mean shape: {feature_mean.shape}, std shape: {feature_std.shape}")
    
    return feature_mean.astype(np.float32), feature_std.astype(np.float32)


def normalize_features(features, feature_mean, feature_std):
    """
    Normalize features using Z-score (consistent with training script)
    
    Args:
        features: (B, seq_in, agents, 32) or (seq_in, agents, 32) features
        feature_mean: (32,) feature mean
        feature_std: (32,) feature std
    
    Returns:
        normalized_features: Same shape as input, normalized
    """
    # Reshape for broadcasting
    if features.ndim == 4:
        # (B, seq_in, agents, 32)
        mean_vec = feature_mean.reshape(1, 1, 1, 32)
        std_vec = feature_std.reshape(1, 1, 1, 32)
    else:
        # (seq_in, agents, 32)
        mean_vec = feature_mean.reshape(1, 1, 32)
        std_vec = feature_std.reshape(1, 1, 32)
    
    # Z-score normalization
    safe_std = np.where(std_vec < 1e-8, 1.0, std_vec)
    normalized = (features - mean_vec) / safe_std
    
    # Clip outliers (beyond +/-5 sigma)
    normalized = np.clip(normalized, -5.0, 5.0)
    
    return normalized.astype(np.float32)


# =====================================================================
# Inference Batch Function
# =====================================================================

def infer_batch_v4_enhanced(model, 
                            features_batch, 
                            x_orig_batch, 
                            device, 
                            output_mean, 
                            output_std,
                            tail_analyzer: Optional[EnhancedTailDynamicsAnalyzer],
                            edge_threshold=5.0,
                            use_gnn=True,
                            use_tail_enhancement=True,
                            debug=False):
    """
    Inference for one batch (v4 enhanced) and return absolute position predictions
    
    Args:
        model: Model instance
        features_batch: (B, seq_in, agents, 32) 32D feature input
        x_orig_batch: (B, seq_in, agents, 3) Original position input
        device: torch device
        output_mean: (3,) Output delta mean
        output_std: (3,) Output delta std
        tail_analyzer: EnhancedTailDynamicsAnalyzer instance (optional)
        edge_threshold: Adjacency matrix distance threshold
        use_gnn: Whether to use GNN
        use_tail_enhancement: Whether to use tail enhancement
        debug: Whether to print diagnostic information
    
    Returns:
        pred_absolute: np.array (B, seq_out, agents, 3) Absolute position predictions
    """
    model.eval()
    with torch.no_grad():
        features_t = torch.from_numpy(features_batch).float().to(device)
        x_orig_t = torch.from_numpy(x_orig_batch).float().to(device)
        
        # Model forward pass
        pred_delta_norm, _, _ = model(
            features_t, x_orig_t,
            y=None, y_velocity=None, y_accel=None,
            teacher_forcing_ratio=0.0
        )
        
        output_mean_t = torch.tensor(output_mean, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        output_std_t = torch.tensor(output_std, dtype=torch.float32, device=device).view(1, 1, 1, 3)
        
        # Denormalize to get physical displacement (B, seq_out, agents, 3)
        # Model training target: total displacement y_delta = Y_t - X_last
        pred_delta_phys = (pred_delta_norm * output_std_t + output_mean_t).cpu().numpy()
        
        # Tail dynamics enhancement (applied to total displacement)
        if use_tail_enhancement and tail_analyzer is not None:
            pred_delta_phys = enhance_prediction_with_tail_dynamics(
                pred_delta_phys, x_orig_batch, tail_analyzer, decay_factor=0.15
            )
        
        # Apply physical constraints to reconstruct smooth trajectories
        # This ensures continuity with input sequence and smooth turns
        pred_absolute = apply_physical_constraints(
            x_orig_batch,
            pred_delta_phys,
            dt=0.1,
            smoothing_weight=0.3
        )
        
        if debug and features_batch.shape[0] > 0:
            logger.info("=== v4 Enhanced Tail Inference Diagnostics ===")
            logger.info(f"Normalized delta range: [{pred_delta_norm.min().item():.4f}, {pred_delta_norm.max().item():.4f}]")
            if use_tail_enhancement:
                logger.info(f"Tail dynamics enhancement: Enabled")
                # Print behavior classification for first sample
                tail_info = tail_analyzer.compute_tail_dynamics(x_orig_batch[0])
                for agent in range(x_orig_batch.shape[2]):
                    logger.info(f"  Agent {agent}: {tail_info.behavior[agent].value}, "
                              f"confidence={tail_info.behavior_confidence[agent]:.3f}, "
                              f"enhancement_factor={tail_info.enhancement_factor[agent]:.3f}")
    
    return pred_absolute


# =====================================================================
# Main Function
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="v4 Enhanced Tail Dynamics Inference Script")
    parser.add_argument("--model", required=True, help=".pt model file path")
    parser.add_argument("--data_dir", default="swarm_segments", help="Data directory")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=22, help="Number of evaluation samples, -1 for all")
    parser.add_argument("--random_sample", action="store_true", help="Whether to randomly sample")
    parser.add_argument("--output_dir", default="infer_results_v4_enhanced", help="Output directory")
    parser.add_argument("--features_dir", type=str, default="features_32d", help="32D features directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force_v4", action="store_true", help="Force v4 model")
    parser.add_argument("--no_gnn", action="store_true", help="Do not use GNN")
    parser.add_argument("--use_subset", action="store_true", help="Use _subset data")
    parser.add_argument("--edge_threshold", type=float, default=5.0, help="GNN adjacency threshold")
    
    # Tail enhancement parameters
    parser.add_argument("--tail_window", type=int, default=8, help="Tail observation window size (long window)")
    parser.add_argument("--short_window", type=int, default=3, help="Short window size (rapid change detection)")
    parser.add_argument("--medium_window", type=int, default=5, help="Medium window size (trend detection)")
    parser.add_argument("--use_multi_scale", action="store_true", help="Use multi-scale window analysis")
    parser.add_argument("--no_tail_enhancement", action="store_true", help="Disable tail enhancement")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    logger.info("Loading checkpoint: %s", args.model)
    try:
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model, weights_only=False)
    
    config = checkpoint.get("config", {})
    if not config:
        logger.warning("No config in checkpoint, assuming v4 model")
        config = {}
    
    # Detect model version
    if args.force_v4:
        model_version = 'v4'
    else:
        model_version = detect_model_version(checkpoint)
    
    logger.info(f"Detected model version: {model_version}")
    
    # Create model
    if model_version in ['v3', 'v4'] and not args.no_gnn:
        use_gnn = True
        model = DynamicsAwareSwarmGRUModel_with_GNN(
            input_size=config.get('input_size', 32),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 3),
            output_size=3,
            dropout=config.get('dropout', 0.3),
            use_attention=config.get('use_attention', True),
            gnn_hidden=config.get('gnn_hidden', 64),
            num_gnn_heads=config.get('gnn_heads', 4),  # Note: config uses 'gnn_heads' but class expects 'num_gnn_heads'
            edge_threshold=config.get('edge_threshold', 5.0),
            fusion_mode=config.get('gnn_fusion_mode', 'concat'),
        )
    else:
        use_gnn = False
        model = DynamicsAwareSwarmGRUModel(
            input_size=config.get('input_size', 32),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 3),
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    
    # Load statistics
    if "output_mean" not in checkpoint or "output_std" not in checkpoint:
        logger.error("Missing output_mean or output_std in checkpoint")
        sys.exit(1)
    
    output_mean = np.array(checkpoint["output_mean"], dtype=np.float32)
    output_std = np.array(checkpoint["output_std"], dtype=np.float32)
    logger.info("Loaded output statistics from checkpoint: output_mean=%s, output_std=%s", output_mean, output_std)
    
    # Initialize tail dynamics analyzer
    tail_analyzer = None
    if not args.no_tail_enhancement:
        if args.use_multi_scale:
            tail_analyzer = EnhancedTailDynamicsAnalyzer(
                short_window=args.short_window,
                medium_window=args.medium_window,
                long_window=args.tail_window,
                dt=0.1
            )
        else:
            tail_analyzer = EnhancedTailDynamicsAnalyzer(
                short_window=args.tail_window // 3,
                medium_window=args.tail_window // 2,
                long_window=args.tail_window,
                dt=0.1
            )
        logger.info("Tail dynamics enhancement enabled")
        logger.info(f"  Short window: {tail_analyzer.short_window} steps")
        logger.info(f"  Medium window: {tail_analyzer.medium_window} steps")
        logger.info(f"  Long window: {tail_analyzer.long_window} steps")
    else:
        logger.info("Tail dynamics enhancement disabled")
    
    # Load data
    logger.info("Loading data: %s (use_subset=%s)", args.data_dir, args.use_subset)
    X_all, Y_all = load_data_robust(args.data_dir, args.agents, use_subset=args.use_subset)
    logger.info("Loaded data: X_all shape=%s, Y_all shape=%s", X_all.shape, Y_all.shape)
    
    # Load 32D features and compute statistics
    features_all_cache = None
    feature_mean = None
    feature_std = None
    
    if args.features_dir:
        features_all_cache = load_all_32d_features(args.features_dir, args.agents, use_subset=args.use_subset)
        if features_all_cache is not None:
            logger.info("Loaded features: shape=%s", features_all_cache.shape)
            feature_mean, feature_std = compute_feature_statistics(features_all_cache, num_samples=1000)
        else:
            logger.warning("Precomputed feature file not found")
    
    # Sample selection
    total_samples = len(X_all)
    logger.info("Total samples available: %d", total_samples)
    
    original_indices = None  # Store original dataset indices for visualization
    if args.num_samples > 0 and args.num_samples < total_samples:
        if args.random_sample:
            # Ensure random seed is set before sampling for reproducibility
            np.random.seed(args.seed)
            # Randomly sample from the entire dataset (all 23w samples)
            indices = np.random.choice(total_samples, args.num_samples, replace=False)
            original_indices = indices.copy()  # Save original dataset indices
            X_all = X_all[indices]
            Y_all = Y_all[indices]
            if features_all_cache is not None:
                features_all_cache = features_all_cache[indices]
            logger.info("Randomly sampled %d samples (from %d total) with seed=%d", 
                       args.num_samples, total_samples, args.seed)
            logger.info("Selected indices range: [%d, %d]", indices.min(), indices.max())
        else:
            indices = np.arange(args.num_samples)
            original_indices = indices.copy()  # Save original dataset indices
            X_all = X_all[indices]
            Y_all = Y_all[indices]
            if features_all_cache is not None:
                features_all_cache = features_all_cache[indices]
            logger.info("Using first %d samples (from %d total)", args.num_samples, total_samples)
            logger.info("Selected sample indices: [0, %d]", args.num_samples - 1)
    else:
        original_indices = np.arange(total_samples)  # All samples
        logger.info("Using all %d samples", total_samples)
    
    logger.info("Number of samples to evaluate: %d", len(X_all))
    
    # Inference
    predictions = []
    for start in tqdm(range(0, len(X_all), args.batch_size), desc="Inference progress"):
        end = min(start + args.batch_size, len(X_all))
        
        X_batch = X_all[start:end]
        
        # Load 32D features
        if features_all_cache is not None:
            features_batch = features_all_cache[start:end].astype(np.float32)
            # Normalize features (consistent with training)
            if feature_mean is not None and feature_std is not None:
                features_batch = normalize_features(features_batch, feature_mean, feature_std)
        else:
            logger.error("Precomputed feature file required")
            sys.exit(1)
        
        # Inference
        debug_flag = (start == 0)
        pred_batch = infer_batch_v4_enhanced(
            model, features_batch, X_batch, device,
            output_mean, output_std,
            tail_analyzer=tail_analyzer,
            edge_threshold=args.edge_threshold,
            use_gnn=use_gnn,
            use_tail_enhancement=not args.no_tail_enhancement,
            debug=debug_flag
        )
        
        predictions.append(pred_batch)
    
    predictions = np.concatenate(predictions, axis=0)
    logger.info("Inference complete, prediction shape: %s", predictions.shape)
    logger.info("Ground truth shape: %s", Y_all.shape)
    
    # Verify shapes match (same as v3)
    if predictions.shape != Y_all.shape:
        logger.error("Shape mismatch! predictions: %s, Y_all: %s", predictions.shape, Y_all.shape)
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} != Y_all {Y_all.shape}")
    
    # Evaluation (same calculation as v3)
    # Overall MAE: mean over all dimensions (samples, steps, agents, coordinates)
    mae = np.mean(np.abs(predictions - Y_all))
    rmse = np.sqrt(np.mean((predictions - Y_all) ** 2))
    
    # Per-axis MAE: mean over (samples, steps, agents)
    mae_x = np.mean(np.abs(predictions[..., 0] - Y_all[..., 0]))
    mae_y = np.mean(np.abs(predictions[..., 1] - Y_all[..., 1]))
    mae_z = np.mean(np.abs(predictions[..., 2] - Y_all[..., 2]))
    
    # MAE per step: mean over (samples, agents, coordinates) -> (steps,)
    mae_per_step = np.mean(np.abs(predictions - Y_all), axis=(0, 2, 3))
    
    # MAE per agent: mean over (samples, steps, coordinates) -> (agents,)
    mae_per_agent = np.mean(np.abs(predictions - Y_all), axis=(0, 1, 3))
    
    logger.info("\n=== Evaluation Results ===")
    logger.info("Overall MAE: %.6f m (%.2f cm), RMSE: %.6f m", mae, mae*100, rmse)
    logger.info("MAE (X/Y/Z): %.6f / %.6f / %.6f m", mae_x, mae_y, mae_z)
    
    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = out_dir / f"predictions_agents_{args.agents}_v4_enhanced.npz"
    save_dict = {
        'input': X_all,
        'truth': Y_all,
        'prediction': predictions,
        'mae': mae,
        'rmse': rmse,
        'mae_per_step': mae_per_step,
        'mae_per_agent': mae_per_agent,
    }
    # Save original dataset indices if available (for visualization)
    if original_indices is not None:
        save_dict['original_indices'] = original_indices
        logger.info("Saved original dataset indices: min=%d, max=%d", 
                   original_indices.min(), original_indices.max())
    np.savez(result_file, **save_dict)
    logger.info("Results saved: %s", result_file)
    
    report_file = out_dir / f"evaluation_report_agents_{args.agents}_v4_enhanced.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Swarm Trajectory Prediction Evaluation Report (v4 Enhanced Tail)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model version: {model_version}\n")
        f.write(f"Model path: {args.model}\n")
        f.write(f"Number of agents: {args.agents}\n")
        f.write(f"Number of samples: {len(X_all)}\n")
        f.write(f"Use GNN: {use_gnn}\n")
        f.write(f"Feature dimension: 32D (24D + 8D curvature features)\n")
        f.write(f"Tail enhancement: {'Enabled' if not args.no_tail_enhancement else 'Disabled'}\n")
        if tail_analyzer:
            f.write(f"  Short window: {tail_analyzer.short_window} steps\n")
            f.write(f"  Medium window: {tail_analyzer.medium_window} steps\n")
            f.write(f"  Long window: {tail_analyzer.long_window} steps\n")
        f.write("\n")
        f.write(f"Overall MAE: {mae:.6f} m ({mae*100:.2f} cm)\n")
        f.write(f"Overall RMSE: {rmse:.6f} m\n")
        f.write(f"MAE (X): {mae_x:.6f} m\n")
        f.write(f"MAE (Y): {mae_y:.6f} m\n")
        f.write(f"MAE (Z): {mae_z:.6f} m\n\n")
        f.write("MAE per step:\n")
        for step, mae_step in enumerate(mae_per_step):
            f.write(f"  Step {step}: {mae_step:.6f} m\n")
    
    logger.info("Evaluation report saved: %s", report_file)
    logger.info("Inference pipeline complete")


if __name__ == "__main__":
    main()
