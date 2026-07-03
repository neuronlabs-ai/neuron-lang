# ═══════════════════════════════════════════════════════════════════
#  NEURON NeuroCognitive AGI — Large-Scale Training Script
#  Transpiled architecture + PyTorch training harness
#  Target: 500K+ parameters, T4 GPU, FP32
# ═══════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NEURON] Training device: {device}")

# ─────────────────────────────────────────────
#  Hyperparameters (scaled up from NEURON source)
# ─────────────────────────────────────────────
STATE_DIM    = 64       # Environment observation embedding
ACTION_DIM   = 18       # Discrete action space
HIDDEN_DIM   = 256      # MLP hidden layer width
LATENT_DIM   = 128      # World model latent space
MEMORY_CAP   = 2048     # Episodic memory capacity
WM_SLOTS     = 32       # Working memory slots
BATCH_SIZE   = 64       # Training batch size
NUM_EPISODES = 5000     # Total training episodes
STEPS_PER_EP = 50       # Steps per episode
LR           = 3e-4     # Learning rate
GAMMA        = 0.99     # Reward discount
CURIOSITY_COEFF = 0.5   # Intrinsic reward weight

# ─────────────────────────────────────────────
#  Episodic Memory System
#  (from NEURON agent EpisodicMemorySystem)
# ─────────────────────────────────────────────
class EpisodicMemory(nn.Module):
    def __init__(self, embed_dim, capacity):
        super().__init__()
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Buffers (non-learnable storage)
        self.register_buffer('keys', torch.zeros(capacity, embed_dim))
        self.register_buffer('values', torch.zeros(capacity, embed_dim))
        self.register_buffer('timestamps', torch.zeros(capacity, 1))
        self.register_buffer('write_pos', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_size', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_time', torch.tensor(0, dtype=torch.long))

    def push(self, key, value):
        """Store a new experience in episodic memory."""
        pos = self.write_pos.item() % self.capacity
        self.keys[pos] = key.detach()
        self.values[pos] = value.detach()
        self.timestamps[pos] = self.current_time.float()
        self.write_pos += 1
        self.current_size = min(self.current_size + 1, self.capacity)
        self.current_time += 1

    def recall(self, query, k=4):
        """Retrieve top-k relevant memories via attention."""
        if self.current_size.item() == 0:
            return torch.zeros(1, self.embed_dim, device=query.device)
        n = self.current_size.item()
        q = self.query_proj(query)                           # [1, D]
        k_proj = self.key_proj(self.keys[:n])                # [N, D]
        scale = self.embed_dim ** -0.5
        scores = (q @ k_proj.T) * scale                     # [1, N]
        # Recency weighting
        time_diff = (self.current_time.float() - self.timestamps[:n].squeeze(-1)).unsqueeze(0)
        recency = torch.sigmoid(-time_diff * 0.01)
        scores = scores * recency
        attn = F.softmax(scores * 10.0, dim=-1)             # [1, N]
        retrieved = attn @ self.values[:n]                   # [1, D]
        return retrieved

# ─────────────────────────────────────────────
#  Semantic Memory System
#  (from NEURON agent SemanticMemorySystem)
# ─────────────────────────────────────────────
class SemanticMemory(nn.Module):
    def __init__(self, embed_dim, capacity):
        super().__init__()
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.compose_w = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer('subjects', torch.zeros(capacity, embed_dim))
        self.register_buffer('relations', torch.zeros(capacity, embed_dim))
        self.register_buffer('objects', torch.zeros(capacity, embed_dim))
        self.register_buffer('write_pos', torch.tensor(0, dtype=torch.long))

    def store_fact(self, subject, relation, obj):
        pos = self.write_pos.item() % self.capacity
        self.subjects[pos] = subject.detach().squeeze(0)
        self.relations[pos] = relation.detach().squeeze(0)
        self.objects[pos] = obj.detach().squeeze(0)
        self.write_pos += 1

    def query(self, subject, relation):
        """Retrieve the most relevant object for a subject-relation pair."""
        query_embed = subject + relation                     # [1, D]
        scores = query_embed @ self.objects.T                # [1, Cap]
        attn = F.softmax(scores * 10.0, dim=-1)
        result = attn @ self.objects                         # [1, D]
        return result

    def associate(self, a, b):
        return self.compose_w(a + b)

# ─────────────────────────────────────────────
#  Working Memory System
#  (from NEURON agent WorkingMemorySystem)
# ─────────────────────────────────────────────
class WorkingMemory(nn.Module):
    def __init__(self, embed_dim, num_slots):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.read_key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.write_key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.write_gate = nn.Linear(embed_dim, num_slots, bias=False)
        self.erase_gate = nn.Linear(embed_dim, num_slots, bias=False)
        self.register_buffer('slots', torch.zeros(num_slots, embed_dim))

    def read(self, query):
        """Attention-based read from working memory slots."""
        key = self.read_key_proj(query)                      # [B, D]
        scores = key @ self.slots.T                          # [B, S]
        scale = self.embed_dim ** -0.5
        attn = F.softmax(scores * scale, dim=-1)             # [B, S]
        return attn @ self.slots                             # [B, D]

    def write(self, content):
        """Gated write with erase mechanism (Neural Turing Machine style)."""
        write_attn = torch.sigmoid(self.write_gate(content))  # [B, S]
        erase_vec = torch.sigmoid(self.erase_gate(content))   # [B, S]
        erase_gate_val = write_attn * erase_vec                # [B, S]
        keep_gate = 1.0 - erase_gate_val                       # [B, S]
        # Apply erase and write
        self.slots = (self.slots * keep_gate.mean(0).unsqueeze(-1) +
                      write_attn.mean(0).unsqueeze(-1) * content.mean(0).unsqueeze(0))

# ─────────────────────────────────────────────
#  ICM Curiosity Module
#  (from NEURON model CuriosityModule)
# ─────────────────────────────────────────────
class CuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.beta = 0.2
        # Feature encoder
        self.feat_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        # Forward model: predict next features from (features, action)
        self.forward_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Inverse model: predict action from (features_t, features_t+1)
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def encode(self, state):
        return self.feat_net(state)

    def intrinsic_reward(self, state, action, next_state):
        """Compute curiosity-driven intrinsic reward."""
        phi_t = self.encode(state)
        phi_tp1 = self.encode(next_state)
        action_onehot = F.one_hot(action.long(), ACTION_DIM).float()
        phi_pred = self.forward_net(torch.cat([phi_t, action_onehot], dim=-1))
        return ((phi_pred - phi_tp1) ** 2).mean(dim=-1, keepdim=True)

    def compute_loss(self, states, actions, next_states):
        """Combined forward + inverse prediction loss."""
        phi_t = self.encode(states)
        phi_tp1 = self.encode(next_states)
        action_onehot = F.one_hot(actions.long(), ACTION_DIM).float()
        # Forward loss
        phi_pred = self.forward_net(torch.cat([phi_t, action_onehot], dim=-1))
        forward_loss = F.mse_loss(phi_pred, phi_tp1.detach())
        # Inverse loss
        action_pred = self.inverse_net(torch.cat([phi_t, phi_tp1], dim=-1))
        inverse_loss = F.cross_entropy(action_pred, actions.long().squeeze(-1))
        return self.beta * forward_loss + (1 - self.beta) * inverse_loss

# ─────────────────────────────────────────────
#  World Model
#  (from NEURON model WorldModel)
# ─────────────────────────────────────────────
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder: state → latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Transition: (latent, action) → next_latent
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Reward predictor: latent → reward
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Decoder: latent → state (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def encode(self, state):
        return self.encoder(state)

    def predict_next(self, latent, action):
        action_onehot = F.one_hot(action.long(), ACTION_DIM).float()
        return self.transition(torch.cat([latent, action_onehot], dim=-1))

    def predict_reward(self, latent):
        return self.reward_head(latent)

    def decode(self, latent):
        return self.decoder(latent)

    def compute_loss(self, states, actions, next_states, rewards):
        latent = self.encode(states)
        next_latent_pred = self.predict_next(latent, actions)
        next_latent_true = self.encode(next_states)
        reward_pred = self.predict_reward(latent)
        state_pred = self.decode(next_latent_pred)
        transition_loss = F.mse_loss(next_latent_pred, next_latent_true.detach())
        reward_loss = F.mse_loss(reward_pred, rewards)
        reconstruction_loss = F.mse_loss(state_pred, next_states)
        return transition_loss + reward_loss + reconstruction_loss

# ─────────────────────────────────────────────
#  Safety Alignment Filter
#  (from NEURON fn safe_action_filter)
# ─────────────────────────────────────────────
class SafetyFilter(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.constraints = nn.Parameter(torch.ones(action_dim) * 0.1)
        self.threshold = 0.5

    def forward(self, action_logits):
        violations = action_logits * self.constraints
        is_safe = torch.sigmoid((-violations + self.threshold) * 10.0)
        return action_logits * is_safe

# ─────────────────────────────────────────────
#  Unified NeuroCognitive Agent
#  (from NEURON model NeuroCognitiveAgent)
# ─────────────────────────────────────────────
class NeuroCognitiveAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # All cognitive subsystems
        self.episodic = EpisodicMemory(STATE_DIM, MEMORY_CAP)
        self.semantic = SemanticMemory(STATE_DIM, MEMORY_CAP)
        self.working = WorkingMemory(STATE_DIM, WM_SLOTS)
        self.world_model = WorldModel(STATE_DIM, ACTION_DIM, HIDDEN_DIM, LATENT_DIM)
        self.curiosity = CuriosityModule(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        self.safety = SafetyFilter(ACTION_DIM)

        # Policy network: combined cognitive state → action
        self.policy = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM)
        )

        # Value network: combined cognitive state → expected return
        self.value = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def act(self, obs):
        """Full cognitive pipeline: memory → reasoning → safe action."""
        obs = obs.unsqueeze(0) if obs.dim() == 1 else obs

        # 1. Read working memory
        wm_context = self.working.read(obs)

        # 2. Recall episodic memory
        episodic_recall = self.episodic.recall(obs)

        # 3. Query semantic memory
        semantic_assoc = self.semantic.query(obs, obs)

        # 4. Combine all cognitive inputs
        combined = obs + wm_context + episodic_recall + semantic_assoc

        # 5. Policy output
        logits = self.policy(combined)
        safe_logits = self.safety(logits)
        action_probs = F.softmax(safe_logits, dim=-1)

        # 6. Sample action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 7. Value estimate
        value = self.value(combined)

        # 8. Update working memory
        self.working.write(combined.detach())

        # 9. Store episodic memory
        self.episodic.push(obs.squeeze(0), obs.squeeze(0))

        # 10. Store semantic fact
        self.semantic.store_fact(obs, obs, obs)

        return action.item(), log_prob, value.squeeze(-1)

# ─────────────────────────────────────────────
#  Exploration Environment
#  Procedural maze with rewards, hazards, goals
# ─────────────────────────────────────────────
class ExplorationEnv:
    def __init__(self, grid_size=8):
        self.grid_size = grid_size
        self.state_dim = STATE_DIM
        # Action mapping: 4 cardinal + 4 diagonal + special actions
        self.action_map = {
            0: (0, -1),  1: (0, 1),   2: (-1, 0),  3: (1, 0),     # N S W E
            4: (-1,-1),  5: (-1, 1),  6: (1, -1),  7: (1, 1),     # Diagonals
            8: (0, 0),   # Stay
        }
        self.reset()

    def _encode_state(self):
        """Encode grid state into a dense vector."""
        state = torch.zeros(self.state_dim, device=device)
        # Position encoding
        state[0] = self.agent_x / self.grid_size
        state[1] = self.agent_y / self.grid_size
        # Goal encoding
        state[2] = self.goal_x / self.grid_size
        state[3] = self.goal_y / self.grid_size
        # Hazard encoding
        for i, (hx, hy) in enumerate(self.hazards[:10]):
            state[4 + i*2] = hx / self.grid_size
            state[5 + i*2] = hy / self.grid_size
        # Visited count encoding (exploration signal)
        state[24] = self.visit_count.get((self.agent_x, self.agent_y), 0) / 10.0
        # Step count
        state[25] = self.step_count / self.max_steps
        # Distance to goal
        dx = abs(self.agent_x - self.goal_x)
        dy = abs(self.agent_y - self.goal_y)
        state[26] = dx / self.grid_size
        state[27] = dy / self.grid_size
        # Random noise for remaining dims (diverse representation)
        state[28:] = torch.randn(self.state_dim - 28, device=device) * 0.01
        return state

    def reset(self):
        self.agent_x = 0
        self.agent_y = 0
        self.goal_x = self.grid_size - 1
        self.goal_y = self.grid_size - 1
        self.hazards = [(np.random.randint(1, self.grid_size-1),
                         np.random.randint(1, self.grid_size-1))
                        for _ in range(self.grid_size)]
        # Remove hazard from start/goal
        self.hazards = [(x,y) for (x,y) in self.hazards
                        if (x,y) != (0,0) and (x,y) != (self.goal_x, self.goal_y)]
        self.step_count = 0
        self.max_steps = STEPS_PER_EP
        self.visit_count = {}
        self.total_reward = 0
        return self._encode_state()

    def step(self, action):
        self.step_count += 1
        # Move
        if action < len(self.action_map):
            dx, dy = self.action_map[action]
        else:
            dx, dy = 0, 0
        new_x = max(0, min(self.grid_size - 1, self.agent_x + dx))
        new_y = max(0, min(self.grid_size - 1, self.agent_y + dy))
        self.agent_x, self.agent_y = new_x, new_y

        # Track visits for exploration bonus
        pos = (self.agent_x, self.agent_y)
        self.visit_count[pos] = self.visit_count.get(pos, 0) + 1

        # Compute reward
        reward = -0.01  # Step penalty
        done = False

        # Goal reward
        if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
            reward += 10.0
            done = True

        # Hazard penalty
        if pos in [(h[0], h[1]) for h in self.hazards]:
            reward -= 1.0

        # Exploration bonus (visit novelty)
        if self.visit_count[pos] == 1:
            reward += 0.1

        self.total_reward += reward
        done = done or (self.step_count >= self.max_steps)
        return self._encode_state(), reward, done

# ─────────────────────────────────────────────
#  Experience Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (torch.stack(states), torch.tensor(actions, device=device).unsqueeze(-1).float(),
                torch.tensor(rewards, device=device).unsqueeze(-1).float(),
                torch.stack(next_states),
                torch.tensor(dones, device=device).unsqueeze(-1).float())

    def __len__(self):
        return len(self.buffer)

# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────
def train():
    print("═" * 65)
    print("  NEURON NeuroCognitive AGI — Large-Scale Training")
    print("═" * 65)

    # Initialize
    agent = NeuroCognitiveAgent().to(device)
    env = ExplorationEnv(grid_size=8)
    replay = ReplayBuffer(capacity=100000)

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Episodes: {NUM_EPISODES}")
    print("═" * 65)

    # Optimizers
    policy_optimizer = torch.optim.Adam(
        list(agent.policy.parameters()) + list(agent.value.parameters()) +
        list(agent.working.parameters()) + list(agent.episodic.parameters()) +
        list(agent.semantic.parameters()) + list(agent.safety.parameters()),
        lr=LR
    )
    world_optimizer = torch.optim.Adam(agent.world_model.parameters(), lr=LR)
    curiosity_optimizer = torch.optim.Adam(agent.curiosity.parameters(), lr=LR)

    # Tracking
    episode_rewards = []
    world_losses = []
    curiosity_losses = []
    policy_losses = []
    goals_reached = 0
    best_avg_reward = -float('inf')

    start_time = time.time()

    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        log_probs = []
        values = []
        rewards_list = []
        intrinsic_rewards = []

        for step in range(STEPS_PER_EP):
            # Agent acts through full cognitive pipeline
            action, log_prob, value = agent.act(state)

            # Environment step
            next_state, reward, done = env.step(action)

            # Compute intrinsic curiosity reward
            with torch.no_grad():
                action_t = torch.tensor([action], device=device).float()
                intrinsic = agent.curiosity.intrinsic_reward(
                    state.unsqueeze(0), action_t, next_state.unsqueeze(0)
                ).item()

            # Combined reward
            total_reward = reward + CURIOSITY_COEFF * intrinsic

            # Store experience
            replay.push(state, action, total_reward, next_state, done)
            log_probs.append(log_prob)
            values.append(value)
            rewards_list.append(total_reward)
            intrinsic_rewards.append(intrinsic)
            episode_reward += reward  # Track extrinsic only

            state = next_state
            if done:
                if env.agent_x == env.goal_x and env.agent_y == env.goal_y:
                    goals_reached += 1
                break

        episode_rewards.append(episode_reward)

        # ─── Train World Model + Curiosity (from replay buffer) ───
        if len(replay) >= BATCH_SIZE:
            s, a, r, ns, d = replay.sample(BATCH_SIZE)

            # World Model loss
            wm_loss = agent.world_model.compute_loss(s, a, ns, r)
            world_optimizer.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.world_model.parameters(), 1.0)
            world_optimizer.step()
            world_losses.append(wm_loss.item())

            # Curiosity loss
            cur_loss = agent.curiosity.compute_loss(s, a, ns)
            curiosity_optimizer.zero_grad()
            cur_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.curiosity.parameters(), 1.0)
            curiosity_optimizer.step()
            curiosity_losses.append(cur_loss.item())

        # ─── Train Policy (REINFORCE with baseline) ───
        if len(log_probs) > 0:
            # Compute discounted returns
            returns = []
            G = 0
            for r in reversed(rewards_list):
                G = r + GAMMA * G
                returns.insert(0, G)
            returns = torch.tensor(returns, device=device)
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Policy gradient loss
            policy_loss = 0
            value_loss = 0
            for lp, val, ret in zip(log_probs, values, returns):
                advantage = ret - val.detach()
                policy_loss -= lp * advantage
                value_loss += F.mse_loss(val, ret.unsqueeze(0))

            total_policy_loss = policy_loss + 0.5 * value_loss
            policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            policy_optimizer.step()
            policy_losses.append(total_policy_loss.item())

        # ─── Logging ───
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_wm = np.mean(world_losses[-100:]) if world_losses else 0
            avg_cur = np.mean(curiosity_losses[-100:]) if curiosity_losses else 0
            avg_pol = np.mean(policy_losses[-100:]) if policy_losses else 0
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.state_dict(), 'neuron_agi_best.pt')

            print(f"  Episode {episode+1:5d}/{NUM_EPISODES} │ "
                  f"Reward: {avg_reward:7.2f} │ "
                  f"WM Loss: {avg_wm:7.4f} │ "
                  f"Curiosity: {avg_cur:7.4f} │ "
                  f"Policy: {avg_pol:7.2f} │ "
                  f"Goals: {goals_reached} │ "
                  f"{eps_per_sec:.1f} ep/s")

    # ─── Final Summary ───
    elapsed = time.time() - start_time
    print("\n" + "═" * 65)
    print("  TRAINING COMPLETE")
    print("═" * 65)
    print(f"  Total time:          {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Total episodes:      {NUM_EPISODES:,}")
    print(f"  Goals reached:       {goals_reached} / {NUM_EPISODES}")
    print(f"  Best avg reward:     {best_avg_reward:.2f}")
    print(f"  Final avg reward:    {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Final WM loss:       {np.mean(world_losses[-100:]):.4f}")
    print(f"  Final curiosity:     {np.mean(curiosity_losses[-100:]):.4f}")
    print(f"  Model saved to:      neuron_agi_best.pt")
    print("═" * 65)

    # Save final model
    torch.save({
        'model_state': agent.state_dict(),
        'episode_rewards': episode_rewards,
        'world_losses': world_losses,
        'curiosity_losses': curiosity_losses,
        'policy_losses': policy_losses,
        'config': {
            'state_dim': STATE_DIM, 'action_dim': ACTION_DIM,
            'hidden_dim': HIDDEN_DIM, 'latent_dim': LATENT_DIM,
            'total_params': total_params, 'episodes': NUM_EPISODES,
        }
    }, 'neuron_agi_final.pt')

    return agent, episode_rewards, world_losses, curiosity_losses

if __name__ == "__main__":
    agent, rewards, wm_losses, cur_losses = train()
