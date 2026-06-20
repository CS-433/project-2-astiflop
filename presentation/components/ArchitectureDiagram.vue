<script setup lang="ts">
defineProps<{
  stage: 'feature' | 'attention' | 'temporal'
}>()
</script>

<template>
  <div v-if="stage === 'feature'" class="arch-diagram">
    <div class="arch-row">
      <div v-for="v in ['X', 'Y', 'Speed']" :key="v" class="arch-node input-node">
        <span class="node-label">{{ v }}</span>
        <span class="node-shape">(900 frames)</span>
      </div>
    </div>
    <div class="arch-connector">↓ parallel CNN branches ↓</div>
    <div class="arch-row">
      <div v-for="i in 3" :key="i" class="arch-node cnn-node">
        CNN {{ i }}
        <span class="node-sub">Conv1d × 3 + pool</span>
      </div>
    </div>
    <div class="arch-connector">↓ concat + linear ↓</div>
    <div class="arch-node output-node">
      Per-variate embeddings
      <span class="node-sub">(embed_dim)</span>
    </div>
  </div>

  <div v-else-if="stage === 'attention'" class="arch-diagram">
    <div class="arch-section">
      <div class="section-title">Variate Attention</div>
      <div class="arch-flow horizontal">
        <div class="arch-node">X emb</div>
        <div class="arch-node">Y emb</div>
        <div class="arch-node">Speed emb</div>
        <div class="arch-op">GatedAttention</div>
        <div class="arch-node output-node">Segment embedding</div>
      </div>
    </div>
    <div class="arch-divider" />
    <div class="arch-section">
      <div class="section-title">+ Time encoding</div>
      <div class="arch-flow horizontal">
        <div class="arch-node">Lifetime scalar</div>
        <div class="arch-op">Sinusoidal embed</div>
        <div class="arch-node output-node">Time-aware segment emb</div>
      </div>
    </div>
    <div class="arch-divider" />
    <div class="arch-section">
      <div class="section-title">Segment Attention (after temporal)</div>
      <div class="arch-flow horizontal">
        <div class="arch-node">T segments</div>
        <div class="arch-op">GatedAttention</div>
        <div class="arch-node output-node">Context vector</div>
      </div>
    </div>
  </div>

  <div v-else class="arch-diagram temporal">
    <div class="temporal-options">
      <div class="temporal-card">
        <div class="temporal-header bilstm">BiLSTM</div>
        <ul>
          <li>Bidirectional LSTM layers</li>
          <li>Captures long-range dependencies</li>
          <li>Orthogonality regularization</li>
          <li>Output: enriched segment sequence</li>
        </ul>
      </div>
      <div class="temporal-or">OR</div>
      <div class="temporal-card">
        <div class="temporal-header tcn">TCN</div>
        <ul>
          <li>Temporal Convolutional Network</li>
          <li>Dilated causal convolutions</li>
          <li>Exponential receptive field growth</li>
          <li>Often paired with Gaussian / Weibull heads</li>
        </ul>
      </div>
    </div>
    <div class="shared-output glass-card">
      <strong>Shared downstream:</strong> Segment attention → MLP regressor → remaining lifespan prediction
    </div>
  </div>
</template>

<style scoped>
.arch-diagram {
  width: 100%;
  padding: 0.5rem;
}

.arch-row {
  display: flex;
  justify-content: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin: 0.5rem 0;
}

.arch-node {
  padding: 0.65rem 0.85rem;
  border-radius: 0.5rem;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  text-align: center;
  font-size: 0.85rem;
  font-weight: 500;
  box-shadow: var(--shadow);
}

.input-node {
  border-color: #93c5fd;
  background: linear-gradient(135deg, #eff6ff, #dbeafe);
}

.cnn-node {
  border-color: #86efac;
  background: linear-gradient(135deg, #ecfdf5, #d1fae5);
  min-width: 100px;
}

.output-node {
  border-color: #c4b5fd;
  background: linear-gradient(135deg, #f5f3ff, #ede9fe);
  font-weight: 600;
}

.node-label {
  display: block;
  font-weight: 700;
  font-size: 1rem;
}

.node-shape,
.node-sub {
  display: block;
  font-size: 0.7rem;
  color: var(--text-muted);
  font-weight: 400;
  margin-top: 0.15rem;
}

.arch-connector {
  text-align: center;
  color: var(--accent);
  font-size: 0.8rem;
  font-weight: 600;
  margin: 0.35rem 0;
  opacity: 0.8;
}

.arch-section {
  margin: 0.5rem 0;
}

.section-title {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.35rem;
}

.arch-flow {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
  justify-content: center;
}

.arch-flow.horizontal .arch-node {
  font-size: 0.78rem;
  padding: 0.5rem 0.65rem;
}

.arch-op {
  padding: 0.4rem 0.6rem;
  background: #1e293b;
  color: white;
  border-radius: 0.4rem;
  font-size: 0.72rem;
  font-weight: 500;
}

.arch-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
  margin: 0.65rem 0;
}

.temporal-options {
  display: flex;
  align-items: stretch;
  gap: 1rem;
  justify-content: center;
}

.temporal-card {
  flex: 1;
  max-width: 280px;
  padding: 1rem;
  border-radius: 0.875rem;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  box-shadow: var(--shadow);
}

.temporal-header {
  font-weight: 700;
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
  padding-bottom: 0.35rem;
  border-bottom: 2px solid;
}

.temporal-header.bilstm {
  color: #2563eb;
  border-color: #93c5fd;
}

.temporal-header.tcn {
  color: #059669;
  border-color: #6ee7b7;
}

.temporal-card ul {
  margin: 0;
  padding-left: 1.1rem;
  font-size: 0.82rem;
}

.temporal-card li {
  margin-bottom: 0.25rem;
}

.temporal-or {
  display: flex;
  align-items: center;
  font-weight: 700;
  color: var(--text-muted);
  font-size: 0.9rem;
}

.shared-output {
  margin-top: 1rem;
  padding: 0.75rem 1rem;
  text-align: center;
  font-size: 0.85rem;
}
</style>
