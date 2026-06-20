<script setup lang="ts">
import { useSlideContext } from '@slidev/client'

const { $clicks } = useSlideContext()

const variates = [
  { name: 'X', color: '#3b82f6', bars: [0.5, 0.8, 0.4, 0.7, 0.6, 0.9] },
  { name: 'Y', color: '#059669', bars: [0.7, 0.5, 0.9, 0.4, 0.8, 0.5] },
  { name: 'Speed', color: '#8b5cf6', bars: [0.6, 0.7, 0.5, 0.85, 0.45, 0.75] },
]

const weights = [
  { name: 'X', value: 0.42 },
  { name: 'Y', value: 0.31 },
  { name: 'Speed', value: 0.27 },
]
</script>

<template>
  <div class="attn-diagram">
    <!-- Step 1: three variate embeddings (visible on arrival) -->
    <div class="stage active">
      <div class="stage-label">
        <span class="step-num">1</span>
        Per-variate embeddings
        <span class="stage-meta">(B·T, 3, embed_dim)</span>
      </div>
      <div class="variate-row">
        <div v-for="v in variates" :key="v.name" class="variate-card">
          <span class="variate-name" :style="{ color: v.color }">{{ v.name }}</span>
          <div class="mini-bars">
            <div
              v-for="(h, i) in v.bars"
              :key="i"
              class="mini-bar"
              :style="{ height: `${h * 100}%`, background: v.color }"
            />
          </div>
        </div>
      </div>
    </div>

    <div v-click="1" class="connector" :class="{ active: $clicks >= 1 }">
      <span class="arrow">↓</span>
      <span>GatedAttention — one weight per variate</span>
    </div>

    <!-- Step 2: gated attention mechanism -->
    <div v-click="1" class="stage" :class="{ active: $clicks >= 1 }">
      <div class="stage-label">
        <span class="step-num">2</span>
        Gated attention
        <span class="stage-meta">Ilse et al. 2018</span>
      </div>
      <div class="gate-flow">
        <div class="gate-node input-node">x</div>
        <span class="gate-arrow">→</span>

        <div class="gate-split">
          <div class="gate-branch">
            <div class="gate-op tanh-op">V · x</div>
            <span class="gate-arrow-sm">→</span>
            <div class="gate-fn">tanh</div>
          </div>
          <div class="gate-branch">
            <div class="gate-op sig-op">U · x</div>
            <span class="gate-arrow-sm">→</span>
            <div class="gate-fn">σ</div>
          </div>
        </div>

        <span class="gate-arrow">→</span>
        <div class="gate-join">
          <span class="join-symbol">⊙</span>
        </div>
        <span class="gate-arrow">→</span>
        <div class="gate-node linear-node">w ·</div>
        <span class="gate-arrow">→</span>
        <div class="gate-node softmax-node">softmax</div>
        <span class="gate-arrow">→</span>
        <div class="gate-node output-node">α</div>
      </div>
    </div>

    <div v-click="2" class="connector" :class="{ active: $clicks >= 2 }">
      <span class="arrow">↓</span>
      <span>weighted sum, then add lifetime encoding</span>
    </div>

    <!-- Step 3: aggregation + time encoding -->
    <div v-click="2" class="stage" :class="{ active: $clicks >= 2 }">
      <div class="stage-label">
        <span class="step-num">3</span>
        Time encoding
      </div>

      <div class="agg-panel">
        <div class="agg-formula">
          <span v-for="(w, i) in weights" :key="w.name" class="agg-term">
            <span v-if="i > 0" class="agg-plus">+</span>
            <span class="agg-weight">{{ w.value.toFixed(2) }}</span>
            <span class="agg-var" :style="{ color: variates[i].color }">{{ w.name }}</span>
          </span>
          <span class="agg-eq">=</span>
          <span class="agg-result">seg_emb</span>
        </div>

        <div class="time-fusion">
          <div class="time-branch">
            <span class="time-label">Lifetime</span>
            <span class="time-arrow">→</span>
            <span class="time-op">sin / cos</span>
            <span class="time-arrow">→</span>
            <span class="time-emb">time_emb</span>
          </div>
          <div class="fusion-line">
            <span class="fusion-out">seg_emb </span>
            <span class="fusion-plus">+</span>
            <span class="time-emb">time_emb</span>
            <span class="fusion-plus">=</span>
            <span class="fusion-out">time-aware segment embedding</span>
            <span class="fusion-shape">(B, T, embed_dim)</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.attn-diagram {
  width: 100%;
  padding: 0;
  font-size: 0.76rem;
}

.stage {
  border-radius: 0.65rem;
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  padding: 0.45rem 0.6rem;
  opacity: 0.3;
  transform: translateY(4px);
  transition: opacity 0.35s ease, transform 0.35s ease, box-shadow 0.35s ease;
}

.stage.active {
  opacity: 1;
  transform: translateY(0);
  box-shadow: var(--shadow);
}

.stage-label {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-weight: 600;
  font-size: 0.74rem;
  margin-bottom: 0.4rem;
  color: var(--text-primary);
}

.step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.1rem;
  height: 1.1rem;
  border-radius: 9999px;
  background: var(--accent);
  color: white;
  font-size: 0.65rem;
  font-weight: 700;
  flex-shrink: 0;
}

.stage-meta {
  font-weight: 400;
  color: var(--text-muted);
  font-size: 0.66rem;
}

.connector {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.35rem;
  margin: 0.15rem 0;
  font-size: 0.66rem;
  font-weight: 600;
  color: var(--accent);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.connector.active {
  opacity: 0.85;
}

.arrow {
  font-size: 0.9rem;
}

.variate-row {
  display: flex;
  gap: 0.5rem;
  justify-content: center;
}

.variate-card {
  flex: 1;
  max-width: 5.5rem;
  padding: 0.35rem 0.4rem;
  border-radius: 0.45rem;
  background: linear-gradient(180deg, #f8fafc, #f1f5f9);
  border: 1px solid #e2e8f0;
  text-align: center;
}

.variate-name {
  display: block;
  font-weight: 700;
  font-size: 0.78rem;
  margin-bottom: 0.25rem;
}

.mini-bars {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 1.5rem;
}

.mini-bar {
  flex: 1;
  min-width: 3px;
  border-radius: 1px 1px 0 0;
  opacity: 0.85;
}

/* Gated attention flow — horizontal */
.gate-flow {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.2rem 0.15rem;
  padding: 0.15rem 0;
}

.gate-arrow {
  color: #94a3b8;
  font-size: 0.85rem;
  font-weight: 600;
  flex-shrink: 0;
}

.gate-arrow-sm {
  color: #94a3b8;
  font-size: 0.72rem;
  flex-shrink: 0;
}

.gate-node {
  padding: 0.25rem 0.5rem;
  border-radius: 0.4rem;
  font-size: 0.68rem;
  font-weight: 600;
  text-align: center;
  border: 1px solid var(--card-border);
  flex-shrink: 0;
}

.input-node {
  background: linear-gradient(135deg, #eff6ff, #dbeafe);
  border-color: #93c5fd;
  font-family: var(--font-mono);
}

.linear-node {
  background: #f1f5f9;
  font-family: var(--font-mono);
  font-size: 0.65rem;
}

.softmax-node {
  background: linear-gradient(135deg, #ecfdf5, #d1fae5);
  border-color: #6ee7b7;
  color: #047857;
  font-size: 0.65rem;
}

.output-node {
  background: linear-gradient(135deg, #f5f3ff, #ede9fe);
  border-color: #c4b5fd;
  font-family: var(--font-mono);
  color: #6d28d9;
}

.gate-split {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  position: relative;
  padding: 0.25rem 0.35rem;
  border-left: 2px solid #cbd5e1;
  border-right: 2px solid #cbd5e1;
}

.gate-split::before,
.gate-split::after {
  content: '';
  position: absolute;
  left: -2px;
  width: 0.5rem;
  height: 2px;
  background: #cbd5e1;
}

.gate-split::before {
  top: 28%;
}

.gate-split::after {
  bottom: 28%;
}

.gate-branch {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 0.15rem;
}

.gate-op {
  padding: 0.15rem 0.35rem;
  border-radius: 0.35rem;
  font-size: 0.6rem;
  font-family: var(--font-mono);
  font-weight: 500;
  border: 1px solid;
  white-space: nowrap;
}

.tanh-op {
  background: #eff6ff;
  border-color: #93c5fd;
  color: #1d4ed8;
}

.sig-op {
  background: #fff7ed;
  border-color: #fdba74;
  color: #c2410c;
}

.gate-fn {
  font-size: 0.62rem;
  font-weight: 700;
  padding: 0.1rem 0.35rem;
  border-radius: 0.3rem;
  background: #1e293b;
  color: white;
  white-space: nowrap;
}

.gate-join {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.join-symbol {
  font-size: 1rem;
  font-weight: 700;
  color: var(--accent);
  line-height: 1;
}

/* Aggregation + time */
.agg-panel {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.agg-formula {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 0.2rem 0.35rem;
  padding: 0.4rem;
  border-radius: 0.4rem;
  background: linear-gradient(135deg, #f8fafc, #eff6ff);
  border: 1px solid #bfdbfe;
  font-size: 0.7rem;
}

.agg-term {
  display: inline-flex;
  align-items: baseline;
  gap: 0.15rem;
}

.agg-plus {
  color: var(--text-muted);
  margin-right: 0.15rem;
}

.agg-weight {
  font-family: var(--font-mono);
  font-weight: 600;
  color: #6d28d9;
}

.agg-var {
  font-weight: 700;
}

.agg-eq {
  color: var(--text-muted);
  margin: 0 0.15rem;
}

.agg-result {
  font-family: var(--font-mono);
  font-weight: 600;
  color: #1e293b;
}

.time-fusion {
  padding: 0.35rem 0.45rem;
  border-radius: 0.4rem;
  background: linear-gradient(135deg, #fff7ed, #ffedd5);
  border: 1px solid #fdba74;
}

.time-branch {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.25rem;
  font-size: 0.66rem;
  margin-bottom: 0.3rem;
}

.time-label {
  font-weight: 600;
  color: #c2410c;
}

.time-arrow {
  color: var(--text-muted);
}

.time-op {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  padding: 0.1rem 0.3rem;
  background: white;
  border-radius: 0.25rem;
  border: 1px solid #fed7aa;
}

.time-emb {
  font-family: var(--font-mono);
  font-weight: 600;
  color: #c2410c;
}

.fusion-line {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.35rem;
  font-size: 0.68rem;
}

.fusion-plus {
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--accent);
}

.fusion-out {
  font-family: var(--font-mono);
  font-weight: 600;
  color: #1e293b;
}

.fusion-shape {
  font-size: 0.6rem;
  color: var(--text-muted);
}
</style>
