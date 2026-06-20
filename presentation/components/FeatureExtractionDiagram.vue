<script setup lang="ts">
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'

const { $clicks } = useSlideContext()

// Synthetic worm-movement-like signal (normalized 0–1)
const signal = [
  0.42, 0.45, 0.48, 0.52, 0.55, 0.58, 0.62, 0.65, 0.68, 0.72,
  0.75, 0.78, 0.74, 0.7, 0.65, 0.6, 0.55, 0.5, 0.48, 0.52,
  0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8, 0.77, 0.73, 0.68,
  0.63, 0.58, 0.54, 0.5, 0.46, 0.5, 0.54, 0.58, 0.62, 0.66,
  0.7, 0.74, 0.78, 0.75, 0.71, 0.66, 0.61, 0.56, 0.52, 0.48,
  0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8,
]

const chartW = 280
const chartH = 56
const padX = 4
const padY = 6

const polyline = computed(() => {
  const n = signal.length
  return signal
    .map((y, i) => {
      const x = padX + (i / (n - 1)) * (chartW - 2 * padX)
      const py = chartH - padY - y * (chartH - 2 * padY)
      return `${x},${py}`
    })
    .join(' ')
})

const convLayers = [
  { label: 'Conv1d  k=7, s=2', channels: 32, width: 100 },
  { label: 'Conv1d  k=5, s=2', channels: 64, width: 50 },
  { label: 'Conv1d  k=3, s=2', channels: 128, width: 25 },
]

const embedBars = Array.from({ length: 16 }, (_, i) => 0.25 + 0.75 * Math.abs(Math.sin(i * 0.9 + 1.2)))
</script>

<template>
  <div class="feat-diagram">
    <div class="diagram-header">
      <span class="variate-tag">Example on variate <strong>X</strong></span>
      <span class="header-hint" style="font-style:italic">(identic structure for Y &amp; Speed)</span>
    </div>

    <!-- Step 1: input time series -->
    <div v-click="1" class="stage" :class="{ active: $clicks >= 1 }">
      <div class="stage-label">
        <span class="step-num">1</span>
        Input segment
        <span class="stage-meta">(1 × 900 frames)</span>
      </div>
      <div class="series-panel">
        <svg :viewBox="`0 0 ${chartW} ${chartH}`" class="series-chart" preserveAspectRatio="none">
          <defs>
            <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#3b82f6" />
              <stop offset="100%" stop-color="#8b5cf6" />
            </linearGradient>
            <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25" />
              <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.02" />
            </linearGradient>
          </defs>
          <polygon
            :points="`${padX},${chartH - padY} ${polyline} ${chartW - padX},${chartH - padY}`"
            fill="url(#areaGrad)"
          />
          <polyline
            :points="polyline"
            fill="none"
            stroke="url(#lineGrad)"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          <!-- sliding kernel window hint -->
          <rect x="72" y="4" width="28" height="48" rx="3" class="kernel-window" />
        </svg>
        <div class="axis-labels">
          <span>t = 0</span>
          <span>t = 900 frames</span>
        </div>
      </div>
    </div>

    <div v-click="2" class="connector" :class="{ active: $clicks >= 2 }">
      <span class="arrow">↓</span>
      <span>1D convolutions scan the signal</span>
    </div>

    <!-- Step 2: CNN feature maps / mask -->
    <div v-click="2" class="stage" :class="{ active: $clicks >= 2 }">
      <div class="stage-label">
        <span class="step-num">2</span>
        CNN feature maps
        <span class="stage-meta">(length shrinks, channels grow)</span>
      </div>
      <div class="cnn-panel">
        <div v-for="(layer, i) in convLayers" :key="layer.label" class="conv-row">
          <span class="conv-label">{{ layer.label }}</span>
          <div class="conv-track">
            <div
              class="conv-heat"
              :style="{
                width: `${layer.width}%`,
                opacity: 0.55 + i * 0.15,
              }"
            >
              <span
                v-for="c in 8"
                :key="c"
                class="heat-cell"
                :style="{ opacity: 0.35 + ((c + i) % 5) * 0.12 }"
              />
            </div>
            <span class="conv-ch">{{ layer.channels }} ch</span>
          </div>
        </div>
        <div class="pool-row">
          <span class="conv-label">MaxPool</span>
          <span class="pool-badge">→ single vector per channel</span>
        </div>
      </div>
    </div>

    <div v-click="3" class="connector" :class="{ active: $clicks >= 3 }">
      <span class="arrow">↓</span>
      <span>fully-connected layer</span>
    </div>

    <!-- Step 3: embedding -->
    <div v-click="3" class="stage" :class="{ active: $clicks >= 3 }">
      <div class="stage-label">
        <span class="step-num">3</span>
        Per-variate embedding
        <span class="stage-meta">(embed_dim = 128)</span>
      </div>
      <div class="embed-panel">
        <div class="embed-bars">
          <div
            v-for="(h, i) in embedBars"
            :key="i"
            class="embed-bar"
            :style="{ height: `${h * 100}%` }"
          />
        </div>
        <div class="embed-caption">
          dense feature vector — one per variate, per segment
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.feat-diagram {
  width: 100%;
  padding: 0;
  font-size: 0.78rem;
}

.diagram-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
  flex-wrap: wrap;
}

.variate-tag {
  font-weight: 500;
  color: var(--text-primary);
}

.repeat-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 2rem;
  padding: 0.1rem 0.45rem;
  border-radius: 9999px;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  color: white;
  font-weight: 700;
  font-size: 0.85rem;
}

.header-hint {
  font-size: 0.68rem;
  color: var(--text-muted);
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
  font-size: 0.75rem;
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
  font-size: 0.68rem;
}

.series-panel {
  background: linear-gradient(180deg, #f8fafc, #eff6ff);
  border-radius: 0.45rem;
  border: 1px solid #bfdbfe;
  padding: 0.35rem 0.5rem 0.25rem;
}

.series-chart {
  width: 100%;
  height: 2.75rem;
  display: block;
}

.kernel-window {
  fill: rgba(139, 92, 246, 0.12);
  stroke: #8b5cf6;
  stroke-width: 1.5;
  stroke-dasharray: 4 2;
}

.axis-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.62rem;
  color: var(--text-muted);
  margin-top: 0.15rem;
}

.connector {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.35rem;
  margin: 0.15rem 0;
  font-size: 0.68rem;
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

.cnn-panel {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.conv-row {
  display: grid;
  grid-template-columns: 6.5rem 1fr;
  align-items: center;
  gap: 0.4rem;
}

.conv-label {
  font-size: 0.65rem;
  font-family: var(--font-mono);
  color: var(--text-muted);
}

.conv-track {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.conv-heat {
  display: flex;
  gap: 1px;
  height: 1.25rem;
  border-radius: 0.25rem;
  overflow: hidden;
  background: linear-gradient(90deg, #dbeafe, #bbf7d0, #fde68a);
  border: 1px solid #93c5fd;
  min-width: 2rem;
}

.heat-cell {
  flex: 1;
  background: #1d4ed8;
  min-width: 2px;
}

.conv-ch {
  font-size: 0.62rem;
  color: var(--text-muted);
  white-space: nowrap;
}

.pool-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.15rem;
}

.pool-badge {
  font-size: 0.65rem;
  color: #059669;
  font-weight: 500;
}

.embed-panel {
  background: linear-gradient(135deg, #f5f3ff, #ede9fe);
  border: 1px solid #c4b5fd;
  border-radius: 0.45rem;
  padding: 0.5rem 0.65rem;
}

.embed-bars {
  display: flex;
  align-items: flex-end;
  gap: 3px;
  height: 2rem;
}

.embed-bar {
  flex: 1;
  min-width: 4px;
  border-radius: 2px 2px 0 0;
  background: linear-gradient(180deg, #8b5cf6, #4f46e5);
}

.embed-caption {
  margin-top: 0.35rem;
  font-size: 0.65rem;
  color: var(--text-muted);
  text-align: center;
}
</style>
