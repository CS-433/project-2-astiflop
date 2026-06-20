<script setup lang="ts">
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'

const { $clicks } = useSlideContext()

const steps = [
  {
    icon: '🎥',
    title: 'Raw recording',
    desc: 'Microscope video tracked frame-by-frame (0.5 fps)',
  },
  {
    icon: '📊',
    title: 'Tracking CSV',
    desc: 'X, Y coordinates + timestamps per worm',
  },
  {
    icon: '🧹',
    title: 'Cleaning',
    desc: 'Remove jumps, stitch trajectories, crop death',
  },
  {
    icon: '📐',
    title: 'Segmentation',
    desc: '900-frame sessions + Lifetime index',
  },
  {
    icon: '⚡',
    title: 'Features',
    desc: 'Speed, turning rate, normalized coords',
  },
  {
    icon: '🧠',
    title: 'Model tensor',
    desc: 'Shape (B, T, V, L) — batch, segments, variates, length',
  },
]

const tensorDetail = computed(() => {
  const click = $clicks.value
  if (click >= 6) {
    return {
      B: 'batch of worms',
      T: 'segments over lifespan',
      V: 'X, Y, Speed, Lifetime',
      L: '900 frames per segment',
    }
  }
  return null
})
</script>

<template>
  <div class="preprocessing-flow">
    <div class="pipeline">
      <template v-for="(step, i) in steps" :key="step.title">
        <div
          v-click="i + 1"
          class="pipeline-step"
          :class="{ active: $clicks >= i + 1 }"
        >
          <div class="step-icon">{{ step.icon }}</div>
          <div class="step-title">{{ step.title }}</div>
          <div class="step-desc">{{ step.desc }}</div>
        </div>
        <div v-if="i < steps.length - 1" v-click="i + 2" class="pipeline-arrow">
          →
        </div>
      </template>
    </div>

    <div v-click="6" class="tensor-box glass-card mt-6">
      <div class="text-sm font-semibold mb-2 text-slate-700">
        Final tensor fed to the model
      </div>
      <div class="tensor-visual">
        <div v-for="(dim, key) in { B: 'Batch', T: 'Time (segments)', V: 'Variates', L: 'Length' }" :key="key" class="tensor-dim">
          <span class="dim-letter">{{ key }}</span>
          <span class="dim-name">{{ dim }}</span>
        </div>
      </div>
      <div v-if="tensorDetail" v-click="7" class="tensor-detail mt-4">
        <div v-for="(desc, key) in tensorDetail" :key="key" class="detail-row">
          <code>{{ key }}</code>
          <span>{{ desc }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.preprocessing-flow {
  width: 100%;
}

.step-icon {
  font-size: 1.75rem;
  margin-bottom: 0.35rem;
}

.step-title {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.step-desc {
  font-size: 0.72rem;
  color: var(--text-muted);
  line-height: 1.35;
}

.pipeline-step {
  opacity: 0.35;
}

.pipeline-step.active {
  opacity: 1;
}

.tensor-visual {
  display: flex;
  gap: 0.5rem;
  justify-content: center;
  flex-wrap: wrap;
}

.tensor-dim {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0.75rem 1rem;
  background: linear-gradient(135deg, #eff6ff, #f0fdf4);
  border-radius: 0.5rem;
  border: 1px solid #bfdbfe;
  min-width: 90px;
}

.dim-letter {
  font-family: var(--font-mono);
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--accent);
}

.dim-name {
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
  text-align: center;
}

.tensor-detail {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
}

.detail-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
}

.detail-row code {
  background: #1e293b;
  color: white;
  padding: 0.15rem 0.4rem;
  border-radius: 0.25rem;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  min-width: 1.5rem;
  text-align: center;
}

.tensor-box {
  padding: 1.25rem;
}

@media (max-width: 768px) {
  .pipeline {
    flex-direction: column;
    align-items: center;
  }

  .pipeline-arrow {
    transform: rotate(90deg);
  }
}
</style>
