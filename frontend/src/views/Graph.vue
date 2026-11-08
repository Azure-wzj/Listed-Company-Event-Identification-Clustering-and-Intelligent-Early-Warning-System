<template>
  <div class="graph-page">
    <el-card shadow="never">
      <template #header>
        <div class="toolbar">
          <div class="title-wrap">
            <span class="panel-title">事件关系图谱</span>
            <span class="muted total-text">{{ summary }}</span>
          </div>
          <el-button :icon="Refresh" @click="load" />
        </div>

        <div class="filters">
          <div class="filter-item">
            <span class="filter-label">聚焦公司</span>
            <el-select
              v-model="company"
              filterable
              clearable
              placeholder="全部公司（概览）"
              class="w-220"
              @change="load"
            >
              <el-option
                v-for="c in companies"
                :key="c.code"
                :label="`${c.name}（${c.code}） · ${c.events} 事件`"
                :value="c.code"
              />
            </el-select>
          </div>

          <div class="filter-item">
            <span class="filter-label">事件类型</span>
            <el-select v-model="eventType" clearable filterable placeholder="全部类型" class="w-200" @change="load">
              <el-option v-for="t in eventTypes" :key="t.name" :label="`${t.name}（${t.count}）`" :value="t.name" />
            </el-select>
          </div>

          <div class="filter-item">
            <span class="filter-label">预警等级</span>
            <el-checkbox-group v-model="alertLevels" @change="load">
              <el-checkbox-button v-for="lv in ['P0', 'P1', 'P2', 'P3']" :key="lv" :value="lv">{{ lv }}</el-checkbox-button>
            </el-checkbox-group>
          </div>

          <div class="filter-item">
            <span class="filter-label">最小报道数</span>
            <el-input-number v-model="minArticles" :min="1" :max="20" size="small" @change="load" />
          </div>

          <div class="filter-item">
            <span class="filter-label">事件上限</span>
            <el-input-number v-model="limit" :min="10" :max="200" :step="10" size="small" @change="load" />
          </div>

          <div class="filter-item">
            <el-checkbox v-model="includeIndustries" @change="load">行业节点</el-checkbox>
            <el-checkbox v-model="hideIsolated" @change="load">隐藏孤立点</el-checkbox>
            <el-checkbox v-model="includeEventTypes" @change="load">事件类型节点</el-checkbox>
            <el-checkbox v-model="includeSources" @change="load">来源节点</el-checkbox>
          </div>
        </div>
      </template>

      <el-empty v-if="!hasData" description="当前筛选条件下没有可展示的关系" />

      <v-chart v-else :option="chart" class="graph-chart" autoresize />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { GraphChart } from 'echarts/charts'
import { LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { Refresh } from '@element-plus/icons-vue'
import { fetchGraph, fetchGraphCompanies, fetchTimelineStats } from '../api'

use([GraphChart, TooltipComponent, LegendComponent, CanvasRenderer])

type Company = { code: string; name: string; events: number; alerted: number }
type EventType = { name: string; count: number }

const companies = ref<Company[]>([])
const eventTypes = ref<EventType[]>([])
const company = ref<string>('')
const eventType = ref<string>('')
const alertLevels = ref<string[]>(['P0', 'P1', 'P2'])
const minArticles = ref(1)
const limit = ref(40)
const includeIndustries = ref(true)
const hideIsolated = ref(true)
const includeEventTypes = ref(false)
const includeSources = ref(false)

const chart = ref<any>({})
const summary = ref('')
const hasData = ref(true)

const palette: Record<string, string> = {
  event: '#2f80ed',
  company: '#2fa67a',
  industry: '#00a3a3',
  event_type: '#9a6dd7',
  source: '#c9a227',
}
const alertColor: Record<string, string> = {
  P0: '#d94b4b',
  P1: '#e8893c',
  P2: '#d8bc2e',
  P3: '#8fa3bf',
}

function nodeSize(n: any) {
  if (n.type === 'company') return 52
  if (n.type === 'industry') return 38
  if (n.type === 'event_type') return 36
  if (n.type === 'source') return 20
  return Math.max(22, Math.min(54, 18 + (n.article_count || 0) * 3))
}
function nodeColor(n: any) {
  if (n.type === 'event') return alertColor[n.alert_level] || palette.event
  return palette[n.type] || '#8a96a6'
}
const legendLabel: Record<string, string> = {
  event: '事件',
  company: '公司',
  industry: '行业',
  event_type: '事件类型',
  source: '来源',
}

async function loadMeta() {
  try {
    const [comp, stats] = await Promise.all([fetchGraphCompanies(50), fetchTimelineStats()])
    companies.value = comp.companies || []
    eventTypes.value = Object.entries(stats.by_event_type || {})
      .map(([name, count]) => ({ name, count: count as number }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 40)
    if (!company.value && companies.value.length) {
      company.value = companies.value[0].code // 默认聚焦事件最多的公司，保证图清晰
    }
  } catch {
    /* 元数据失败不阻塞主图 */
  }
}

async function load() {
  const res = await fetchGraph({
    company: company.value || undefined,
    event_type: eventType.value || undefined,
    alert_levels: alertLevels.value.length ? alertLevels.value.join(',') : undefined,
    min_article_count: minArticles.value,
    include_sources: includeSources.value,
    include_event_types: includeEventTypes.value,
    include_industries: includeIndustries.value,
    hide_isolated: hideIsolated.value,
    limit: limit.value,
    sort: 'articles',
  })

  const shown = res.shown_events || 0
  summary.value = `筛选命中 ${res.total_events || 0} 个事件 · 展示 ${shown} 个 · ${res.total_nodes || 0} 节点 / ${res.total_edges || 0} 关系`
  hasData.value = (res.nodes || []).length > 0

  const nodes = (res.nodes || []).map((n: any) => ({
    ...n,
    id: String(n.id),
    name: n.label || n.id,
    category: n.type || 'event',
    symbolSize: nodeSize(n),
    itemStyle: {
      color: nodeColor(n),
      borderColor: '#ffffff',
      borderWidth: n.type === 'company' ? 2 : 1,
      shadowBlur: n.type === 'company' ? 8 : 0,
      shadowColor: 'rgba(47,166,122,0.35)',
    },
    label: {
      show: true,
      fontSize: n.type === 'company' ? 12 : 10,
      fontWeight: n.type === 'company' ? 'bold' : 'normal',
      color: '#1f2d3d',
      overflow: 'truncate',
      width: n.type === 'company' ? 110 : 96,
    },
  }))
  const links = (res.edges || []).map((e: any) => ({
    source: String(e.source),
    target: String(e.target),
    value: e.relation || '',
    event_name: e.event_name,
    alert_level: e.alert_level,
    lineStyle: {
      color: e.alert_level && alertColor[e.alert_level] ? alertColor[e.alert_level] : '#b9c4d0',
      opacity: 0.45,
      width: Math.max(1, Math.min(3.5, Number(e.weight || 1) * 1.6)),
      curveness: 0.08,
    },
  }))
  const types = [...new Set(nodes.map((n: any) => n.category))]

  chart.value = {
    tooltip: {
      trigger: 'item',
      formatter: (p: any) => {
        if (p.dataType === 'edge') {
          const d = p.data
          return `${d.event_name || '关联'}<br/>预警：${d.alert_level || '-'}`
        }
        const d = p.data || {}
        if (d.category === 'company') {
          return `<b>${d.name}</b><br/>类型：公司<br/>代码：${d.id}`
        }
        if (d.category === 'industry') {
          return `<b>${d.name}</b><br/>类型：行业`
        }
        return `<b>${d.name}</b><br/>类型：${legendLabel[d.category] || d.category}` +
          `<br/>事件类型：${d.event_name || '-'}<br/>预警：${d.alert_level || '-'}` +
          `<br/>报道数：${d.article_count ?? '-'}<br/>可信度：${d.credibility_score ?? '-'}`
      },
    },
    legend: {
      top: 0,
      data: types,
      formatter: (name: string) => legendLabel[name] || name,
    },
    series: [{
      type: 'graph',
      layout: 'force',
      top: 34,
      roam: true,
      draggable: true,
      data: nodes,
      links,
      categories: types.map((t: any) => ({ name: t })),
      force: {
        repulsion: company.value ? 260 : 190,
        edgeLength: company.value ? [70, 130] : [50, 110],
        gravity: company.value ? 0.12 : 0.08,
      },
      label: { position: 'right' },
      lineStyle: { color: '#b9c4d0', curveness: 0.08 },
      emphasis: { focus: 'adjacency', lineStyle: { width: 3, opacity: 0.85 } },
    }],
  }
}

onMounted(async () => {
  await loadMeta()
  await load()
})
</script>

<style scoped>
.graph-page { padding: 0; }
.toolbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
.title-wrap { display: flex; align-items: baseline; }
.total-text { margin-left: 10px; font-size: 13px; font-weight: 400; }
.filters {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 18px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed var(--el-border-color-lighter);
}
.filter-item { display: flex; align-items: center; gap: 8px; }
.filter-label { font-size: 13px; color: var(--el-text-color-secondary); white-space: nowrap; }
.w-220 { width: 220px; }
.w-200 { width: 200px; }
.graph-chart { height: calc(100vh - 232px); min-height: 520px; }
@media (max-width: 900px) {
  .filters { flex-direction: column; align-items: flex-start; }
  .graph-chart { height: 620px; }
}
</style>
