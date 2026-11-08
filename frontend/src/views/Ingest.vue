<template>
  <div class="ingest-page">
    <el-card shadow="never" class="status-card">
      <template #header>
        <div class="toolbar">
          <div>
            <span class="panel-title">增量事件接入</span>
            <span class="muted" style="margin-left: 10px">两阶段纠错 · 实时归组 · 动态预警</span>
          </div>
          <div class="toolbar-right">
            <el-tag v-if="status.ready" type="success" effect="dark">索引就绪</el-tag>
            <el-tag v-else-if="status.building" type="warning" effect="dark">索引构建中…</el-tag>
            <el-tag v-else type="danger" effect="dark">索引不可用</el-tag>
            <el-button size="small" :icon="Refresh" @click="loadStatus" />
          </div>
        </div>
      </template>

      <el-row :gutter="16" class="stat-row">
        <el-col :xs="12" :md="6"><div class="stat"><div class="stat-num">{{ fmt(status.groups_indexed) }}</div><div class="stat-label">TF-IDF 索引组数</div></div></el-col>
        <el-col :xs="12" :md="6"><div class="stat"><div class="stat-num">{{ fmt(status.articles_indexed) }}</div><div class="stat-label">TF-IDF 索引报道</div></div></el-col>
        <el-col :xs="12" :md="6">
          <div class="stat">
            <div class="stat-num">
              <span v-if="nlStatus.index?.ready" class="lv"><span class="dot p2"></span>{{ fmt(nlStatus.index.groups) }} 组</span>
              <span v-else class="lv muted">未就绪</span>
            </div>
            <div class="stat-label">BGE 存量组索引（归组用）</div>
          </div>
        </el-col>
        <el-col :xs="12" :md="6">
          <div class="stat">
            <div class="stat-num">
              <el-tag v-if="llmEffective.enabled" type="success" size="small" effect="plain">
                {{ llmEffective.source }} · {{ llmEffective.model }}
              </el-tag>
              <el-tag v-else type="info" size="small" effect="plain">未配 LLM Key</el-tag>
            </div>
            <div class="stat-label">大模型初评（可选增强）</div>
          </div>
        </el-col>
      </el-row>
      <el-alert v-if="status.error" type="error" :title="String(status.error)" :closable="false" class="err" />
      <el-alert
        v-if="nlStatus.hint && status.ready && !status.error"
        type="info"
        title="当前使用轻量归组模式"
        description="语义索引尚未配置，事件接入会使用 TF-IDF 检索。两种归组方式的结果可能不同。"
        :closable="false"
        class="err"
      />
      <el-alert
        v-else-if="nlStatus.hint && !status.building"
        type="warning"
        title="语义索引尚未配置，轻量索引暂不可用"
        description="请检查接入索引状态，索引就绪后即可使用轻量归组。"
        :closable="false"
        class="err"
      />
    </el-card>

    <el-card shadow="never">
      <el-tabs v-model="tab">
        <el-tab-pane label="两阶段智能接入（自然语言）" name="nl">
          <el-collapse v-model="llmPanel" class="llm-collapse">
            <el-collapse-item name="llm">
              <template #title>
                <span class="llm-title-sm">大模型配置（可选，用于「初评」阶段）</span>
                <el-tag v-if="llmCfg.api_key" size="small" type="success" effect="dark" style="margin-left: 8px">已填写</el-tag>
                <el-tag v-else size="small" type="info" effect="plain" style="margin-left: 8px">未填写 · 自动跳过初评</el-tag>
                <el-tag v-if="serverLlm.enabled" size="small" type="success" effect="plain" style="margin-left: 6px">
                  服务器已配（{{ serverLlm.model }}）
                </el-tag>
              </template>
              <el-form label-width="90px" size="small" class="llm-form">
                <el-form-item label="接口地址">
                  <el-select v-model="llmCfg.base_url" filterable allow-create default-first-option style="width: 100%">
                    <el-option v-for="p in providers" :key="p.url" :label="`${p.name}（${p.url}）`" :value="p.url" />
                  </el-select>
                </el-form-item>
                <el-form-item label="模型">
                  <el-input v-model="llmCfg.model" placeholder="deepseek-chat" />
                </el-form-item>
                <el-form-item label="API Key">
                  <el-input v-model="llmCfg.api_key" type="password" show-password placeholder="sk-..." />
                </el-form-item>
                <el-form-item>
                  <el-button size="small" :loading="testing" @click="testLlm">测试连接</el-button>
                  <el-button size="small" @click="clearLlm">清除</el-button>
                  <span class="muted llm-note">仅保存在本浏览器 localStorage，随请求下发，不写入服务器磁盘</span>
                </el-form-item>
                <el-alert
                  v-if="llmTest" :closable="false"
                  :type="llmTest.ok ? 'success' : 'error'"
                  :title="llmTest.ok
                    ? `连接正常（${llmTest.latency_ms}ms，模型 ${llmTest.model}）`
                    : `连接失败：${llmTest.error}`"
                />
              </el-form>
            </el-collapse-item>
          </el-collapse>

          <div class="nl-row">
            <div class="nl-input">
              <el-input
                v-model="nlText" type="textarea" :rows="9" spellcheck="false"
                placeholder="输入一段新闻描述，例如：示例公司宣布投资建设新项目。"
              />
              <div class="nl-footer">
                <el-input v-model="nlSource" placeholder="来源（可选，如：用户提交）" style="width: 260px" size="small" />
                <div class="nl-actions">
                  <el-button size="small" @click="fillNlSample">示例</el-button>
                  <el-button size="small" @click="submitNl('matched')">示例：相关报道</el-button>
                  <el-button size="small" @click="submitNl('new')">示例：其他报道</el-button>
                  <el-button type="primary" size="small" :loading="nlLoading" @click="submitNl()">接入</el-button>
                </div>
              </div>
              <div class="muted" style="margin-top: 8px; font-size: 12px">
                流程：大模型初评（结构化+初步分类+初评级）→ 语义归组（可降级至轻量检索）→ 事件分类（可选）→ 可信度与影响评分 → 预警终评。
                在上方面板填入 API Key 即可启用初评；未填写或调用失败时自动跳过，由本地规则处理。
              </div>
            </div>
            <div class="nl-result">
              <el-empty v-if="!nlResult" description="提交自然语言新闻后展示两阶段结果" />
              <template v-else>
                <el-alert v-if="nlResult.error" type="error" :title="nlResult.error" :closable="false" />
                <el-alert v-if="nlResult.hint" type="warning" :title="nlResult.hint" :closable="false" class="err" />
                <div class="nl-head">
                  <span class="panel-title">处理结果</span>
                  <span class="muted">耗时 {{ nlResult.elapsed_ms }}ms</span>
                  <el-tag v-if="nlResult.status === 'matched'" type="success" size="small" effect="dark">归入存量组</el-tag>
                  <el-tag v-else-if="nlResult.status === 'new'" type="primary" size="small" effect="dark">新建事件组</el-tag>
                  <el-tag v-else-if="nlResult.status === 'duplicate'" type="info" size="small" effect="dark">重复</el-tag>
                  <el-tag v-else-if="nlResult.status === 'error'" type="danger" size="small" effect="dark">错误</el-tag>
                </div>

                <div v-if="nlResult.llm?.initial && Object.keys(nlResult.llm.initial).length" class="llm-card">
                  <div class="llm-title">大模型初评</div>
                  <el-descriptions :column="3" size="small" border>
                    <el-descriptions-item label="事件标题">{{ nlResult.llm.initial.article_title || '—' }}</el-descriptions-item>
                    <el-descriptions-item label="公司/行业">{{ nlResult.llm.initial.secu_abbr || nlResult.llm.initial.industry_name || '—' }}</el-descriptions-item>
                    <el-descriptions-item label="事件类型">{{ nlResult.llm.initial.event_name || '—' }}</el-descriptions-item>
                    <el-descriptions-item label="影响方向">{{ nlResult.llm.initial.event_impact_direction || '—' }}</el-descriptions-item>
                    <el-descriptions-item label="初评级"><b>{{ nlResult.llm.initial.initial_level || '—' }}</b></el-descriptions-item>
                    <el-descriptions-item label="初评理由">{{ nlResult.llm.initial.rationale || '—' }}</el-descriptions-item>
                  </el-descriptions>
                </div>
                <el-alert v-else-if="nlResult.llm?.error" type="warning" :title="`大模型调用失败：${nlResult.llm.error}`" :closable="false" />
                <el-alert v-else-if="nlResult.llm?.initial && Object.keys(nlResult.llm.initial).length === 0" type="info" title="未配置 API Key，已跳过模型初评" :closable="false" class="err" />

                <ol class="steps">
                  <li v-for="(s, i) in nlResult.steps || []" :key="s.step" class="step-item">
                    <span class="step-dot">{{ i + 1 }}</span>
                    <div class="step-body">
                      <div class="step-title">{{ s.label }}</div>
                      <div class="step-detail">{{ s.detail }}</div>
                    </div>
                  </li>
                </ol>

                <div v-if="nlResult.status !== 'error'" class="final-row">
                  <el-descriptions :column="3" size="small" border>
                    <el-descriptions-item label="归属组">
                      <b>{{ nlResult.matched_group_id || nlResult.new_group_id || nlResult.duplicate_group || '—' }}</b>
                    </el-descriptions-item>
                    <el-descriptions-item label="相似度">{{ nlResult.similarity ?? '—' }}</el-descriptions-item>
                    <el-descriptions-item label="预警终评">
                      <el-tag v-if="nlResult.alert.level" size="small" :type="levelType[nlResult.alert.level]" effect="dark">{{ nlResult.alert.level }}</el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="事件标题">{{ nlResult.article?.article_title || '—' }}</el-descriptions-item>
                    <el-descriptions-item label="事件类型">{{ nlResult.article?.event_name || '—' }}</el-descriptions-item>
                    <el-descriptions-item label="提示">{{ nlResult.message }}</el-descriptions-item>
                  </el-descriptions>
                  <div v-if="nlResult.alert.reason" class="reason muted">{{ nlResult.alert.reason }}</div>
                </div>
              </template>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="结构化 JSON 接入" name="json">
          <el-row :gutter="16">
            <el-col :xs="24" :md="11">
              <div class="submit-toolbar">
                <el-button size="small" @click="fillSample('match')">示例：单条报道</el-button>
                <el-button size="small" @click="fillSample('grow')">示例：多条报道</el-button>
                <el-button size="small" type="danger" plain @click="reset">重置演示数据</el-button>
              </div>
              <el-input
                v-model="input" type="textarea" :rows="14" spellcheck="false"
                class="json-input" placeholder='{"article_file_id": "...", "article_title": "...", ...}'
              />
              <div class="submit-footer">
                <span class="muted">支持单条对象、数组或 {"events": [...]}</span>
                <el-button type="primary" :loading="loading" @click="submitJson">接入</el-button>
              </div>
            </el-col>

            <el-col :xs="24" :md="13">
              <el-empty v-if="!results.length" description="尚无接入结果，提交事件或点示例填充" />
              <el-table v-else :data="results" size="small" border>
                <el-table-column prop="article_id" label="文章ID" width="120" />
                <el-table-column label="状态" width="80">
                  <template #default="{ row }">
                    <el-tag size="small" :type="statusType[row.status]">{{ statusText[row.status] }}</el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="归组 / 相似度" min-width="150">
                  <template #default="{ row }">
                    <div v-if="row.status === 'matched' || row.status === 'duplicate'">
                      <b>{{ row.matched_group_id }}</b>
                      <el-progress v-if="row.status === 'matched'" :percentage="Math.round((row.similarity || 0) * 100)" :stroke-width="6" class="sim" :color="(row.similarity || 0) >= 0.55 ? '#2fa67a' : '#e6a23c'" />
                    </div>
                    <span v-else-if="row.status === 'new'"><b>{{ row.matched_group_id }}</b>（最高候选 {{ row.similarity ?? 0 }}）</span>
                  </template>
                </el-table-column>
                <el-table-column label="预警" width="90">
                  <template #default="{ row }">
                    <el-tag v-if="row.alert?.level" size="small" :type="levelType[row.alert.level]" effect="dark">{{ row.alert.level }}</el-tag>
                    <span v-if="row.alert?.changed_reason" class="muted changed">{{ row.alert.changed_reason }}</span>
                  </template>
                </el-table-column>
                <el-table-column prop="message" label="说明" min-width="180" show-overflow-tooltip />
                <el-table-column type="expand">
                  <template #default="{ row }">
                    <div class="cand">
                      <div v-for="c in row.candidates || []" :key="c.group_id" class="cand-item">
                        <el-tag size="small" effect="plain">{{ c.group_id }}</el-tag>
                        <span class="sim-val">{{ (c.similarity || 0).toFixed(3) }}</span>
                        <span class="muted">{{ c.title }}</span>
                        <el-tag v-if="c.alert_level" size="small" type="info" effect="plain">{{ c.alert_level }}</el-tag>
                      </div>
                    </div>
                  </template>
                </el-table-column>
              </el-table>
            </el-col>
          </el-row>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { ingestPost, ingestReset, ingestStatus, ingestNl, ingestNlStatus, ingestNlTest } from '../api'

const tab = ref('nl')
const status = ref<any>({})
const nlStatus = ref<any>({})
const serverLlm = computed(() => nlStatus.value?.llm || {})

const nlText = ref('')
const nlSource = ref('')
const nlResult = ref<any>(null)
const nlLoading = ref(false)

// ── 大模型配置（仅存浏览器本地）──
const providers = [
  { name: 'DeepSeek', url: 'https://api.deepseek.com/v1' },
  { name: '通义千问', url: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
  { name: '智谱 GLM', url: 'https://open.bigmodel.cn/api/paas/v4' },
  { name: 'Kimi', url: 'https://api.moonshot.cn/v1' },
]
const LLM_KEY = 'event_radar_llm_cfg'
const llmCfg = ref({ base_url: 'https://api.deepseek.com/v1', model: 'deepseek-chat', api_key: '' })
const llmPanel = ref<string[]>([])
const llmTest = ref<any>(null)
const testing = ref(false)
try {
  const saved = JSON.parse(localStorage.getItem(LLM_KEY) || 'null')
  if (saved && typeof saved === 'object') llmCfg.value = { ...llmCfg.value, ...saved }
} catch { /* 忽略损坏的本地配置 */ }
watch(llmCfg, (v) => {
  llmTest.value = null  // 配置变化后旧测试结果失效
  localStorage.setItem(LLM_KEY, JSON.stringify(v))
}, { deep: true })

async function testLlm() {
  testing.value = true
  llmTest.value = null
  try {
    llmTest.value = await ingestNlTest({ llm: llmCfg.value })
    if (llmTest.value?.ok) ElMessage.success('连接正常')
  } catch (e: any) {
    llmTest.value = { ok: false, error: e?.response?.data?.detail || String(e) }
  } finally {
    testing.value = false
  }
}

function clearLlm() {
  llmCfg.value = { base_url: 'https://api.deepseek.com/v1', model: 'deepseek-chat', api_key: '' }
  llmTest.value = null
  localStorage.removeItem(LLM_KEY)
  ElMessage.info('已清除本地大模型配置')
}

// 有效的大模型配置：界面填写优先于服务端环境变量
const llmEffective = computed(() => {
  if (llmCfg.value.api_key) {
    return {
      enabled: true,
      source: llmTest.value?.ok ? '界面 · 已连通' : '界面',
      model: llmCfg.value.model,
    }
  }
  if (serverLlm.value?.enabled) {
    return { enabled: true, source: '服务器', model: serverLlm.value.model }
  }
  return { enabled: false, source: '', model: '' }
})

const input = ref('')
const results = ref<any[]>([])
const loading = ref(false)

const fmt = (n?: number) => (n ?? 0).toLocaleString()
const statusText: Record<string, string> = { matched: '已归组', new: '新事件组', duplicate: '重复', error: '错误' }
const statusType: Record<string, string> = { matched: 'success', new: 'primary', duplicate: 'info', error: 'danger' }
const levelType: Record<string, string> = { P0: 'danger', P1: 'warning', P2: '', P3: 'info' }

async function loadStatus() {
  status.value = await ingestStatus()
  try {
    nlStatus.value = await ingestNlStatus()
  } catch {
    nlStatus.value = {}
  }
}

// ── 两阶段（自然语言）──
function fillNlSample() {
  nlText.value = '示例文本：万科A公告投资建设绿色建筑项目，项目计划分阶段推进。本条内容为虚构样例。'
  nlSource.value = '用户提交'
}

function sampleByKind(kind: 'matched' | 'new') {
  if (kind === 'matched') {
    nlText.value = '示例文本：万科A公告投资建设绿色建筑项目，项目计划分阶段推进。本条内容为虚构样例。'
  } else {
    nlText.value = '某地新建智能温室大棚正式投入使用，预计年产蔬菜提升三成，当地农户已开始对接产销合作。'
  }
  nlSource.value = '演示供稿'
}

async function submitNl(kind?: 'matched' | 'new') {
  if (kind) sampleByKind(kind)
  const text = nlText.value.trim()
  if (!text) {
    ElMessage.warning('请先输入自然语言新闻描述')
    return
  }
  nlLoading.value = true
  try {
    const res = await ingestNl({
      text,
      article_source: nlSource.value,
      llm: llmCfg.value.api_key ? llmCfg.value : undefined,
    })
    nlResult.value = res
    if (res.ok) ElMessage.success(res.message)
    loadStatus()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '接入失败')
  } finally {
    nlLoading.value = false
  }
}

// ── 结构化 JSON ──
function fillSample(kind: 'match' | 'grow') {
  if (kind === 'match') {
    input.value = JSON.stringify({
      article_file_id: `DEMO-${Date.now()}`,
      article_title: '示例：万科A公告投资绿色建筑项目',
      article_publish_time: new Date().toISOString().slice(0, 19),
      article_source: '演示财经网',
      content: '示例文本：万科A公告投资建设绿色建筑项目，项目计划分阶段推进。本条内容为虚构样例。',
      event_name: '技术研发进展', trading_code: '000002', secu_abbr: '万科A',
    }, null, 2)
  } else {
    const t = Date.now()
    input.value = JSON.stringify({ events: [
      { article_file_id: `DEMO-${t}-1`, article_title: '某地新建智能温室大棚投入使用 年产蔬菜预计提升三成', article_publish_time: '2025-06-19T09:00:00', article_source: '地方新闻网', content: '智能温室大棚项目本周正式投产，预计年产蔬菜提升三成。', event_name: '产能扩张' },
      { article_file_id: `DEMO-${t}-2`, article_title: '智能温室大棚正式投产 年产蔬菜预计提升三成', article_publish_time: '2025-06-20T09:00:00', article_source: '农业周报', content: '智能温室大棚投产，年产蔬菜预计提升三成。', event_name: '产能扩张' },
      { article_file_id: `DEMO-${t}-3`, article_title: '智能温室大棚投产效果获验证 蔬菜年产预计提升三成', article_publish_time: '2025-06-21T09:00:00', article_source: '财经快讯', content: '智能温室大棚投产效果获验证，年产蔬菜预计提升三成。', event_name: '产能扩张' },
    ] }, null, 2)
  }
}

async function submitJson() {
  let payload: any
  try {
    payload = JSON.parse(input.value)
  } catch (e) {
    ElMessage.error('JSON 格式有误，请检查')
    return
  }
  loading.value = true
  try {
    const res = await ingestPost(payload)
    results.value = [...res.results].reverse().concat(results.value)
    ElMessage.success(`已处理 ${res.total} 条`)
    loadStatus()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '接入失败')
  } finally {
    loading.value = false
  }
}

async function reset() {
  try {
    await ElMessageBox.confirm('将删除全部增量接入记录，恢复到全量批处理基线，确定？', '重置演示数据', { type: 'warning' })
  } catch {
    return
  }
  const s = await ingestReset()
  status.value = s
  results.value = []
  ElMessage.success('已恢复基线')
}

onMounted(loadStatus)
</script>

<style scoped>
.status-card { margin-bottom: 16px; }
.toolbar { display: flex; justify-content: space-between; align-items: center; }
.toolbar-right { display: flex; align-items: center; gap: 8px; }
.stat-row { text-align: center; }
.stat { padding: 6px 0; }
.stat-num { font-size: 20px; font-weight: 600; }
.stat-num .lv { font-size: 14px; font-weight: 500; }
.stat-label { font-size: 12px; color: var(--el-text-color-secondary); margin-top: 4px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
.p2 { background: #2fa67a; }
.p3 { background: #409eff; }
.err { margin-top: 12px; }

.nl-row { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
.llm-collapse { margin-bottom: 14px; }
.llm-title-sm { font-weight: 600; }
.llm-form { max-width: 680px; margin-top: 6px; }
.llm-note { margin-left: 10px; font-size: 12px; }
.nl-input { flex: 1 1 360px; min-width: 300px; }
.nl-footer { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.nl-actions { display: flex; gap: 6px; flex-wrap: wrap; }
.nl-result { flex: 1 1 420px; min-width: 320px; max-width: 100%; }
.nl-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
.llm-card { margin-bottom: 12px; }
.llm-title { font-weight: 600; margin-bottom: 6px; color: #9a6dd7; }
.steps { list-style: none; margin: 10px 0 14px; padding: 0; }
.step-item { position: relative; display: flex; gap: 10px; padding-bottom: 12px; }
.step-item:not(:last-child)::before {
  content: '';
  position: absolute;
  left: 10px;
  top: 24px;
  bottom: 0;
  width: 1px;
  background: #e4e9f0;
}
.step-dot {
  flex: none;
  width: 21px;
  height: 21px;
  border-radius: 50%;
  background: #eef4ff;
  color: #2f80ed;
  font-size: 12px;
  font-weight: 600;
  display: grid;
  place-items: center;
  position: relative;
  z-index: 1;
}
.step-body { min-width: 0; flex: 1; }
.step-title { font-size: 13px; font-weight: 600; color: #263241; line-height: 21px; }
.step-detail { font-size: 12px; color: #5b6879; line-height: 1.6; margin-top: 2px; word-break: break-word; white-space: pre-wrap; }
.final-row :deep(.el-descriptions__label) { width: 90px; }
.reason { margin-top: 8px; font-size: 12px; }
.err { margin-bottom: 10px; }

.submit-toolbar { margin-bottom: 10px; display: flex; gap: 4px; flex-wrap: wrap; }
.json-input :deep(textarea) { font-family: Consolas, Menlo, monospace; font-size: 12px; }
.submit-footer { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; }
.sim { width: 120px; }
.sim-val { font-family: Consolas, monospace; margin: 0 8px; }
.changed { display: block; font-size: 11px; }
.cand { padding: 4px 12px; }
.cand-item { display: flex; align-items: center; gap: 8px; padding: 3px 0; }
@media (max-width: 1200px) {
  .nl-row { flex-direction: column; }
  .nl-input, .nl-result { min-width: 0; width: 100%; }
}
@media (max-width: 992px) {
  .stat-row { text-align: left; }
}
</style>
