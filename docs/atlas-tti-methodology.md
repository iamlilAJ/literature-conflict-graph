# aigraph 假设生成方法论 — Atlas × test-time inference(2026-06-03 调研收敛版)

**一句话原则(本 session 反复验证):** Atlas 的价值是**验证器/选择器**,不是 prompt 内容;test-time compute 只在配一个**强验证器**时才有用。

## 证据基础(都做过实验)
- 浅 grounding-injection(把 Atlas bottleneck dump 进 prompt,M3)→ 无提升。
- bottleneck_open_q_alignment 塞进生成 prompt → per-hyp 质量零提升;价值在「选 anomaly」。
- best-of-N **单独**(无 Atlas)→ 几乎不比 baseline 好。
- **best-of-N + 强 Atlas prior-art oracle 过滤(M5)→ novelty 翻倍(50% vs 25%)、grounding 最高(83%)、排名最优。**(sub-agent web 验证,盲评)
- 弱 oracle(8155 method 名)→ 最差;强 oracle(4.2M 篇标题+摘要)→ 最好。**oracle 强度是决定性的。**
- refine loop(M6)→ 失败(过度改写,反而降 novelty)。
- 评估:绝对 rubric 会饱和(全 10/10);同模型自评不可信;**独立 web sub-agent 盲评**才是可信信号。

---

## 端到端流程(分阶段)

### Stage 0 — 语料 + 元数据(Atlas 当源)
`intern_atlas_loader`:论文来自 Atlas,白拿 `cited_by_count / influential_citation_count / venue_tier`。94–100% 覆盖,省 arxiv 429。**[已上线]**

### Stage 1 — 选种子(Atlas 当选择器)
- `bottleneck_open_q_alignment`:用 Atlas 第三方 bottleneck 选「哪些论文有真实、被别人指出的弱点」值得生成。**[已建]**
- 丢掉同篇假冲突 / X-on-X 退化 anomaly。**[已上线]**

### Stage 2 — 生成(test-time best-of-N,前瞻框架)
- **前瞻框架**(proven 4.00 vs 1.06):提新方法,不回溯解释冲突。
- **best-of-N 多样采样**(6 个 angle × 高温)。
- ~~graph-RAG grounding~~ **[已测,无效,删除]**:遍历演化图锁定「反复出现的 bottleneck」让 idea 针对它 —— sub-agent 盲评显示 graph-RAG 生成(M7)**比 baseline 更差**(50% already-exists,排名最末)。原因:稠密图上「反复出现的 bottleneck」=被研究最多的区域 → 针对它的 idea 最不新颖,直觉反了。这是第 3 次确认「Atlas 塞进生成 = 失败」(浅 grounding M3、refine M6、graph-RAG M7 全败)。**Stage 2 就用「前瞻框架 + best-of-N」,不加任何 Atlas 生成注入。**

### Stage 3 — Atlas 新颖性过滤(已验证的赢家 = M5)
对每个候选:
1. 查**强 Atlas oracle**——扫 4.2M 篇标题+摘要,找 distinctive token 命中的先验工作(polars lazy,2–5s/query)。
2. LLM 判:相对这些先验,novel 吗?**already-exists 的丢掉。**
3. 留下的才进选择。
→ **这是让 novelty 翻倍的那一招。**

### Stage 4 — 选择 + 交付
排序 + MMR 多样性 + Atlas overlap 相关性过滤(已上线)→ top-K。

### Stage 5 — 评估(怎么知道有没有变好)
- **不用** in-pipeline 同模型 judge(饱和 + 自评偏置)。
- **独立 web-enabled sub-agent,盲评**:每个 idea 去 arxiv/web 搜真实先验,分类 novel / incremental / already-exists + 查 grounding/test。**这是唯一可信的质量信号。**

---

## 诚实的边界
- **绝对质量天花板 ~5/10 = 生成模型(DeepSeek-V4-Flash)。** Atlas+TTI 提升的是 **novelty + grounding**,不是 raw quality。要破天花板得换更强生成模型。
- graph-RAG **只在成熟 cohort 有效**(图要密)。
- P2 方向误标的根在**冻结**抽取层,要真正修需 freeze thaw。

---

## 目前最该 ship 的一步(可立即落地)
把已上线的 P5 novelty gate(`idea_cascade._novelty_gate`,现在**只查被引那一篇**的摘要)→ **升级成查强 Atlas oracle(4.2M 篇)**。这就是 M5 配方进生产,drop-in、不需 thaw、已验证。改完用 sub-agent 复测确认线上 novelty 真涨。

**TL;DR 配方:** 前瞻框架生成 N 个 → 强 Atlas oracle(4.2M 篇)过滤掉已存在的 → 排序选 top-K;成熟论文再加 graph-RAG 锁定开放缺口;质量永远用独立 web sub-agent 盲评,不用自评。
