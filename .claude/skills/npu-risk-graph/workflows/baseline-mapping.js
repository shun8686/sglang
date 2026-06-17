// NPU Baseline Mapping Workflow
// Maps NPU test files to features via Agent semantic analysis.
//
// Run via: Workflow({scriptPath: ".claude/skills/npu-risk-graph/workflows/baseline-mapping.js", args: {batchCount: 16}})
//   batchCount: number of batch files in .sglang-risk/prompts/baseline_batches/
//               (printed by: python .claude/skills/npu-risk-graph/scripts/run_full.py)

export const meta = {
  name: 'npu-baseline-mapping',
  description: 'Map NPU test files to features via Agent semantic analysis',
  phases: [{ title: 'Map' }],
}

// Schema each Agent batch must conform to.  Keep in sync with:
//   apply_agent_baseline.py  (validation + merge)
//   baseline_prompts.py      (prompt generation)
const MAPPING_SCHEMA = {
  type: 'object',
  properties: {
    mappings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          test_file: {
            type: 'string',
            description:
              'Relative path from repo root, e.g. test/registered/ascend/basic_function/backends/test_npu_sampling_backend.py',
          },
          features_tested: {
            type: 'array',
            items: { type: 'string' },
            description:
              'Feature names exactly as listed in the batch prompt feature table',
          },
          quality_score: {
            type: 'integer',
            minimum: 1,
            maximum: 5,
            description:
              '1=smoke, 2=threshold assert, 3=multi-config/parametrize, 4=reference oracle (GSM8K/HF), 5=dual assert+oracle',
          },
          has_gsm8k_oracle: {
            type: 'boolean',
            description:
              'True if test uses GSM8K exact-match or GSM8KAscendMixin',
          },
          assertion_type: {
            type: 'string',
            enum: ['threshold', 'reference_comparison', 'mixed'],
            description: 'Type of correctness assertion used',
          },
          has_reference_oracle: { type: 'boolean' },
          rationale: {
            type: 'string',
            description:
              'Brief explanation of why these features are mapped (1-2 sentences)',
          },
        },
        required: [
          'test_file',
          'features_tested',
          'quality_score',
          'has_gsm8k_oracle',
          'assertion_type',
          'has_reference_oracle',
          'rationale',
        ],
      },
    },
  },
  required: ['mappings'],
}

// Read batchCount from args; default to 16 for backward compat.
// The caller should pass the number printed by run_full.py.
const BATCH_COUNT = (args && args.batchCount) ? args.batchCount : 16

phase('Map')
const results = await parallel(
  Array.from({ length: BATCH_COUNT }, (_, i) => {
    const padded = String(i).padStart(2, '0')
    return () =>
      agent(
        `Read the file .sglang-risk/prompts/baseline_batches/batch_${padded}.txt and follow ALL instructions in it exactly. Return ONLY valid JSON — no preamble, no markdown fences.`,
        { label: `batch-${padded}`, phase: 'Map', schema: MAPPING_SCHEMA }
      )
  })
)

return { mappings: results.filter(Boolean).flatMap((r) => r.mappings) }
