// Defect Backfill Workflow
// Maps unmatched defects to features via Agent semantic analysis.
//
// Run via: Workflow({scriptPath: ".claude/skills/npu-risk-graph/workflows/defect-backfill.js", args: {batchCount: N}})
//   batchCount: printed by backfill_defect_features.py

export const meta = {
  name: 'defect-backfill',
  description: 'Map unmatched defects to features via Agent semantic analysis',
  phases: [{ title: 'Map' }],
}

const MAPPING_SCHEMA = {
  type: 'object',
  properties: {
    mappings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          defect_index: { type: 'integer' },
          features_affected: { type: 'array', items: { type: 'string' } },
        },
        required: ['defect_index', 'features_affected'],
      },
    },
  },
  required: ['mappings'],
}

const BATCH_COUNT = (args && args.batchCount) ? args.batchCount : 1

phase('Map')
const results = await parallel(
  Array.from({ length: BATCH_COUNT }, (_, i) => {
    const padded = String(i).padStart(2, '0')
    return () =>
      agent(
        `Read the file .sglang-risk/prompts/defect_backfill_batches/batch_${padded}.txt and follow ALL instructions in it exactly. Return ONLY valid JSON.`,
        { label: `batch-${padded}`, phase: 'Map', schema: MAPPING_SCHEMA }
      )
  })
)

return { mappings: results.filter(Boolean).flatMap((r) => r.mappings) }
