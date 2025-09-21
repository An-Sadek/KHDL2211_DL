// Tạo pipeline
CALL gds.beta.pipeline.linkPrediction.create("pipe");

// Thêm Node
CALL gds.beta.pipeline.linkPrediction.addNodeProperty('pipe', 'fastRP', {
  mutateProperty: 'embedding',
  embeddingDimension: 256,
  randomSeed: 42
})
