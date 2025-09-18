// Tạo pipeline
CALL gds.beta.pipeline.linkPrediction.create("link_prediction");

// Thêm node
CALL gds.beta.pipeline.linkPrediction.addNodeProperty(
  "link_prediction",
  "fastRP",
  {mutateProperty: 'embedding',
  embeddingDimension: 256,
  randomSeed: 42
})

//
