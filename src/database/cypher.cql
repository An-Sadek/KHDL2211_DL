// Tạo pipeline
CALL gds.beta.pipeline.linkPrediction.create("pipe");

CALL gds.beta.pipeline.linkPrediction.addFeature('pipe', 'SAME_CATEGORY', {
  nodeProperties: ['id']
}) YIELD featureSteps
