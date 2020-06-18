const outputs = [];
const features = ["DropPosition", "Bounciness", "BallSize"];
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  // console.log(dropPosition, bounciness, size, bucketLabel);
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  // Write code here to analyze stuff

  const testSetSize = 100;
  const k = 10;

  // normalizedData = minMax(outputs, 3);
  _.range(0, 3).forEach((feature) => {
    const data = _.map(outputs, (row) => [row[feature], _.last(row)]);
    const [testSet, trainingSet] = splitDataSet(minMax(data, 1), testSetSize);
    const numberCorrect = testSet.filter((testPoint) => {
      return _.last(testPoint) === knn(trainingSet, _.initial(testPoint), k);
    }).length;

    console.log(
      `feature=${features[feature]},k = ${k}, Accuracy : ${
        numberCorrect / testSetSize
      }`
    );
  });
}

function knn(data, point, k) {
  return _.chain(data)
    .map((row) => [distance(_.initial(row), point), _.last(row)])
    .sortBy((row) => row[0])
    .slice(0, k)
    .countBy((row) => row[1])
    .toPairs()
    .sortBy((row) => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}
function distance(pointA, pointB) {
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5
  );
}

function splitDataSet(data, testCount) {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);
  return [testSet, trainingSet];
}

function minMax(data, featureCount) {
  const clonedData = _.cloneDeep(data);
  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map((row) => row[i]);
    const min = _.min(column);
    const max = _.max(column);

    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }
  return clonedData;
}
