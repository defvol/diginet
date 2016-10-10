var jStat = require('jStat').jStat

module.exports = {
  randomWeights,
  sigmoid
}

/**
 * Randomly initialize weights for a layer 'l'
 * @param {number} lin input connections on the layer
 * @param {number} lout output connections on the layer
 * @param {number} epsilon [default=0.12]
 * @returns {array} weights is a matrix of (output) x (1 + input)
 */
function randomWeights (lin, lout, epsilon) {
  epsilon = epsilon || 0.12
  var W = jStat.multiply(jStat.rand(lout, 1 + lin), 2 * epsilon)
  return jStat.subtract(W, epsilon)
}

/**
 * Compute the sigmoid function of z
 * https://en.wikipedia.org/wiki/Sigmoid_function
 * @param {array} z is a matrix
 * @returns {array} g is a matrix
 */
function sigmoid (z) {
  if (jStat.utils.isNumber(z)) return 1.0 / (1.0 + Math.exp(-z))

  return jStat.pow(jStat(z).multiply(-1).exp().add(1), -1)
}
