var v = require('vectorious')

module.exports = {
  randomWeights,
  sigmoid
}

const isNumber = x => x.constructor === Number

/**
 * Randomly initialize weights for a layer 'l'
 * Will count for bias units
 * @param {number} lin input connections on the layer
 * @param {number} lout output connections on the layer
 * @param {number} epsilon [default=0.12]
 * @returns {array} weights is a matrix of (output) x (1 + input)
 */
function randomWeights (lin, lout, epsilon) {
  epsilon = epsilon || 0.12
  var W = v.Matrix.random(lout, 1 + lin).map(t => t * 2 * epsilon - epsilon)
  return W.toArray()
}

/**
 * Compute the sigmoid function of z
 * https://en.wikipedia.org/wiki/Sigmoid_function
 * @param {number} z is a real number or matrix
 * @returns {number} g is the computed value or matrix
 */
function sigmoid (z) {
  return isNumber(z) ? 1.0 / (1.0 + Math.exp(-z)) : z.map(sigmoid)
}
