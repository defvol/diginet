var jStat = require('jStat').jStat
var log = Math.log
var sigmoid = require('./math').sigmoid

module.exports = {
  costFunction,
  forwardPropagation
}

/**
 * Perform forward propagation and return the cost
 * @param {array} parameters unrolled in a vector
 * @param {array} X is our training set
 * @param {array} y are our labeled results
 * @returns {number} J is our current cost
 */
function costFunction (parameters, X, y) {
  var hx = 0
  var J = 0
  var m = X.length

  for (var t = 0; t < m; t++) {
    hx = forwardPropagation(null, parameters, X[t])
    J += y[t] * log(hx) - (1 - y[t]) * log(1 - hx)
  }

  J /= m

  return J
}

/**
 * Compute a forward propagation
 * @param {object} L holds the hyper parameters of the layers
 * @param {array} Theta unrolled in a vector
 * @param {array} x is a training example, represented as a vector of features
 * @returns {number} h(x) is a vector or number with the network prediction
 */
function forwardPropagation (L, Theta, x) {
  Theta = Theta.map(jStat)

  var a1 = jStat([1].concat(x)).transpose()

  var z2 = jStat.multiply(Theta[0], a1)
  var a2 = jStat([[1]].concat(sigmoid(z2)))

  var z3 = jStat.multiply(Theta[1], a2)
  var a3 = sigmoid(z3)

  var hx = a3
  return hx
}
