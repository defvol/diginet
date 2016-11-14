var g = require('./math').sigmoid
var v = require('vectorious')
var log = Math.log

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
  var Theta = [
    v.Matrix(parameters[0]),
    v.Matrix(parameters[1])
  ]

  for (var t = 0; t < m; t++) {
    hx = forwardPropagation(null, Theta, X[t])
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
  var a1 = v.Matrix([[1].concat(x)])

  var z2 = Theta[0].multiply(a1.T)
  var a2 = v.Matrix([[1]].concat(g(z2).toArray()))

  var z3 = Theta[1].multiply(a2)
  var a3 = g(z3)

  return a3.toArray()
}
