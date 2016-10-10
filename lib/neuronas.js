var jStat = require('jStat').jStat
var sigmoid = require('./math').sigmoid

module.exports = {
  forwardPropagation
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

  var hx = a3 > 0.5 ? 1 : 0
  return hx
}
