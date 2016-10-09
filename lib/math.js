var jStat = require('jStat').jStat

module.exports = {
  sigmoid
}

/**
 * Compute the sigmoid function of z
 * https://en.wikipedia.org/wiki/Sigmoid_function
 * @param {array} z is a matrix
 * @returns {array} g is a matrix
 */
function sigmoid (z) {
  var g = jStat(z).multiply(-1).exp().add(1).pow(-1)
  return jStat.multiply(g, 1)
}
