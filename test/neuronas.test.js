var math = require('../lib/math')
var nnet = require('../lib/neuronas')
var test = require('tape')

var X
var L

test('setup simple XNOR net [2, 2, 1]', function (t) {
  L = [2, 2, 1]
  X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]

  t.end()
})

test('[neuronas] forward propagation for XNOR', function (t) {
  var Theta1 = [
    [-30, +20, +20],
    [+10, -20, -20]
  ]
  var Theta2 = [ [-10, 20, 20] ]
  var Theta = [Theta1, Theta2]
  var hx = 1337

  hx = nnet.forwardPropagation(L, Theta, X[0])
  t.true(hx >= 0.5, '(0 XNOR 0) = 1')
  hx = nnet.forwardPropagation(L, Theta, X[1])
  t.true(hx < 0.5, '(0 XNOR 1) = 0')
  hx = nnet.forwardPropagation(L, Theta, X[2])
  t.true(hx < 0.5, '(1 XNOR 0) = 0')
  hx = nnet.forwardPropagation(L, Theta, X[3])
  t.true(hx >= 0.5, '(1 XNOR 1) = 1')

  t.end()
})

test('[neuronas] cost function', function (t) {
  var y = [1, 0, 0, 1]
  var Theta = [
    math.randomWeights(2, 2),
    math.randomWeights(2, 1)
  ]

  var J = nnet.costFunction(Theta, X, y)
  t.true(J < 10, `compute the initial cost ${J.toFixed(4)}`)

  t.end()
})
