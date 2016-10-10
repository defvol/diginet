var nnet = require('../lib/neuronas')
var test = require('tape')

test('[neuronas] forward propagation for XNOR', function (t) {
  var X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]
  var L = [X.length, 2, 1]
  var Theta1 = [
    [-30, +20, +20],
    [+10, -20, -20]
  ]
  var Theta2 = [-10, 20, 20]
  var Theta = [Theta1, Theta2]

  var hx = 1337

  hx = nnet.forwardPropagation(L, Theta, X[0])
  t.equal(hx, 1, '(0 XNOR 0) = 1')
  hx = nnet.forwardPropagation(L, Theta, X[1])
  t.equal(hx, 0, '(0 XNOR 1) = 0')
  hx = nnet.forwardPropagation(L, Theta, X[2])
  t.equal(hx, 0, '(1 XNOR 0) = 0')
  hx = nnet.forwardPropagation(L, Theta, X[3])
  t.equal(hx, 1, '(1 XNOR 1) = 1')

  t.end()
})
