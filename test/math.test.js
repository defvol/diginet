var math = require('../lib/math')
var test = require('tape')

// Reduce to float precision 'p' on each element from matrix 'g'
var trim = (g, p) => g.map(row => row.map(x => parseFloat(x.toFixed(p))))

// Confirm that a value is within a range
var inRange = (v, range) => range[0] <= v && v <= range[1]

test('[math] sigmoid', function (t) {
  var r = [
    [0.73106, 0.88080],
    [0.95257, 0.98201]
  ]
  var z = [[1, 2], [3, 4]]
  var g = math.sigmoid(z)

  t.deepEqual(trim(g, 5), r, 'octave:1> 1.0 ./ (1.0 + exp(-z))')
  t.end()
})

test('[math] randomize initial weights', function (t) {
  var layer = { input: 400, output: 25 }
  var W = math.randomWeights(layer.input, layer.output)

  var within = x => inRange(x, [-0.12, 0.12])
  var validRange = W.map(row => row.every(within)).every(r => r)

  t.equal(W.length, 25, 'S_(l+1)')
  t.equal(W[0].length, 401, '(S_l)+1')
  t.true(validRange, 'every weight is within the range [-0.12, 0.12]')

  t.end()
})
