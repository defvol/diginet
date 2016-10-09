var math = require('../lib/math')
var test = require('tape')

// Reduce to float precision 'p' on each element from matrix 'g'
var trim = (g, p) => g.map(row => row.map(x => parseFloat(x.toFixed(p))))

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
