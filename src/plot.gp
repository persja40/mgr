set term png
set out 'g.png'
set title 'g(h)'
set xlabel "h"
plot 'g.data' using 1:2 with linespoints

