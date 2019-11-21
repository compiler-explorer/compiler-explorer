module example

import StdInt

sumOverArray::[Int] -> Int
sumOverArray [x:xs] = x + sumOverArray xs
sumOverArray [] =  0

Start = sumOverArray [1,3,4]
