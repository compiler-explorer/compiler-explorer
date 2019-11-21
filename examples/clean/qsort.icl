module example

import StdEnv

qsort :: [Int] -> [Int]
qsort [] = []
qsort [p:l] = (qsort [x \\ x<-l | x<p])
              ++[p]
              ++(qsort [x \\ x<-l | x>=p])

Start :: [Int]
Start = qsort [3,5,1,4,2]
