module example

import StdEnv

maxArray :: [Real] [Real] -> [Real]
maxArray list1 list2
 = [ (max e1 e2)
   \\ e1 <- list1
    & e2 <- list2
   ]

Start :: [Real]
Start = maxArray [1.3,4.1,6.6,2.3,3.5] [2.1,3.3,88.9,-5.0,3.56]
