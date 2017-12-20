module Example where

sumOverArray :: [Int] -> Int
sumOverArray (x:xs) = x + sumOverArray xs
sumOverArray [] =  0
