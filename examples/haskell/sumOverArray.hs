sumOverArray :: [Int] -> Int
sumOverArray (x:xs) = x + sumOverArray xs
sumOverArray [] =  0

main = return $ sumOverArray [1,3,4]
