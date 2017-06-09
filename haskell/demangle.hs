import Encoding
import Data.Char

main :: IO ()
main = do
   inp <- getContents
   let encoded = lines inp
       decoded = map Encoding.zDecodeString encoded in
                mapM_ putStrLn decoded
