let sum l =
  let rec sum' acc = function
    | [] -> acc
    | x::tl -> sum' (acc + x) tl
  in sum' 0 l
