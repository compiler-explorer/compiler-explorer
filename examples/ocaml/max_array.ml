let max_array array =
  let max = ref 0 in
  for i = 0 to Array.length array do
    let el = array.(i) in
    if el > !max
    then max := el;
  done;
  !max
