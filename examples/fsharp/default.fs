module Program

let square num = num * num

[<EntryPoint>]
let main _ = 
    printfn "%d" (square 42)
    0
