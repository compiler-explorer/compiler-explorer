actor Main
  var _env: Env

  new create(env: Env) =>
    _env = env

  fun square(num: I32): I32 =>
    num * num
