class MegaGreeter:
  names := []

  constructor name="World":
    names.add name

  say_hi:
    // Greet everyone individually!
    names.do: print "Hello $it!"
  say_bye:
    everyone := names.join ", "
    print "Bye $everyone, come back soon."

main:
  greeter := MegaGreeter
  greeter.say_hi
  greeter.say_bye

  greeter.names.add "Lars"
  greeter.names.add "Kasper"
  greeter.names.add "Rikke"
  greeter.say_hi
  greeter.say_bye
