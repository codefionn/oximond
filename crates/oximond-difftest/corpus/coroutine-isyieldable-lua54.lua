-- oximond-difftest repro profile=lua54 from=lua54 seed 2
print("o1#", coroutine.isyieldable())
print("o2#", coroutine.wrap(function() return coroutine.isyieldable() end)())
